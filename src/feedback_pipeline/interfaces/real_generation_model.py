"""
Real Generation Model - 팀 생성 파이프라인을 래핑하는 실제 구현체

팀 코드(generation_pipeline)의 실제 추천/조합/VTON 흐름을
GenerationModelInterface로 래핑하여 ManagerAgent에서 사용.
"""

import unicodedata
import re
import torch
from typing import Optional, Dict, Any, List, Set
from loguru import logger
import traceback

from .generation_model import GenerationModelInterface, GenerationResult
from ..models import OutfitSet, ItemInfo
from ..adapters.main_adapter import convert_outfit_to_outfitset, load_classification_results


class RealGenerationModel(GenerationModelInterface):
    """
    팀 생성 파이프라인을 래핑하는 GenerationModelInterface 구현체.

    생성 흐름:
        UnderstandModel.chat(prompt) -> LLM이 의도 파싱
        -> FashionRecommender.recommend_from_agent(parsed_json) -> 카테고리별 추천
        -> OutfitPlanner.generate_combinations(recs) -> 조합 생성
        -> OutfitPlanner.evaluate_outfits(combos, ...) -> 최적 조합 평가
        -> VTONManager.try_on(best, ...) -> 가상 피팅
        -> convert_outfit_to_outfitset() -> OutfitSet 변환
    """

    def __init__(
        self,
        understand_model,
        encoder,
        recommender,
        planner,
        vton,
        config: Dict[str, Any] = None,
    ):
        self.understand_model = understand_model
        self.encoder = encoder
        self.recommender = recommender
        self.planner = planner
        self.vton = vton
        self.config = config or {}

        # 설정에서 가중치 로드 (하드코딩 금지)
        self.item_top_k = self.config.get("item_top_k", 3)
        self.combination_top_k = self.config.get("combination_top_k", 3)
        self.overall_weight = self.config.get("overall_weight", 0.7)
        self.cat_map_planner = self.config.get("cat_to_db", {
            "상의": "shirt", "하의": "pant", "아우터": "outer"
        })
        self.scope_keys = ("TOP", "BOTTOM", "OUTER")
        self.valid_scopes = set(self.scope_keys)
        self.scope_to_planner = {"TOP": "shirt", "BOTTOM": "pant", "OUTER": "outer"}
        self.category_alias = {
            "상의": "TOP",
            "shirt": "TOP",
            "top": "TOP",
            "하의": "BOTTOM",
            "바지": "BOTTOM",
            "pant": "BOTTOM",
            "bottom": "BOTTOM",
            "아우터": "OUTER",
            "outer": "OUTER",
        }
        self.kor_to_scope = {
            "상의": "TOP",
            "하의": "BOTTOM",
            "바지": "BOTTOM",
            "아우터": "OUTER",
            "shirt": "TOP",
            "top": "TOP",
            "pant": "BOTTOM",
            "bottom": "BOTTOM",
            "outer": "OUTER",
        }

        # classification results (옷 세부카테고리 매핑)
        self._classification_results = load_classification_results()

    def generate(
        self,
        prompt: str,
        user_id: str,
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        프롬프트로부터 코디 생성.

        Args:
            prompt: 사용자 자연어 요청
            user_id: 사용자 ID
            constraints: 제약 조건 (색상, 핏 등)
            context: 추가 컨텍스트

        Returns:
            GenerationResult
        """
        try:
            from src.generation_pipeline.understand_model.understand_model import extract_json_format

            model_response = self.understand_model.initial_chat(prompt)
            model_response_json = extract_json_format(model_response)

            if not model_response_json:
                return GenerationResult(
                    success=False,
                    message="LLM 의도 파싱 실패",
                    metadata={"raw_response": model_response}
                )
            
            required_keys = ["time_context", "location", "mood"]
            missing_or_low_confidence = []
            
            for key in required_keys:
                field = model_response_json.get(key)
                value = field.get("value")
                confidence = field.get("confidence")
                
                if value is None or confidence < 0.7:
                    missing_or_low_confidence.append(key)
            
            for missing_key in missing_or_low_confidence:
                model_response = self.understand_model.request_additional_info_chat(model_response_json, missing_key)
                model_response_json = extract_json_format(model_response)
                
            
            model_response_json = self._normalize_initial_agent_json(model_response_json, prompt)

            # constraints가 있으면 agent_json에 반영
            if constraints:
                model_response_json = self._apply_constraints(model_response_json, constraints)

            return self._run_generation_pipeline(model_response_json, user_id)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                success=False,
                message=f"코디 생성 중 오류: {str(e)}",
            )

    def regenerate(
        self,
        original_result: GenerationResult,
        feedback: str,
        constraints: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        피드백 기반 재생성.

        Args:
            original_result: 이전 생성 결과
            feedback: 사용자 피드백 텍스트
            constraints: ManagerAgent에서 추출한 제약 조건
            target_categories: 변경 대상 카테고리

        Returns:
            GenerationResult
        """
        try:
            # constraints에서 structured_query 추출 (QueryBuilder 결과)
            structured_query = {}
            candidate_pool = {}
            if constraints:
                structured_query = constraints.get("structured_query", {})
                candidate_pool = constraints.get("candidate_pool", {})
            candidate_pool = self._restrict_candidate_pool_to_targets(candidate_pool, target_categories)

            # 피드백 기반 model_response_json 재구성
            model_response_json = self._build_agent_json_from_feedback(
                feedback, structured_query, target_categories
            )
            include_outer = self._should_include_outer(feedback, target_categories, structured_query)
            keep_map = self._build_keep_map(original_result, target_categories, include_outer)

            # 동일 파이프라인 재실행
            return self._run_generation_pipeline(
                model_response_json,
                user_id="default",
                exclude_items=self._get_previous_items(original_result),
                exclude_map=self._build_exclude_map(original_result, target_categories),
                candidate_pool=candidate_pool,
                include_outer=include_outer,
                keep_map=keep_map,
            )

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            logger.debug(traceback.format_exc())
            return GenerationResult(
                success=False,
                message=f"재생성 중 오류: {str(e)}",
            )

    def _run_generation_pipeline(
        self,
        model_response_json: Dict[str, Any],
        user_id: str,
        exclude_items: Optional[List[str]] = None,
        exclude_map: Optional[Dict[str, set]] = None,
        candidate_pool: Optional[Dict[str, List[str]]] = None,
        include_outer: bool = True,
        keep_map: Optional[Dict[str, set]] = None,
    ) -> GenerationResult:
        """추천 -> 조합 -> 평가 -> VTON 실행"""

        self.recommender.load_user_wardrobe()
        self.recommender.load_styles()

        # candidate_pool 기반 재생성에서는 기본 top_k(예: 3)로 먼저 자르면
        # 유효 후보가 교집합 단계에서 사라질 수 있어 충분히 크게 조회한다.
        rec_top_k = max(self.item_top_k, 20) if candidate_pool else self.item_top_k
        recs_raw = self.recommender.recommend_from_agent(
            model_response_json, top_k=rec_top_k
        )
        recs_scope = self._to_scope_recs(recs_raw)
        recs_scope = self._apply_candidate_pool(recs_scope, candidate_pool)
        recs_scope = self._apply_category_exclusions(recs_scope, exclude_map)
        recs_scope = self._apply_keep_map(recs_scope, keep_map)
        recs_scope = self._ensure_keep_items_present(recs_scope, keep_map)
        if not include_outer:
            recs_scope["OUTER"] = []

        required_scopes = ["TOP", "BOTTOM"] + (["OUTER"] if include_outer else [])
        if not recs_scope or any(len(recs_scope.get(scope, [])) == 0 for scope in required_scopes):
            logger.warning(
                "재생성 후보 부족: "
                + ", ".join(f"{s}={len(recs_scope.get(s, []))}" for s in ["TOP", "BOTTOM", "OUTER"])
            )
            return GenerationResult(
                success=False,
                message="옷장에서 매칭되는 아이템을 찾을 수 없습니다.",
            )

        recs = {self.scope_to_planner[k]: v for k, v in recs_scope.items() if k in self.scope_to_planner}

        combos = self.planner.generate_combinations(
            recs,
            top_n=self.combination_top_k,
            include_outer=include_outer,
        )

        if not combos:
            return GenerationResult(
                success=False,
                message="충분한 카테고리 아이템이 없어 조합을 만들 수 없습니다.",
            )

        # context 기반 쿼리 임베딩 생성
        context_text = self._build_context_text(model_response_json)
        avg_q_emb = self.encoder.encode_text(context_text).to(torch.float32)
        avg_q_emb /= (avg_q_emb.norm() + 1e-8)

        target_style = self._extract_target_style(model_response_json)

        best_outfits = self.planner.evaluate_outfits(
            combos,
            self.recommender.style_profiles,
            target_style,
            query_embedding=avg_q_emb,
            overall_weight=self.overall_weight,
        )

        if not best_outfits:
            return GenerationResult(
                success=False,
                message="조합 평가에 실패했습니다.",
            )

        best_outfit = best_outfits[0]
        user_body_image = self.config.get("user_body_image", "")

        if not self.vton:
            return GenerationResult(
                success=False,
                message="VTONManager가 초기화되지 않았습니다.",
            )

        if not user_body_image:
            return GenerationResult(
                success=False,
                message="user_body_image 설정이 필요합니다.",
            )

        vton_result = self.vton.try_on(
            user_body_image,
            best_outfit["combination"],
            f"output_{user_id}",
            0,
        )

        if not vton_result:
            return GenerationResult(
                success=False,
                message="VTON 이미지 생성에 실패했습니다.",
            )

        output_image = ""
        if vton_result and vton_result.get("final_path"):
            output_image = vton_result["final_path"]

        outfit_set = convert_outfit_to_outfitset(
            best_outfit, output_image, self._classification_results
        )

        return GenerationResult(
            success=True,
            outfits=[outfit_set],
            message="코디가 생성되었습니다.",
            metadata={
                "generator": "RealGenerationModel",
                "best_score": best_outfit.get("final_score", 0),
                "harmony_score": best_outfit.get("harmony_score", 0),
                "vton_result": vton_result,
                "model_response_json": model_response_json,
            },
        )

    def _apply_constraints(
        self, model_response_json: Dict, constraints: Dict
    ) -> Dict:
        """constraints를 agent_json에 반영"""
        if "colors" in constraints:
            if "color" not in model_response_json:
                model_response_json["color"] = {"value": [], "confidence": 0.8}
            model_response_json["color"]["value"] = constraints["colors"]

        if "styles" in constraints:
            if "style" not in model_response_json:
                model_response_json["style"] = {"value": [], "confidence": 0.8}
            model_response_json["style"]["value"] = constraints["styles"]

        return model_response_json

    def _build_agent_json_from_feedback(
        self,
        feedback: str,
        structured_query: Dict,
        target_categories: Optional[List[str]] = None,
    ) -> Dict:
        """피드백과 구조화된 정보로 model_response_json 재구성"""
        model_response_json = {
            "style": {
                "value": structured_query.get("mood", ["casual"]),
                "confidence": 0.8,
            },
            "color": {
                "value": structured_query.get("requirements_en", []),
                "confidence": 0.7,
            },
            "mood": {
                "value": structured_query.get("mood", []),
                "confidence": 0.7,
            },
        }

        location = structured_query.get("location", "")
        if location:
            model_response_json["location"] = {"value": [location], "confidence": 0.6}

        time_val = structured_query.get("time", "")
        if time_val:
            model_response_json["season"] = {"value": [time_val], "confidence": 0.6}

        if feedback:
            model_response_json["user_constraints"] = {"value": [feedback], "confidence": 0.8}

        return model_response_json

    def _normalize_initial_agent_json(self, model_response_json: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        understand_model 출력 스키마를 recommender 입력 스키마로 정규화.
        - style가 없으면 mood/prompt 기반으로 생성
        - 필드 형식이 dict/value 구조가 아니면 보정
        """
        normalized = dict(model_response_json or {})
        base_fields = ["color", "size_fit", "season", "location", "mood", "user_constraints", "user_requirements"]
        for field in base_fields:
            value = normalized.get(field, {"value": [], "confidence": 0.0})
            if not isinstance(value, dict):
                if isinstance(value, list):
                    value = {"value": value, "confidence": 0.5}
                elif value is None:
                    value = {"value": [], "confidence": 0.0}
                else:
                    value = {"value": [str(value)], "confidence": 0.5}
            value.setdefault("value", [])
            value.setdefault("confidence", 0.0)
            normalized[field] = value

        if "style" not in normalized or not isinstance(normalized.get("style"), dict):
            style_values = []
            mood_vals = normalized.get("mood", {}).get("value", []) or []
            for m in mood_vals:
                m_low = str(m).lower()
                if any(k in m_low for k in ["romantic", "데이트", "date"]):
                    style_values.append("로맨틱")
                elif any(k in m_low for k in ["sport", "운동", "athletic"]):
                    style_values.append("스포티")
                elif any(k in m_low for k in ["street", "스트릿"]):
                    style_values.append("스트릿")
                elif any(k in m_low for k in ["minimal", "미니멀"]):
                    style_values.append("미니멀")

            p = (prompt or "").lower()
            if re.search(r"운동|트레이닝|헬스|jog|run|workout|sport", p):
                style_values.append("스포티")
            if re.search(r"데이트|date", p):
                style_values.append("로맨틱")
            if re.search(r"여름|summer", p):
                style_values.append("리조트")

            if not style_values:
                style_values = ["캐주얼"]

            # 순서 유지 중복 제거
            dedup = list(dict.fromkeys(style_values))
            normalized["style"] = {"value": dedup, "confidence": 0.7}

        return normalized

    def _should_include_outer(
        self,
        feedback_text: str,
        target_categories: Optional[List[str]],
        structured_query: Dict[str, Any],
    ) -> bool:
        text = (feedback_text or "").lower()
        constraints = " ".join(structured_query.get("constraints", [])) if structured_query else ""
        merged = f"{text} {constraints}".lower()

        has_outer_token = ("아우터" in merged) or ("겉옷" in merged) or ("outer" in merged)
        has_remove_token = (
            ("제외" in merged) or ("빼" in merged) or ("없애" in merged) or ("remove" in merged) or ("without" in merged)
        )
        if has_outer_token and has_remove_token:
            return False

        scopes = {(s or "").upper() for s in (target_categories or [])}
        # 파트 선택 재생성에서는 OUTER를 명시적으로 변경/제외하지 않는 한 유지한다.
        if scopes and "FULL" not in scopes and "OUTER" not in scopes:
            return True
        return True

    @staticmethod
    def _basename(value: Any) -> str:
        text = str(value or "")
        return text.split("/")[-1] if text else ""

    def _item_name_tokens(self, item: Dict[str, Any]) -> Set[str]:
        path_name = self._basename(item.get("path", ""))
        item_name = self._basename(item.get("id", ""))
        return {token for token in [path_name, item_name] if token}

    def _is_full_scope_requested(self, target_categories: Optional[List[str]]) -> bool:
        requested = {(s or "").upper() for s in (target_categories or [])}
        return not requested or "FULL" in requested

    def _extract_change_scopes(self, target_categories: Optional[List[str]]) -> Set[str]:
        requested = {(s or "").upper() for s in (target_categories or [])}
        return {s for s in requested if s in self.valid_scopes}

    def _map_product_scope(self, category_main: Optional[str]) -> Optional[str]:
        raw = (category_main or "").strip().lower()
        mapped = self.category_alias.get(raw)
        if mapped:
            return mapped
        upper = (category_main or "").strip().upper()
        return upper if upper in self.valid_scopes else None

    def _filter_items_by_name_set(
        self,
        items: List[Dict[str, Any]],
        names: Set[str],
        *,
        include: bool,
    ) -> List[Dict[str, Any]]:
        if not names:
            return list(items) if not include else []

        filtered = []
        for item in items:
            matched = bool(self._item_name_tokens(item) & names)
            if (include and matched) or (not include and not matched):
                filtered.append(item)
        return filtered

    def _build_exclude_map(
        self,
        original_result: GenerationResult,
        target_categories: Optional[List[str]],
    ) -> Dict[str, set]:
        exclude_map = {scope: set() for scope in self.scope_keys}
        target_scopes = (
            set(self.scope_keys)
            if self._is_full_scope_requested(target_categories)
            else self._extract_change_scopes(target_categories)
        )

        for outfit in original_result.outfits:
            for product in outfit.products:
                scope_cat = self._map_product_scope(product.category_main)
                if scope_cat not in target_scopes:
                    continue
                if product.product_name:
                    exclude_map[scope_cat].add(product.product_name)

        return exclude_map

    def _apply_category_exclusions(
        self,
        recs_raw: Dict[str, List[Dict]],
        exclude_map: Optional[Dict[str, set]],
    ) -> Dict[str, List[Dict]]:
        if not exclude_map:
            return recs_raw

        filtered = {}
        for category, items in recs_raw.items():
            excludes = exclude_map.get(category, set())
            if not excludes:
                filtered[category] = items
                continue
            filtered[category] = self._filter_items_by_name_set(items, excludes, include=False)
        return filtered

    def _apply_keep_map(
        self,
        recs_raw: Dict[str, List[Dict]],
        keep_map: Optional[Dict[str, set]],
    ) -> Dict[str, List[Dict]]:
        if not keep_map:
            return recs_raw

        filtered = {}
        for category, items in recs_raw.items():
            keep_names = keep_map.get(category, set())
            if not keep_names:
                filtered[category] = items
                continue
            filtered[category] = self._filter_items_by_name_set(items, keep_names, include=True)
        return filtered

    def _ensure_keep_items_present(
        self,
        recs_raw: Dict[str, List[Dict]],
        keep_map: Optional[Dict[str, set]],
    ) -> Dict[str, List[Dict]]:
        if not keep_map:
            return recs_raw

        result = dict(recs_raw)
        for scope, names in keep_map.items():
            if not names:
                continue
            existing = result.get(scope, [])
            if existing:
                continue

            recovered = []
            for item in self.recommender.item_db.values():
                fname = self._basename(item.get("path", ""))
                if fname not in names:
                    continue
                recovered.append(item)
            if recovered:
                result[scope] = recovered[:1]
        return result

    def _apply_candidate_pool(
        self,
        recs_raw: Dict[str, List[Dict]],
        candidate_pool: Optional[Dict[str, List[str]]],
    ) -> Dict[str, List[Dict]]:
        if not candidate_pool:
            return recs_raw

        normalized_pool = {}
        for scope, ids in candidate_pool.items():
            scope_key = (scope or "").upper()
            if scope_key not in self.valid_scopes:
                continue
            normalized_pool[scope_key] = set(str(x) for x in (ids or []))

        if not normalized_pool:
            return recs_raw

        filtered = {}
        for cat, items in recs_raw.items():
            if cat not in normalized_pool:
                filtered[cat] = items
                continue
            allowed = normalized_pool.get(cat, set())
            filtered[cat] = self._filter_items_by_name_set(items, allowed, include=True)
        return filtered

    def _to_scope_recs(self, recs_raw: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        scope_recs = {scope: [] for scope in self.scope_keys}
        for k, items in recs_raw.items():
            scope = self.kor_to_scope.get(k, k if k in scope_recs else None)
            if scope:
                scope_recs[scope] = items
        return scope_recs

    def _build_context_text(self, model_response_json: Dict) -> str:
        """agent_json에서 평가용 텍스트 생성"""
        parts = []
        if model_response_json.get("color", {}).get("value"):
            parts.append(", ".join(model_response_json["color"]["value"]))
        if model_response_json.get("mood", {}).get("value"):
            parts.append(", ".join(model_response_json["mood"]["value"]))
        if model_response_json.get("location", {}).get("value"):
            parts.append(", ".join(model_response_json["location"]["value"]))
        return " ".join(parts) if parts else "fashion outfit"

    def _extract_target_style(self, model_response_json: Dict) -> str:
        """agent_json에서 타겟 스타일 추출"""
        style_aliases = {
            "casual": "캐주얼",
            "캐주얼": "캐주얼",
            "classic": "클래식",
            "클래식": "클래식",
            "minimal": "미니멀",
            "minimalist": "미니멀",
            "미니멀": "미니멀",
            "street": "스트릿",
            "스트릿": "스트릿",
            "sporty": "스포티",
            "athleisure": "스포티",
            "스포티": "스포티",
            "retro": "레트로",
            "vintage": "레트로",
            "레트로": "레트로",
            "romantic": "로맨틱",
            "feminine": "로맨틱",
            "로맨틱": "로맨틱",
            "resort": "리조트",
            "vacation": "리조트",
            "리조트": "리조트",
            "chic": "시크",
            "시크": "시크",
            "cityboy": "시티보이",
            "city boy": "시티보이",
            "city_boy": "시티보이",
            "시티보이": "시티보이",
            "outdoor": "아웃도어",
            "gorpcore": "고프코어",
            "gorp": "고프코어",
            "아웃도어": "아웃도어",
            "고프코어": "고프코어",
            "office": "오피스",
            "business casual": "오피스",
            "오피스": "오피스",
            "workwear": "워크웨어",
            "utility": "워크웨어",
            "워크웨어": "워크웨어",
            "preppy": "프레피",
            "ivy": "프레피",
            "프레피": "프레피",
            "formal": "포멀",
        }
        styles = model_response_json.get("style", {}).get("value", [])
        if styles:
            style = unicodedata.normalize("NFC", str(styles[0])).strip()
            style_key = " ".join(style.lower().replace("_", " ").replace("-", " ").split())
            return style_aliases.get(style_key, style_aliases.get(style, style))
        return "캐주얼"

    def _get_previous_items(self, result: GenerationResult) -> List[str]:
        """이전 결과에서 아이템 목록 추출 (재생성 시 제외용)"""
        return [
            product.product_name
            for outfit in result.outfits
            for product in outfit.products
            if product.product_name
        ]

    def _build_keep_map(
        self,
        original_result: GenerationResult,
        target_categories: Optional[List[str]],
        include_outer: bool,
    ) -> Dict[str, set]:
        if self._is_full_scope_requested(target_categories):
            return {}

        change_scopes = self._extract_change_scopes(target_categories)
        keep_scopes = set(self.scope_keys) - change_scopes
        if not include_outer and "OUTER" in keep_scopes:
            keep_scopes.remove("OUTER")

        keep_map = {scope: set() for scope in self.scope_keys}
        for outfit in original_result.outfits:
            for product in outfit.products:
                scope_cat = self._map_product_scope(product.category_main)
                if scope_cat not in keep_scopes:
                    continue
                if product.product_name:
                    keep_map[scope_cat].add(product.product_name)
        return keep_map

    def _restrict_candidate_pool_to_targets(
        self,
        candidate_pool: Optional[Dict[str, List[str]]],
        target_categories: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        if not candidate_pool:
            return {}
        if self._is_full_scope_requested(target_categories):
            return candidate_pool

        change_scopes = self._extract_change_scopes(target_categories)
        if not change_scopes:
            return candidate_pool
        return {k: v for k, v in candidate_pool.items() if k in change_scopes}
