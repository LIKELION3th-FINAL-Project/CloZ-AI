"""
Real Generation Model - 팀 생성 파이프라인을 래핑하는 실제 구현체

팀 코드(generation_pipeline)의 실제 추천/조합/VTON 흐름을
GenerationModelInterface로 래핑하여 ManagerAgent에서 사용.
"""

import unicodedata
import torch
from typing import Optional, Dict, Any, List, Set
from loguru import logger

from .generation_model import GenerationModelInterface, GenerationResult
from ..utils.detail_category import DETAIL_CAT_RULES, normalize_detail_category
from ..models import OutfitSet, ItemInfo
from ..adapters.main_adapter import convert_outfit_to_outfitset, load_classification_results
from src.generation_pipeline.fashion_engine.video_preview import GeminiVideoPreviewGenerator


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
        # 상의/하의만 사용 (아우터 배제)
        self.cat_map_planner = self.config.get("cat_to_db", {
            "상의": "shirt", "하의": "pant"
        })
        # NOTE: 실서비스 정책상 아우터는 제외하고 상의/하의만 사용.
        self.scope_keys = ("TOP", "BOTTOM")
        self.valid_scopes = set(self.scope_keys)
        self.scope_to_planner = {"TOP": "shirt", "BOTTOM": "pant"}
        self.category_alias = {
            "상의": "TOP",
            "shirt": "TOP",
            "top": "TOP",
            "하의": "BOTTOM",
            "바지": "BOTTOM",
            "pant": "BOTTOM",
            "bottom": "BOTTOM",
        }
        self.kor_to_scope = {
            "상의": "TOP",
            "하의": "BOTTOM",
            "바지": "BOTTOM",
            "shirt": "TOP",
            "top": "TOP",
            "pant": "BOTTOM",
            "bottom": "BOTTOM",
        }

        # classification results (옷 세부카테고리 매핑)
        self._classification_results = load_classification_results()

        self._canonical_styles = [
            "고프코어", "레트로", "로맨틱", "리조트", "미니멀",
            "스트릿", "스포티", "시크", "시티보이", "아웃도어",
            "오피스", "워크웨어", "캐주얼", "클래식", "프레피"
        ]
        self._style_alias_map = {
            "casual": "캐주얼",
            "캐주얼": "캐주얼",
            "classic": "클래식",
            "클래식": "클래식",
            "minimal": "미니멀",
            "minimalist": "미니멀",
            "미니멀": "미니멀",
            "street": "스트릿",
            "streetwear": "스트릿",
            "스트릿": "스트릿",
            "sporty": "스포티",
            "athletic": "스포티",
            "athleisure": "스포티",
            "스포티": "스포티",
            "retro": "레트로",
            "vintage": "레트로",
            "레트로": "레트로",
            "romantic": "로맨틱",
            "데이트": "로맨틱",
            "date": "로맨틱",
            "로맨틱": "로맨틱",
            "resort": "리조트",
            "vacation": "리조트",
            "리조트": "리조트",
            "chic": "시크",
            "시크": "시크",
            "cityboy": "시티보이",
            "city boy": "시티보이",
            "시티보이": "시티보이",
            "outdoor": "아웃도어",
            "아웃도어": "아웃도어",
            "gorpcore": "고프코어",
            "gorp": "고프코어",
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
            "formal": "오피스",
            "포멀": "오피스",
        }
        self.video_preview = GeminiVideoPreviewGenerator()

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

            model_result = self.understand_model.chat(prompt)
            agent_json = extract_json_format(model_result)

            if not agent_json:
                return GenerationResult(
                    success=False,
                    message="LLM 의도 파싱 실패",
                    metadata={"raw_response": model_result}
                )
            agent_json = self._normalize_initial_agent_json(agent_json, prompt)

            # constraints가 있으면 agent_json에 반영
            if constraints:
                agent_json = self._apply_constraints(agent_json, constraints)

            return self._run_generation_pipeline(agent_json, user_id)

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
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
            # 재생성 시에도 사용자별 outfit 경로/메타를 유지하기 위해
            # 이전 생성 결과에 담긴 user_id를 우선 사용한다.
            source_user_id = (original_result.metadata or {}).get("user_id", "default")

            # constraints에서 structured_query 추출 (QueryBuilder 결과)
            structured_query = {}
            candidate_pool = {}
            refined_query = ""
            use_candidate_pool = False
            if constraints:
                structured_query = constraints.get("structured_query", {})
                candidate_pool = constraints.get("candidate_pool", {})
                refined_query = constraints.get("refined_query", "")
                use_candidate_pool = bool(constraints.get("use_candidate_pool", True) and candidate_pool)
            if use_candidate_pool:
                candidate_pool = self._restrict_candidate_pool_to_targets(candidate_pool, target_categories)
            else:
                candidate_pool = {}

            # 원문 피드백은 부정/제외 의도 보존용, refined_query는 추천 품질 보강용으로 사용
            signal_text = " ".join(part for part in [feedback, refined_query] if part).strip()
            agent_feedback = refined_query or feedback

            # 피드백 기반 agent_json 재구성
            agent_json = self._build_agent_json_from_feedback(
                agent_feedback, structured_query, target_categories
            )
            # 아우터 제외 정책으로 외부 layer 재생성은 항상 비활성화
            include_outer = False
            keep_map = {}
            if self._should_apply_keep_map(target_categories):
                keep_map = self._build_keep_map(original_result, target_categories, include_outer)

            # 동일 파이프라인 재실행
            return self._run_generation_pipeline(
                agent_json,
                user_id=source_user_id,
                exclude_items=self._get_previous_items(original_result),
                exclude_map=self._build_exclude_map(original_result, target_categories),
                candidate_pool=candidate_pool,
                include_outer=include_outer,
                keep_map=keep_map,
            )

        except Exception as e:
            logger.exception(f"Regeneration failed: {e}")
            return GenerationResult(
                success=False,
                message=f"재생성 중 오류: {str(e)}",
            )

    def _run_generation_pipeline(
        self,
        agent_json: Dict[str, Any],
        user_id: str,
        exclude_items: Optional[List[str]] = None,
        exclude_map: Optional[Dict[str, set]] = None,
        candidate_pool: Optional[Dict[str, List[str]]] = None,
        include_outer: bool = True,
        keep_map: Optional[Dict[str, set]] = None,
    ) -> GenerationResult:
        """추천 -> 조합 -> 평가 -> VTON 실행"""

        # 실서비스에서 OUTER는 무효화되어 항상 False.
        include_outer = False

        self.recommender.load_user_wardrobe()
        self.recommender.load_styles()

        # candidate_pool 기반 재생성에서는 기본 top_k(예: 3)로 먼저 자르면
        # 유효 후보가 교집합 단계에서 사라질 수 있어 충분히 크게 조회한다.
        # 필터가 걸린 재생성에서는 후보폭을 넓혀두고 최종 랭킹 점수로 선택한다.
        filter_pool_has_values = any(v for v in (candidate_pool or {}).values())
        filter_exclude_has_values = any(v for v in (exclude_map or {}).values())
        filter_keep_has_values = any(v for v in (keep_map or {}).values())
        has_filters = filter_pool_has_values or filter_exclude_has_values or filter_keep_has_values
        rec_top_k = max(self.item_top_k, 20) if candidate_pool else (max(self.item_top_k, 10) if has_filters else self.item_top_k)
        recs_raw = self.recommender.recommend_from_agent(
            agent_json, top_k=rec_top_k
        )
        recs_scope = self._to_scope_recs(recs_raw)
        scope_order = ("TOP", "BOTTOM")

        def _scope_counts(scope_map: Dict[str, List[Dict[str, Any]]]) -> str:
            return ", ".join(f"{scope}={len(scope_map.get(scope, []))}" for scope in scope_order)

        def _scope_preview(scope_map: Dict[str, List[Dict[str, Any]]], scope: str, limit: int = 3) -> str:
            items = scope_map.get(scope, []) or []
            preview = []
            for item in items[:limit]:
                if not isinstance(item, dict):
                    preview.append(str(item))
                    continue
                preview.append(str(item.get("id") or item.get("path") or item.get("filename") or "n/a"))
            return ", ".join(preview) if preview else "-"

        logger.info(f"[REGEN][CAND] raw scope counts: {_scope_counts(recs_scope)} (top_k={rec_top_k})")
        logger.debug(
            "[REGEN][CAND] raw previews | "
            + ", ".join(
                f"{scope}=[{_scope_preview(recs_scope, scope)}]" for scope in scope_order
            )
        )

        if candidate_pool:
            pool_counts = {scope: len(candidate_pool.get(scope, []) or []) for scope in scope_order}
            logger.info(f"[REGEN][CAND] candidate_pool filter input: {pool_counts}")
        recs_scope = self._apply_candidate_pool(recs_scope, candidate_pool)
        logger.info(f"[REGEN][CAND] after candidate_pool: {_scope_counts(recs_scope)}")
        recs_scope = self._apply_category_exclusions(recs_scope, exclude_map)
        if exclude_map:
            exclude_counts = {scope: len(exclude_map.get(scope, set()) or set()) for scope in scope_order}
            logger.info(f"[REGEN][CAND] after exclusions ({exclude_counts}): {_scope_counts(recs_scope)}")
        recs_scope = self._apply_keep_map(recs_scope, keep_map)
        if keep_map:
            keep_counts = {scope: len(keep_map.get(scope, set()) or set()) for scope in scope_order}
            logger.info(f"[REGEN][CAND] after keep_map ({keep_counts}): {_scope_counts(recs_scope)}")
        recs_scope = self._ensure_keep_items_present(recs_scope, keep_map)
        logger.info(f"[REGEN][CAND] final scope counts: {_scope_counts(recs_scope)}")
        logger.debug(
            "[REGEN][CAND] final previews | "
            + ", ".join(
                f"{scope}=[{_scope_preview(recs_scope, scope)}]" for scope in scope_order
            )
        )
        # 아우터 비활성화 정책으로 상의/하의만 필수 스코프 유지
        required_scopes = ["TOP", "BOTTOM"]
        if not recs_scope or any(len(recs_scope.get(scope, [])) == 0 for scope in required_scopes):
            logger.warning(
                "재생성 후보 부족: "
                + ", ".join(f"{s}={len(recs_scope.get(s, []))}" for s in ["TOP", "BOTTOM"])
            )
            return GenerationResult(
                success=False,
                message="옷장에서 매칭되는 아이템을 찾을 수 없습니다.",
            )

        recs = {self.scope_to_planner[k]: v for k, v in recs_scope.items() if k in self.scope_to_planner}

        combos = self.planner.generate_combinations(
            recs,
            top_n=self.combination_top_k,
        )

        if not combos:
            return GenerationResult(
                success=False,
                message="충분한 카테고리 아이템이 없어 조합을 만들 수 없습니다.",
            )

        # context 기반 쿼리 임베딩 생성
        context_text = self._build_context_text(agent_json)
        avg_q_emb = self.encoder.encode_text(context_text).to(torch.float32)
        avg_q_emb /= (avg_q_emb.norm() + 1e-8)

        target_style = self._extract_target_style(agent_json)

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
        if not isinstance(best_outfit, dict):
            logger.error(f"[FATAL] best_outfit format invalid: {type(best_outfit)}")
            return GenerationResult(
                success=False,
                message="생성 결과 형식이 비정상입니다.",
            )
        if not best_outfit.get("combination"):
            logger.error(f"[FATAL] best_outfit has no combination: {best_outfit}")
            return GenerationResult(
                success=False,
                message="생성 조합 데이터가 비어 있습니다.",
            )

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

        output_video = self.video_preview.generate_from_image(output_image) if output_image else None

        outfit_set = convert_outfit_to_outfitset(
            best_outfit, output_image, self._classification_results
        )

        return GenerationResult(
                success=True,
                outfits=[outfit_set],
                message="코디가 생성되었습니다.",
                metadata={
                    "user_id": user_id,
                    "generator": "RealGenerationModel",
                    "best_score": best_outfit.get("final_score", 0),
                    "harmony_score": best_outfit.get("harmony_score", 0),
                    "vton_result": vton_result,
                    "video_preview_path": output_video,
                    "video_preview_enabled": self.video_preview.enabled,
                    "agent_json": agent_json,
            },
        )

    def _apply_constraints(
        self, agent_json: Dict, constraints: Dict
    ) -> Dict:
        """constraints를 agent_json에 반영"""
        if "colors" in constraints:
            if "color" not in agent_json:
                agent_json["color"] = {"value": [], "confidence": 0.8}
            agent_json["color"]["value"] = constraints["colors"]

        if "styles" in constraints:
            if "style" not in agent_json:
                agent_json["style"] = {"value": [], "confidence": 0.8}
            styles = constraints["styles"]
            if isinstance(styles, list):
                normalized = [self._canonicalize_style(v) for v in styles]
                agent_json["style"]["value"] = list(dict.fromkeys([s for s in normalized if s]))
            elif isinstance(styles, str):
                normalized = self._canonicalize_style(styles)
                agent_json["style"]["value"] = [normalized] if normalized else []
            else:
                agent_json["style"]["value"] = []

        return agent_json

    @staticmethod
    def _to_float(value: Any, default: float = 0.0, min_value: float = 0.0, max_value: float = 1.0) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, num))

    def _canonicalize_style(self, style: Any) -> str:
        if not isinstance(style, str):
            return ""
        normalized = unicodedata.normalize("NFC", style.strip()).replace("-", " ").replace("_", " ")
        if not normalized:
            return ""
        lowered = normalized.lower()
        mapped = self._style_alias_map.get(lowered)
        if mapped:
            return mapped
        if normalized in self._canonical_styles:
            return normalized
        for style_name in self._canonical_styles:
            if style_name in normalized:
                return style_name
        return ""

    def _extract_styles_from_keywords(self, texts: List[str]) -> List[str]:
        tokens = " ".join(str(t) for t in texts if isinstance(t, str)).lower()
        found: List[str] = []
        keyword_mapping = [
            ("데이트", "로맨틱"), ("데이트룩", "로맨틱"), ("date", "로맨틱"),
            ("운동", "스포티"), ("트레이닝", "스포티"), ("workout", "스포티"), ("sport", "스포티"),
            ("스트릿", "스트릿"), ("street", "스트릿"),
            ("미니멀", "미니멀"), ("minimal", "미니멀"),
            ("레트로", "레트로"), ("vintage", "레트로"),
            ("시크", "시크"), ("chic", "시크"),
            ("시티", "시티보이"), ("city", "시티보이"),
            ("리조트", "리조트"), ("vacation", "리조트"), ("resort", "리조트"),
            ("아웃도어", "아웃도어"), ("outdoor", "아웃도어"),
            ("사무실", "오피스"), ("office", "오피스"),
            ("워크웨어", "워크웨어"), ("workwear", "워크웨어"),
            ("프레피", "프레피"), ("preppy", "프레피"),
            ("고프코어", "고프코어"), ("gorpcore", "고프코어"),
            ("클래식", "클래식"), ("classic", "클래식"),
            ("오피스", "오피스"),
        ]
        for key, style in keyword_mapping:
            if key in tokens and style not in found:
                found.append(style)
        if not found:
            found.append("캐주얼")
        return found[:2]

    def _build_style_from_structured_query(self, structured_query: Dict[str, Any]) -> List[str]:
        style_scores: Dict[str, float] = {}

        def add(style_value: Any, score: float, decay: float = 1.0) -> None:
            style = self._canonicalize_style(style_value)
            if not style:
                return
            if score < 0:
                score = 0.0
            value = score * decay
            if value > style_scores.get(style, -1.0):
                style_scores[style] = value

        resolved_styles = structured_query.get("resolved_styles", {})
        if isinstance(resolved_styles, dict):
            primary = resolved_styles.get("primary")
            if isinstance(primary, dict):
                primary_score = self._to_float(primary.get("final_score"), None)
                if primary_score is None:
                    primary_score = (self._to_float(primary.get("location_score"), 0.0) + self._to_float(primary.get("mood_score"), 0.0)) / 2
                add(primary.get("style"), primary_score, 1.0)

            secondary = resolved_styles.get("secondary")
            if isinstance(secondary, dict):
                secondary_score = self._to_float(secondary.get("final_score"), None)
                if secondary_score is None:
                    secondary_score = (self._to_float(secondary.get("location_score"), 0.0) + self._to_float(secondary.get("mood_score"), 0.0)) / 2
                add(secondary.get("style"), secondary_score, 0.8)

        for candidate in structured_query.get("style_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            score = self._to_float(candidate.get("final_score"), 0.0)
            if score == 0.0:
                score = (self._to_float(candidate.get("location_score"), 0.0) + self._to_float(candidate.get("mood_score"), 0.0)) / 2
            add(candidate.get("style"), score)

        if not style_scores:
            mood_values = structured_query.get("mood", []) or []
            fallback_tokens = list(mood_values)
            loc = structured_query.get("location")
            if loc:
                fallback_tokens.append(loc)
            for style in self._extract_styles_from_keywords(fallback_tokens):
                add(style, 0.3)

        if not style_scores:
            return ["캐주얼"]

        ordered = sorted(style_scores.items(), key=lambda item: item[1], reverse=True)
        return [style for style, _ in ordered[:2]]

    def _resolve_style_confidence(self, structured_query: Dict[str, Any], styles: List[str]) -> float:
        resolved_styles = structured_query.get("resolved_styles", {})
        if isinstance(resolved_styles, dict):
            primary = resolved_styles.get("primary")
            if isinstance(primary, dict) and self._canonicalize_style(primary.get("style")):
                return 0.9
            secondary = resolved_styles.get("secondary")
            if isinstance(secondary, dict) and self._canonicalize_style(secondary.get("style")):
                return 0.85
        if structured_query.get("style_candidates"):
            return 0.76
        if styles and styles[0] != "캐주얼":
            return 0.78
        return 0.65

    def _build_agent_json_from_feedback(
        self,
        feedback: str,
        structured_query: Dict,
        target_categories: Optional[List[str]] = None,
    ) -> Dict:
        """피드백과 구조화된 정보로 agent_json 재구성"""
        query = structured_query or {}
        style_values = self._build_style_from_structured_query(query)
        style_confidence = self._resolve_style_confidence(query, style_values)

        requirements_en = query.get("requirements_en", [])
        if isinstance(requirements_en, str):
            requirements_en = [requirements_en]
        if not isinstance(requirements_en, list):
            requirements_en = []

        agent_json = {
            "style": {
                "value": style_values,
                "confidence": style_confidence,
            },
            "target_detail_cats": self._normalize_detail_cats(query.get("target_detail_cats")),
            "avoid_detail_cats": self._normalize_detail_cats(query.get("avoid_detail_cats")),
            "color": {
                "value": requirements_en,
                "confidence": 0.7,
            },
            "mood": {
                "value": query.get("mood", []),
                "confidence": 0.7,
            },
        }

        prefer_brightness = query.get("prefer_brightness")
        if prefer_brightness:
            agent_json["prefer_brightness"] = {"value": prefer_brightness, "confidence": 0.7}

        location = query.get("location", "")
        if location:
            location_conf = self._to_float(query.get("location_confidence"), 0.6, 0.0, 1.0)
            if location_conf <= 0.0:
                location_conf = 0.6
            agent_json["location"] = {"value": [location], "confidence": location_conf}

        time_val = query.get("time", "")
        if time_val:
            agent_json["season"] = {"value": [time_val], "confidence": 0.6}

        if feedback:
            agent_json["user_constraints"] = {"value": [feedback], "confidence": 0.8}

        return agent_json

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _normalize_detail_cats(self, raw: Any) -> List[str]:
        """
        세부카테고리 문자열을 추천기에서 비교 가능한 형태로 정규화.
        """
        normalized: List[str] = []
        for item in self._to_str_list(raw):
            text = unicodedata.normalize("NFKC", str(item)).strip()
            if not text:
                continue
            canonical = normalize_detail_category(text)
            if canonical in DETAIL_CAT_RULES:
                normalized.append(canonical)

        return sorted(dict.fromkeys(normalized))

    def _normalize_initial_agent_json(self, agent_json: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        understand_model 출력 스키마를 recommender 입력 스키마로 정규화.
        - style가 없으면 mood/prompt 기반으로 생성
        - 필드 형식이 dict/value 구조가 아니면 보정
        """
        normalized = dict(agent_json or {})
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
            style_values = self._extract_styles_from_keywords(
                normalized.get("mood", {}).get("value", []) + [prompt]
            )
            normalized["style"] = {"value": style_values, "confidence": 0.7}

        return normalized

    def _should_include_outer(
        self,
        feedback_text: str,
        target_categories: Optional[List[str]],
        structured_query: Dict[str, Any],
    ) -> bool:
        # 아우터 기능은 서비스 정책상 비활성화.
        return False

    @staticmethod
    def _basename(value: Any) -> str:
        text = str(value or "")
        return text.split("/")[-1] if text else ""

    def _item_name_tokens(self, item: Dict[str, Any]) -> Set[str]:
        path_name = self._basename(item.get("path", ""))
        item_name = self._basename(item.get("id", ""))
        return {token for token in [path_name, item_name] if token}

    def _normalize_identifier_tokens(self, raw: Any) -> Set[str]:
        """아이템 식별자 매칭에 사용할 토큰 집합 정규화."""
        if raw is None:
            return set()

        text = str(raw).strip()
        if not text:
            return set()

        normalized = {text, text.lower()}
        basename = self._basename(text)
        if basename:
            normalized.add(basename)
            normalized.add(basename.lower())
            stem = basename.rsplit(".", 1)[0]
            if stem:
                normalized.add(stem)
                normalized.add(stem.lower())

        return normalized

    def _collect_product_exclude_tokens(self, product) -> Set[str]:
        """옷장 아이템(피드백 기준)에서 추천 후보와 비교할 토큰 추출."""
        tokens: Set[str] = set()
        tokens.update(self._normalize_identifier_tokens(product.product_name))
        tokens.update(self._normalize_identifier_tokens(getattr(product, "product_image_path", None)))
        tokens.update(self._normalize_identifier_tokens(getattr(product, "product_id", None)))
        return tokens

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
                exclude_map[scope_cat].update(self._collect_product_exclude_tokens(product))

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

    def _build_context_text(self, agent_json: Dict) -> str:
        """agent_json에서 평가용 텍스트 생성"""
        parts = []
        if agent_json.get("color", {}).get("value"):
            parts.append(", ".join(agent_json["color"]["value"]))
        if agent_json.get("mood", {}).get("value"):
            parts.append(", ".join(agent_json["mood"]["value"]))
        if agent_json.get("location", {}).get("value"):
            parts.append(", ".join(agent_json["location"]["value"]))
        return " ".join(parts) if parts else "fashion outfit"

    def _extract_target_style(self, agent_json: Dict) -> str:
        """agent_json에서 타겟 스타일 추출"""
        styles = agent_json.get("style", {}).get("value", [])
        if styles:
            for style in styles:
                normalized = self._canonicalize_style(style)
                if normalized:
                    return normalized
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
                keep_map[scope_cat].update(self._collect_product_exclude_tokens(product))
        return keep_map

    def _should_apply_keep_map(self, target_categories: Optional[List[str]]) -> bool:
        """keep_map은 단일 파트 변경 요청일 때만 적용한다."""
        if self._is_full_scope_requested(target_categories):
            return False
        return len(self._extract_change_scopes(target_categories)) == 1

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
