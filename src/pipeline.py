"""
CloZ-AI 통합 파이프라인

생성 파이프라인(팀) + 피드백 파이프라인(내 코드)을 하나로 통합.

실행:
    python -m src.pipeline

전체 흐름:
    프롬프트 입력 (자연어)
    -> UnderstandModel이 JSON 파싱
    -> FashionRecommender가 옷장 임베딩과 비교 (카테고리별 top_k)
    -> OutfitPlanner가 최적 조합 선택
    -> VTONManager가 가상 피팅 이미지 생성
    -> 사용자 판단 -> 추천-피드백 루프

피드백 시나리오:
    - YES -> APPROVED (저장 종료)
    - NO + 옷장에 있음 -> REGENERATE (재생성)
    - NO + 옷장에 없음 -> BUYING (하바티 상품 추천)
    - NO + 피드백 불명확 -> ASK_MORE (재질문)
"""

import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger

from src.generation_pipeline import (
    CLIPEncoder,
    FashionRecommender,
    OutfitPlanner,
    VTONManager,
    UnderstandModel,
    load_config,
)
from src.feedback_pipeline.agents.manager_agent import ManagerAgent, ManagerConfig
from src.feedback_pipeline.models.feedback import (
    FeedbackInput,
    FeedbackScope,
    OutfitSet,
    ItemInfo,
    ActionType,
)
from src.feedback_pipeline.models.session import SessionStatus
from src.feedback_pipeline.interfaces.real_generation_model import RealGenerationModel


class CloZPipeline:
    """
    CloZ-AI 통합 파이프라인.

    생성 -> 피드백 -> 재생성 순환을 관리.
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: generation_model.yaml 경로.
                         None이면 configs/generation_model.yaml 사용.
        """
        # 설정 로드
        if config_path is None:
            config_path = project_root / "configs" / "generation_model.yaml"
        self.gen_config = load_config(str(config_path))

        logger.info("Initializing generation pipeline components...")

        # 생성 파이프라인 컴포넌트
        self.encoder = CLIPEncoder()
        self.recommender = FashionRecommender(self.encoder)
        self.planner = OutfitPlanner(self.encoder)
        self.vton = self._init_vton()
        self.understand_model = UnderstandModel()

        # 옷장/스타일 데이터 로드
        self.recommender.load_user_wardrobe()
        self.recommender.load_styles()

        # RealGenerationModel 생성
        self.generation_model = RealGenerationModel(
            understand_model=self.understand_model,
            encoder=self.encoder,
            recommender=self.recommender,
            planner=self.planner,
            vton=self.vton,
            config=self.gen_config,
        )

        # 피드백 파이프라인 (RealGenerationModel 주입)
        self.manager = ManagerAgent(
            config=ManagerConfig(max_regenerate_count=1),
            generation_model=self.generation_model,
        )

        logger.info("CloZ Pipeline initialized.")

    def _init_vton(self):
        """VTONManager 초기화 (실패 시 예외 발생)"""
        try:
            return VTONManager()
        except Exception as e:
            logger.error(f"VTONManager 초기화 실패: {e}")
            raise RuntimeError(f"VTONManager 초기화 실패: {e}") from e

    def run(self, user_prompt: str, user_id: str = "default") -> Dict[str, Any]:
        """
        전체 파이프라인 1회 실행.

        Args:
            user_prompt: 사용자 자연어 요청
            user_id: 사용자 ID

        Returns:
            {
                "session": SessionLog,
                "outfit": OutfitSet or None,
                "generation_result": GenerationResult,
            }
        """
        logger.info(f"Running pipeline for: '{user_prompt}'")

        gen_result = self.generation_model.generate(
            prompt=user_prompt,
            user_id=user_id,
        )

        if not gen_result.success:
            logger.error(f"Generation failed: {gen_result.message}")
            return {
                "session": None,
                "outfit": None,
                "generation_result": gen_result,
            }

        outfit = gen_result.outfits[0] if gen_result.outfits else None

        session = self.manager.start_session(
            user_id=user_id,
            original_prompt=user_prompt,
            initial_outfit=outfit,
        )

        return {
            "session": session,
            "outfit": outfit,
            "generation_result": gen_result,
        }

    def process_feedback(
        self,
        session_id: str,
        user_id: str,
        is_positive: bool,
        current_outfit: OutfitSet,
        feedback_text: str = "",
        feedback_scopes: Optional[List[FeedbackScope]] = None,
    ) -> Dict[str, Any]:
        """
        피드백 처리.

        REGENERATE 시 자동으로 재생성하여 새 코디를 반환.

        Returns:
            {
                "decision": ManagerDecision,
                "new_outfit": OutfitSet or None,
                "new_generation_result": GenerationResult or None,
            }
        """
        feedback = FeedbackInput(
            session_id=session_id,
            user_id=user_id,
            is_positive=is_positive,
            current_outfit=current_outfit,
            feedback_text=feedback_text,
            feedback_scopes=feedback_scopes or [FeedbackScope.FULL],
        )

        decision = self.manager.process_feedback(feedback)

        result = {
            "decision": decision,
            "new_outfit": None,
            "new_generation_result": None,
        }

        # REGENERATE -> 실제 재생성 실행
        if decision.action == ActionType.REGENERATE:
            regen_data = decision.payload.get("regenerate_data", {})
            structured_query = regen_data.get("structured_query", {})
            candidate_pool = regen_data.get("candidate_pool", {})
            refined_query = regen_data.get("refined_query", feedback_text)

            gen_result = self.generation_model.regenerate(
                original_result=self._last_generation_result,
                feedback=refined_query,
                constraints={
                    "structured_query": structured_query,
                    "candidate_pool": candidate_pool,
                },
                target_categories=decision.target_categories,
            )
            result["new_generation_result"] = gen_result

            if gen_result.success and gen_result.outfits:
                result["new_outfit"] = gen_result.outfits[0]
                self._last_generation_result = gen_result

        return result

    # ==================== Interactive CLI ====================

    def interactive_session(self, user_id: str = "default"):
        """
        대화형 CLI.

        기존 feedback_pipeline/main.py의 mock을 실제 파이프라인으로 대체.
        """
        self._clear_screen()
        self._print_header("CloZ-AI Pipeline")

        prompt = input("\n어떤 코디를 추천받고 싶으신가요?\n입력: ").strip()
        if not prompt:
            prompt = "캐주얼 데일리 코디 추천해줘"
            print(f"기본값 사용: {prompt}")

        print("\n코디를 생성하고 있습니다...")
        result = self.run(prompt, user_id)

        if not result["session"]:
            print(f"\n[오류] {result['generation_result'].message}")
            return

        session = result["session"]
        current_outfit = result["outfit"]
        self._last_generation_result = result["generation_result"]

        self._display_outfit(current_outfit, result["generation_result"])

        while True:
            self._print_header("피드백")
            self._display_outfit_summary(current_outfit)

            is_positive_input = input("\n이 코디가 마음에 드시나요? (y/n): ").lower().strip()
            is_positive = is_positive_input == "y"

            if is_positive:
                fb_result = self.process_feedback(
                    session_id=session.session_id,
                    user_id=user_id,
                    is_positive=True,
                    current_outfit=current_outfit,
                    feedback_text="",
                )
                print(f"\n{fb_result['decision'].message}")
                self.manager.end_session(session.session_id, SessionStatus.COMPLETED)
                break

            # NO 피드백
            scopes = self._get_scopes_from_user()
            feedback_text = input("\n어떤 점이 마음에 안 드시나요?\n입력: ").strip()

            if not feedback_text:
                print("피드백을 입력해주세요.")
                continue

            print("\n분석 중...")
            fb_result = self.process_feedback(
                session_id=session.session_id,
                user_id=user_id,
                is_positive=False,
                current_outfit=current_outfit,
                feedback_text=feedback_text,
                feedback_scopes=scopes,
            )

            decision = fb_result["decision"]
            print(f"\n[결정]: {decision.action.value}")
            print(f"[메시지]: {decision.message}")

            if decision.action == ActionType.APPROVED:
                self.manager.end_session(session.session_id, SessionStatus.COMPLETED)
                break

            elif decision.action == ActionType.BUYING:
                if decision.buying_recommendations:
                    print("\n[상품 추천 목록]:")
                    grouped = getattr(decision.buying_recommendations, "grouped_products", None)
                    printed = False
                    if grouped:
                        for group_name, group_items in grouped.items():
                            if not group_items:
                                continue
                            printed = True
                            print(f"  - [{group_name}]")
                            for i, prod in enumerate(group_items[:3], 1):
                                if hasattr(prod, "to_dict"):
                                    prod = prod.to_dict()
                                name = prod.get("product_name", "")
                                brand = prod.get("brand", "")
                                print(f"    {i}. {name} ({brand})")
                    if not printed:
                        recs = getattr(decision.buying_recommendations, "products", [])
                        for i, prod in enumerate(recs, 1):
                            if hasattr(prod, "to_dict"):
                                prod = prod.to_dict()
                            name = prod.get("product_name", "")
                            brand = prod.get("brand", "")
                            print(f"  {i}. {name} ({brand})")
                        if not recs:
                            print("  (조건에 맞는 추천 상품이 없습니다)")
                self.manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
                break

            elif decision.action == ActionType.ASK_MORE:
                continue

            elif decision.action == ActionType.REGENERATE:
                if fb_result["new_outfit"]:
                    current_outfit = fb_result["new_outfit"]
                    print("\n새로운 코디가 생성되었습니다!")
                    self._display_outfit(current_outfit, fb_result["new_generation_result"])
                else:
                    failed = fb_result.get("new_generation_result")
                    fail_msg = getattr(failed, "message", "원인 미상")
                    print(f"\n재생성에 실패했습니다. ({fail_msg})")
                continue

        self._print_header("세션 종료")
        print(f"세션 ID: {session.session_id}")

    # ==================== UI Helpers ====================

    @staticmethod
    def _clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def _print_header(title: str):
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    @staticmethod
    def _display_outfit(outfit: Optional[OutfitSet], gen_result=None):
        if not outfit:
            return
        print(f"\n--- 생성된 코디 ---")
        for p in outfit.products:
            print(f"  [{p.category_main}/{p.category_sub}] {p.product_name}")
        if outfit.image_url:
            print(f"  이미지: {outfit.image_url}")
        if gen_result and gen_result.metadata:
            score = gen_result.metadata.get("best_score", 0)
            if score:
                print(f"  점수: {score:.4f}")

    @staticmethod
    def _display_outfit_summary(outfit: Optional[OutfitSet]):
        if not outfit:
            return
        print(f"현재 코디: {[p.product_name for p in outfit.products]}")

    @staticmethod
    def _get_scopes_from_user() -> List[FeedbackScope]:
        print("\n마음에 안 드는 부분을 골라주세요 (번호 공백 구분):")
        print("1. 전체 (FULL)")
        print("2. 상의 (TOP)")
        print("3. 하의 (BOTTOM)")
        print("4. 아우터 (OUTER)")

        choice = input("\n선택 (기본값 1): ").strip()
        if not choice:
            return [FeedbackScope.FULL]

        mapping = {
            "1": FeedbackScope.FULL,
            "2": FeedbackScope.TOP,
            "3": FeedbackScope.BOTTOM,
            "4": FeedbackScope.OUTER,
        }

        # 공백/쉼표 모두 허용: "3 4", "3,4", "3, 4"
        indices = re.findall(r"[1-4]", choice)
        scopes = [mapping[idx] for idx in indices if idx in mapping]
        return scopes if scopes else [FeedbackScope.FULL]


if __name__ == "__main__":
    try:
        pipeline = CloZPipeline()
        pipeline.interactive_session()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
