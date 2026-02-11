#!/usr/bin/env python3
"""
CloZ-AI Feedback Pipeline - Interactive Beta Test Script

이 스크립트는 실제 베타 테스트 환경처럼 사용자로부터 직접 쿼리와 피드백을 입력받아
추천-피드백 파이프라인의 전체 흐름을 테스트합니다.

실행: python -m src.feedback_pipeline.main
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Optional

# 패키지 경로 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.feedback_pipeline.agents.manager_agent import ManagerAgent, ManagerConfig
from src.feedback_pipeline.models.feedback import (
    FeedbackInput, 
    FeedbackScope, 
    OutfitSet, 
    ItemInfo, 
    ActionType
)
from src.feedback_pipeline.models.session import SessionStatus

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def get_multiple_scopes() -> List[FeedbackScope]:
    """사용자로부터 복수의 피드백 범위를 입력받음"""
    print("\n[범위 선택] 마음에 안 드는 부분을 모두 골라주세요 (번호 공백 구분):")
    print("1. 전체 (FULL)")
    print("2. 상의 (TOP)")
    print("3. 하의/바지 (BOTTOM)")
    print("4. 아우터 (OUTER)")
    
    choice = input("\n선택 (기본값 1): ").strip()
    if not choice:
        return [FeedbackScope.FULL]
    
    mapping = {
        "1": FeedbackScope.FULL,
        "2": FeedbackScope.TOP,
        "3": FeedbackScope.BOTTOM,
        "4": FeedbackScope.OUTER
    }
    
    indices = choice.split()
    scopes = [mapping[idx] for idx in indices if idx in mapping]
    return scopes if scopes else [FeedbackScope.FULL]

def mock_generate_outfit(query: str, session_id: str) -> OutfitSet:
    """테스트용 가상 코디 생성 (실제 환경에서는 생성팀 모듈이 호출됨)"""
    return OutfitSet(
        outfit_id=int(datetime.now().timestamp()),
        image_url="https://cloz-ai.s3.amazonaws.com/generated/demo_outfit.jpg",
        products=[
            ItemInfo(product_id=101, product_name=f"'{query}'에 어울리는 추천 상의", category_main="상의", category_sub="티셔츠"),
            ItemInfo(product_id=202, product_name=f"'{query}'에 어울리는 추천 바지", category_main="바지", category_sub="데님 팬츠")
        ]
    )

from datetime import datetime

def run_beta_test():
    clear_screen()
    print_header("CloZ-AI Feedback Pipeline - Beta Test Mode")
    
    # 초기화
    config = ManagerConfig(max_regenerate_count=2) # 베타 테스트를 위해 2회로 상향
    manager = ManagerAgent(config=config)
    user_id = "beta_tester"
    
    # 오리지널 쿼리 입력
    original_prompt = input("\n[Step 1] 어떤 코디를 추천받고 싶으신가요?\n입력: ").strip()
    if not original_prompt:
        original_prompt = "데일리 캐주얼 스타일 추천해줘"
        print(f"기본값 사용: {original_prompt}")
    
    # 가상 초기 코디 (생성팀 모듈 역할 시뮬레이션)
    current_outfit = mock_generate_outfit(original_prompt, "init")
    
    # 세션 시작
    session = manager.start_session(
        user_id=user_id,
        original_prompt=original_prompt,
        initial_outfit=current_outfit
    )
    
    print(f"\n[AI] '{original_prompt}'에 맞는 코디를 생성했습니다!")
    print(f"추천 착장: {current_outfit.products[0].product_name} + {current_outfit.products[1].product_name}")
    
    # 피드백 루프
    while True:
        print_header("사용자 피드백 단계")
        print(f"현재 코디: {[p.product_name for p in current_outfit.products]}")
        
        is_positive_input = input("\n이 코디가 마음에 드시나요? (y/n): ").lower().strip()
        is_positive = (is_positive_input == 'y')
        
        if is_positive:
            feedback = FeedbackInput(
                session_id=session.session_id,
                user_id=user_id,
                is_positive=True,
                current_outfit=current_outfit,
                feedback_text="좋아요!"
            )
            decision = manager.process_feedback(feedback)
            print(f"\n[AI] {decision.message}")
            manager.end_session(session.session_id, SessionStatus.COMPLETED)
            break
        else:
            # NO 피드백: 범위 및 상세 이유 입력
            scopes = get_multiple_scopes()
            feedback_text = input("\n[이유] 어떤 점이 마음에 안 드시나요?\n입력: ").strip()
            
            feedback = FeedbackInput(
                session_id=session.session_id,
                user_id=user_id,
                is_positive=False,
                current_outfit=current_outfit,
                feedback_text=feedback_text,
                feedback_scopes=scopes
            )
            
            print("\n[AI] 분석 중...")
            decision = manager.process_feedback(feedback)
            
            print(f"\n[AI 결정]: {decision.action.value}")
            print(f"[AI 메시지]: {decision.message}")
            
            if decision.action == ActionType.APPROVED:
                manager.end_session(session.session_id, SessionStatus.COMPLETED)
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
                            for prod in group_items[:3]:
                                if hasattr(prod, "to_dict"):
                                    prod = prod.to_dict()
                                print(f"    - {prod.get('product_name')} ({prod.get('brand')})")
                    if not printed:
                        recs = getattr(decision.buying_recommendations, "products", [])
                        for i, prod in enumerate(recs[:3]):
                            if hasattr(prod, "to_dict"):
                                prod = prod.to_dict()
                            print(f"- {prod.get('product_name')} ({prod.get('brand')})")
                        if not recs:
                            print("  (조건에 맞는 추천 상품이 없습니다)")
                manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
                break
            elif decision.action == ActionType.ASK_MORE:
                # 추가 질문 상황이면 루프를 돌며 다시 답변을 받음
                continue
            elif decision.action == ActionType.REGENERATE:
                print(f"\n[추출된 요구사항]: {decision.extracted_requirements}")
                print("\n새로운 코디를 찾는 중입니다...")
                # 가상 재생성 (실제로는 생성팀이 이 data를 받아야 함)
                current_outfit = mock_generate_outfit(decision.message, session.session_id)
                print(f"새 추천 착장: {current_outfit.products[0].product_name} + {current_outfit.products[1].product_name}")
                continue

    print_header("베타 테스트 종료")
    print(f"세션 ID: {session.session_id}")
    print("사용자 취향 데이터가 업데이트되었습니다.")

if __name__ == "__main__":
    try:
        run_beta_test()
    except KeyboardInterrupt:
        print("\n테스트를 중단합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
