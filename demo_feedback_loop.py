#!/usr/bin/env python3
"""
CloZ-AI Feedback Pipeline Demo

입력:
- 생성된 코디 이미지 디렉토리 경로
- 각 파트별 상품 정보 (JSON)

로직:
1. Yes → APPROVED → 세션 종료
2. No → ManagerAgent 판단:
   - ASK_MORE: 추가 질문 → 답변 후 재판단
   - REGENERATE: 옷장에서 대안 검색 (1회만)
   - BUYING: 외부 상품 추천 → 세션 종료
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "embedding_generator"))

try:
    from generate_fashionclip_embeddings import FashionCLIPEmbedder
except ImportError:
    class FashionCLIPEmbedder:
        def __init__(self, **kwargs):
            print("[WARN] FashionCLIPEmbedder not found. Using mock.")
        def embed_text(self, text):
            return [0.0] * 512

from feedback_pipeline.agents.manager_agent import ManagerAgent, ManagerConfig
from feedback_pipeline.checkers.embedding_wardrobe_checker import EmbeddingWardrobeChecker
from feedback_pipeline.checkers.embedding_buying_trigger import EmbeddingBuyingTrigger
from feedback_pipeline.models.feedback import (
    FeedbackInput,
    FeedbackScope,
    OutfitSet,
    ItemInfo,
    ActionType
)
from feedback_pipeline.models.session import SessionStatus


# ============================================================
# Shared Resources (Singleton)
# ============================================================
_shared_resources = {}

def get_shared_resources():
    """Load models once and reuse"""
    if not _shared_resources:
        print("\n[SYSTEM] Loading AI models...")
        try:
            embedder = FashionCLIPEmbedder(use_fp16=False)
            wardrobe_checker = EmbeddingWardrobeChecker(embedder=embedder)
            buying_trigger = EmbeddingBuyingTrigger(embedder=embedder)

            _shared_resources['embedder'] = embedder
            _shared_resources['wardrobe_checker'] = wardrobe_checker
            _shared_resources['buying_trigger'] = buying_trigger
            print("[SYSTEM] Models loaded successfully.\n")
        except Exception as e:
            print(f"[ERROR] Failed to load resources: {e}")
            return None
    return _shared_resources


def load_classification_data() -> Dict[str, Dict]:
    """Load classification_results.json as lookup dict"""
    path = Path("classification_results.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert to dict keyed by input path
            return {item["input"]: item for item in data}
    return {}


# ============================================================
# Input Helpers
# ============================================================
def parse_outfit_input(
    image_dir: str,
    products_json: str,
    classification_lookup: Dict[str, Dict]
) -> OutfitSet:
    """
    Parse outfit input from directory path and products JSON.

    Args:
        image_dir: Directory containing the generated outfit image
        products_json: JSON string with product info, e.g.:
            [
                {"category_main": "상의", "image_path": "closet/tops/tops_1.jpeg"},
                {"category_main": "바지", "image_path": "closet/bottoms/bottoms_1.jpeg"}
            ]

        Or simplified format (image paths only, comma-separated):
            closet/tops/tops_1.jpeg, closet/bottoms/bottoms_1.jpeg

        classification_lookup: Dict from classification_results.json

    Returns:
        OutfitSet with populated products
    """
    outfit_id = int(datetime.now().timestamp())
    image_url = f"file://{os.path.abspath(image_dir)}"

    products_data = []

    # Try JSON format first
    try:
        products_data = json.loads(products_json)
    except json.JSONDecodeError:
        # Try simplified comma-separated format
        if products_json.strip():
            paths = [p.strip() for p in products_json.split(',')]
            for path in paths:
                if path:
                    # Auto-detect category from path
                    cat = "상의"
                    if "bottom" in path.lower():
                        cat = "바지"
                    elif "outer" in path.lower():
                        cat = "아우터"
                    products_data.append({"category_main": cat, "image_path": path})

    products = []
    for i, prod in enumerate(products_data):
        category_main = prod.get("category_main", "상의")
        image_path = prod.get("image_path", "")

        # Lookup category_sub from classification_results.json
        category_sub = "기본"
        product_name = Path(image_path).name if image_path else f"item_{i}"

        if image_path in classification_lookup:
            item_info = classification_lookup[image_path]
            category_sub = item_info.get("detail_cat", "기본")
            # Use broad_category from classification if available
            if "broad_category" in item_info:
                category_main = item_info["broad_category"]

        products.append(ItemInfo(
            product_id=100 + i,
            product_name=product_name,
            category_main=category_main,
            category_sub=category_sub
        ))

    return OutfitSet(
        outfit_id=outfit_id,
        image_url=image_url,
        products=products
    )


def get_feedback_scopes(scope_input: str) -> List[FeedbackScope]:
    """
    Parse scope input string to FeedbackScope list.

    Input: "1" or "2 3" (space-separated numbers)
    1=FULL, 2=TOP, 3=BOTTOM, 4=OUTER
    """
    mapping = {
        "1": FeedbackScope.FULL,
        "2": FeedbackScope.TOP,
        "3": FeedbackScope.BOTTOM,
        "4": FeedbackScope.OUTER
    }

    if not scope_input.strip():
        return [FeedbackScope.FULL]

    scopes = []
    for idx in scope_input.split():
        if idx in mapping:
            scopes.append(mapping[idx])

    return scopes if scopes else [FeedbackScope.FULL]


def scope_to_db_category(scopes: List[FeedbackScope]) -> List[str]:
    """Convert FeedbackScope to DB category names for filtering"""
    mapping = {
        FeedbackScope.TOP: "tops",
        FeedbackScope.BOTTOM: "bottoms",
        FeedbackScope.OUTER: "outers"
    }
    return [mapping[s] for s in scopes if s in mapping]


# ============================================================
# Wardrobe Search
# ============================================================
def search_wardrobe(
    query: str,
    target_categories: List[str],
    wardrobe_checker: EmbeddingWardrobeChecker,
    user_id: str = "mj",
    target_detail_cats: List[str] = None
) -> List[str]:
    """
    Search wardrobe using FashionCLIP embeddings.

    Returns list of matching item IDs.
    """
    print(f"\n{'='*60}")
    print(f"[WARDROBE SEARCH] Query: {query}")
    print(f"[WARDROBE SEARCH] Categories: {target_categories or 'ALL'}")
    if target_detail_cats:
        print(f"[WARDROBE SEARCH] Detail Categories: {target_detail_cats}")
    print(f"{'='*60}")

    if not wardrobe_checker or not wardrobe_checker.wardrobe_collection:
        print("[ERROR] Wardrobe DB not loaded")
        return []

    result = wardrobe_checker.can_fulfill(
        requirements=[query],
        user_id=user_id,
        target_categories=target_categories if target_categories else None,
        target_detail_cats=target_detail_cats if target_detail_cats else None
    )

    print(f"[RESULT] Found {len(result.matching_items)} items (threshold: {wardrobe_checker.threshold})")

    if result.is_possible and result.matching_items:
        collection = wardrobe_checker.wardrobe_collection
        # Show top 3 results
        db_results = collection.get(
            ids=result.matching_items[:3],
            include=['metadatas']
        )

        print("\n  [TOP 3 MATCHES]")
        for i, (item_id, meta) in enumerate(zip(db_results['ids'], db_results['metadatas'])):
            broad = meta.get('broad_cat', meta.get('broad_category', 'unknown'))
            detail = meta.get('detail_cat', meta.get('category_sub', ''))
            path = meta.get('input', item_id)
            print(f"  [{i+1}] {broad} > {detail} ({path})")
    else:
        print(f"  [INFO] {result.reason}")

    return result.matching_items


# ============================================================
# Decision Display
# ============================================================
def print_decision(decision):
    """Print AI decision details"""
    print(f"\n{'='*60}")
    print(f"[AI DECISION] {decision.action.value}")
    print(f"{'='*60}")
    print(f"  Message: {decision.message}")
    print(f"  Reasoning: {decision.reasoning}")

    if decision.extracted_requirements:
        print(f"  Requirements: {decision.extracted_requirements}")

    if decision.target_categories:
        print(f"  Target Categories: {decision.target_categories}")

    if decision.buying_recommendations:
        rec = decision.buying_recommendations
        # Check if it's BuyingRecommendation object or dict
        if hasattr(rec, 'products'):
            products = rec.products
            reasoning = getattr(rec, 'reasoning', '')
        else:
            products = rec.get('products', []) if isinstance(rec, dict) else []
            reasoning = rec.get('reasoning', '') if isinstance(rec, dict) else ''

        print(f"\n  [PRODUCT RECOMMENDATIONS]")
        if reasoning:
            print(f"  Reasoning: {reasoning}")

        if not products:
            print("  (No products found)")
            print("  Possible causes:")
            print("    - products collection not in ChromaDB")
            print("    - Similarity below threshold (0.15)")
            print("    - Metadata cache key mismatch")
        else:
            for i, prod in enumerate(products[:3]):
                if hasattr(prod, 'product_name'):
                    # ProductRecommendation object
                    print(f"    ({i+1}) {prod.product_name} | {prod.brand} | {prod.price}won")
                    if prod.product_url:
                        print(f"        URL: {prod.product_url[:60]}...")
                else:
                    # Dict
                    print(f"    ({i+1}) {prod.get('product_name')} | {prod.get('brand')} | {prod.get('price')}won")
                    url = prod.get('product_url', '')
                    if url:
                        print(f"        URL: {url[:60]}...")


# ============================================================
# Main Demo Loop
# ============================================================
def run_demo():
    """Main demo function"""
    print("\n" + "="*60)
    print("  CloZ-AI Feedback Pipeline Demo")
    print("="*60)

    # 1. Load resources
    shared = get_shared_resources()
    if not shared:
        print("[ERROR] Failed to initialize. Exiting.")
        return

    classification_lookup = load_classification_data()

    # 2. Initialize ManagerAgent (max_regenerate_count=1)
    config = ManagerConfig(
        max_regenerate_count=1,
        enable_wardrobe_check=True,
        enable_buying_recommendation=True
    )
    manager = ManagerAgent(
        config=config,
        wardrobe_checker=shared['wardrobe_checker'],
        buying_trigger=shared['buying_trigger']
    )

    # 3. Get user ID
    user_id = input("\n[STEP 0] User ID (mj / yj): ").strip()
    if not user_id:
        user_id = "mj"
    print(f"  Using user_id: {user_id}")

    # 4. Get initial inputs
    print("\n[STEP 1] Original Query")
    original_prompt = input("  What style do you want? (default: casual date look): ").strip()
    if not original_prompt:
        original_prompt = "casual date look for spring"

    print("\n[STEP 2] Generated Outfit Info")
    image_dir = input("  Outfit image directory (default: output_images): ").strip()
    if not image_dir:
        image_dir = "output_images"

    print("  Enter products JSON (one line):")
    print('  Example: [{"category_main":"상의","image_path":"closet/tops/tops_1.jpeg"},{"category_main":"바지","image_path":"closet/bottoms/bottoms_1.jpeg"}]')
    products_json = input("  > ").strip()
    if not products_json:
        # Default example
        products_json = '[{"category_main":"상의","image_path":"closet/tops/tops_1.jpeg"},{"category_main":"바지","image_path":"closet/bottoms/bottoms_1.jpeg"}]'

    # 4. Parse outfit
    current_outfit = parse_outfit_input(image_dir, products_json, classification_lookup)
    print(f"\n  Parsed outfit: {[f'{p.category_main}:{p.category_sub}' for p in current_outfit.products]}")

    # 5. Start session
    session = manager.start_session(
        user_id=user_id,
        original_prompt=original_prompt,
        initial_outfit=current_outfit,
        context={"source": "demo"}
    )
    print(f"\n[SYSTEM] Session started: {session.session_id}")

    # 6. Feedback loop
    regenerate_used = False

    while True:
        print("\n" + "-"*60)
        print("[FEEDBACK] Current outfit:")
        for p in current_outfit.products:
            print(f"  - {p.category_main}: {p.category_sub} ({p.product_name})")

        # Yes/No input
        response = input("\nDo you like this outfit? (y/n/q to quit): ").lower().strip()

        if response == 'q':
            manager.end_session(session.session_id, SessionStatus.ABANDONED)
            print("\n[SYSTEM] Session abandoned.")
            break

        if response == 'y':
            # Positive feedback
            feedback = FeedbackInput(
                session_id=session.session_id,
                user_id=user_id,
                is_positive=True,
                current_outfit=current_outfit,
                feedback_text="Looks good!",
                feedback_scopes=[FeedbackScope.FULL]
            )
            decision = manager.process_feedback(feedback)
            print_decision(decision)
            manager.end_session(session.session_id, SessionStatus.COMPLETED)
            print("\n[SYSTEM] Session completed successfully!")
            break

        # Negative feedback
        print("\n[SCOPE] Which part don't you like?")
        print("  1=FULL  2=TOP  3=BOTTOM  4=OUTER (space-separated)")
        scope_input = input("  > ").strip()
        scopes = get_feedback_scopes(scope_input)

        feedback_text = input("\n[FEEDBACK] What's wrong with it?\n  > ").strip()
        if not feedback_text:
            feedback_text = "I don't like it"

        feedback = FeedbackInput(
            session_id=session.session_id,
            user_id=user_id,
            is_positive=False,
            current_outfit=current_outfit,
            feedback_text=feedback_text,
            feedback_scopes=scopes
        )

        print("\n[SYSTEM] Processing feedback...")
        decision = manager.process_feedback(feedback)
        print_decision(decision)

        # Handle decision
        if decision.action == ActionType.REGENERATE:
            if regenerate_used:
                # Should not happen due to max_regenerate_count=1, but just in case
                print("\n[SYSTEM] Regenerate already used. Redirecting to buying...")
                manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
                break

            regenerate_used = True
            print("\n[SYSTEM] Searching wardrobe for alternatives...")

            # Get search query and detail categories (prefer English for FashionCLIP)
            query = feedback_text
            target_detail_cats = None
            if decision.payload and 'regenerate_data' in decision.payload:
                structured = decision.payload['regenerate_data'].get('structured_query', {})
                if structured.get('requirements_en'):
                    query = " ".join(structured['requirements_en'])
                if structured.get('target_detail_cats'):
                    target_detail_cats = structured['target_detail_cats']

            target_cats = scope_to_db_category(scopes)
            matching_items = search_wardrobe(
                query,
                target_cats,
                shared['wardrobe_checker'],
                user_id=user_id,
                target_detail_cats=target_detail_cats
            )

            # Get new outfit input
            print("\n[STEP] Enter NEW outfit info based on search results:")
            new_products_json = input("  Products JSON > ").strip()
            if new_products_json:
                current_outfit = parse_outfit_input(image_dir, new_products_json, classification_lookup)
            else:
                print("  [INFO] Keeping same outfit for demo")
            continue

        elif decision.action == ActionType.ASK_MORE:
            print(f"\n[AI QUESTION] {decision.message}")
            answer = input("[YOUR ANSWER] > ").strip()

            if not answer:
                print("[INFO] No answer provided. Continuing...")
                continue

            # Combine original feedback with Q&A
            combined_text = f"{feedback_text} -> Q: '{decision.message}' -> A: '{answer}'"

            feedback = FeedbackInput(
                session_id=session.session_id,
                user_id=user_id,
                is_positive=False,
                current_outfit=current_outfit,
                feedback_text=combined_text,
                feedback_scopes=scopes
            )

            print("\n[SYSTEM] Re-processing with your answer...")
            decision = manager.process_feedback(feedback)
            print_decision(decision)

            # Handle post-ASK_MORE decision
            if decision.action == ActionType.REGENERATE:
                if regenerate_used:
                    print("\n[SYSTEM] Regenerate already used. Redirecting to buying...")
                    manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
                    break

                regenerate_used = True
                query = feedback_text
                target_detail_cats = None
                if decision.payload and 'regenerate_data' in decision.payload:
                    structured = decision.payload['regenerate_data'].get('structured_query', {})
                    if structured.get('requirements_en'):
                        query = " ".join(structured['requirements_en'])
                    if structured.get('target_detail_cats'):
                        target_detail_cats = structured['target_detail_cats']

                target_cats = scope_to_db_category(scopes)
                search_wardrobe(query, target_cats, shared['wardrobe_checker'], user_id=user_id, target_detail_cats=target_detail_cats)

                print("\n[STEP] Enter NEW outfit info:")
                new_products_json = input("  Products JSON > ").strip()
                if new_products_json:
                    current_outfit = parse_outfit_input(image_dir, new_products_json, classification_lookup)
                continue

            elif decision.action == ActionType.BUYING:
                print("\n[SYSTEM] No suitable items in wardrobe. Recommending products...")
                manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
                break

        elif decision.action == ActionType.BUYING:
            print("\n[SYSTEM] Redirecting to product recommendations...")
            manager.end_session(session.session_id, SessionStatus.BUYING_REDIRECT)
            break

        else:
            # APPROVED or unexpected
            manager.end_session(session.session_id, SessionStatus.COMPLETED)
            break

    print("\n" + "="*60)
    print("  Demo Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Demo interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
