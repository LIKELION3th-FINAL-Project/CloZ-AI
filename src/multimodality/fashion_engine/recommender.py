import sqlite3
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
import chromadb
from collections import defaultdict
from typing import Dict, List
from .config import FashionConfig
from .encoder import CLIPEncoder

class FashionRecommender:
    """사용자 쿼리와 스타일 기반 아이템 추천 엔진 클래스 (ChromaDB 캐싱 및 고도화 수식 반영)"""
    def __init__(self, config: FashionConfig, encoder: CLIPEncoder):
        self.config = config
        self.encoder = encoder
        self.item_db = {} # 기존 wardrobe 명칭을 item_db로 변경
        self.style_profiles = {}

    def load_user_wardrobe(
        self,
        collection_name="mycloset-embedding"
    ):
        """ChromaDB 캐시를 활용하여 옷장 아이템 로드 (없을 때만 인코딩)"""
        category_map = {"상의": "top", "하의": "bottom", "아우터": "outer"}
        cat_to_db = {"top": "shirt", "bottom": "pant", "outer": "outer"}

        # Chroma 로드
        client = chromadb.PersistentClient(path=self.config.chromadb_war_dir)
        col = client.get_or_create_collection(name=collection_name)

        loaded_from_chroma = 0
        computed_fallback = 0
        skipped = 0
        
        self.item_db = {}

        for kor_cat, eng_cat in category_map.items():
            path = os.path.join(self.config.base_dir, kor_cat)
            if not os.path.exists(path):
                continue

            for f in os.listdir(path):
                if not f.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
                    continue

                p = os.path.join(path, f)
                db_cat = cat_to_db.get(eng_cat, eng_cat)
                db_id = f"{db_cat}/{f}"
                chroma_id = f"item:{db_id}"

                emb_tensor = None

                # 1) Chroma에서 먼저 가져오기
                try:
                    res = col.get(ids=[chroma_id], include=["embeddings"])
                    if len(res.get("ids", [])) > 0:
                        emb = res["embeddings"][0]
                        emb_tensor = torch.tensor(emb, dtype=torch.float32)
                        emb_tensor = emb_tensor / (emb_tensor.norm() + 1e-8)
                        loaded_from_chroma += 1
                except Exception:
                    pass

                # 2) 없으면 새로 임베딩 후 저장
                if emb_tensor is None:
                    try:
                        emb_tensor = self.encoder.encode_image(p).to(torch.float32)
                        emb_tensor = emb_tensor / (emb_tensor.norm() + 1e-8)
                        computed_fallback += 1

                        metadata = {
                            "entity_type": "item",
                            "item_key": db_id,
                            "broad_cat": db_cat,
                            "source": "sqlite"
                        }
                        col.upsert(
                            ids=[chroma_id],
                            embeddings=[emb_tensor.cpu().tolist()],
                            metadatas=[metadata],
                        )
                    except Exception:
                        skipped += 1
                        continue

                # 메모리에 로드
                self.item_db[f] = {
                    "id": db_id,
                    "category": eng_cat,
                    "path": p,
                    "embedding": emb_tensor.cpu(),
                }

        print(f"[Wardrobe] total loaded: {len(self.item_db)}")
        print(f"  - from chroma: {loaded_from_chroma}")
        print(f"  - fallback computed: {computed_fallback}")
        print(f"  - skipped: {skipped}")
        return self.item_db

    def load_styles(self):
        """ChromaDB에서 스타일별 레퍼런스 임베딩 로드"""
        try:
            client = chromadb.PersistentClient(path=self.config.chromadb_ref_dir)
            collection = client.get_collection(name="reference_embeddings")
            data = collection.get(include=["embeddings", "metadatas"])
            
            style_dict = defaultdict(list)
            for emb, meta in zip(data['embeddings'], data['metadatas']):
                style_name = unicodedata.normalize("NFC", meta['style_cat'])
                style_dict[style_name].append(torch.tensor(emb))
            
            self.style_profiles = {k: torch.stack(v) for k, v in style_dict.items()}
            print(f"[ChromaDB] Total styles loaded: {len(self.style_profiles)}")
        except Exception as e:
            print(f"스타일 로드 실패: {e}")

    def recommend_from_agent(self, agent_json: Dict, top_k: int = 5) -> Dict[str, List[Dict]]:
        """분석된 인텐트를 기반으로 추천 (페널티 로직 포함)"""
        original_query = agent_json.get("original_query", "")
        intent = agent_json.get("analyzed_intent", {})
        target_style = unicodedata.normalize('NFC', intent.get("style", "캐주얼"))
        target_categories = intent.get("categories", ["top", "bottom", "outer"])
        expanded_keywords = agent_json.get("expanded_keywords", [original_query])

        # 쿼리 임베딩
        q_embs = [self.encoder.encode_text(kw) for kw in expanded_keywords]
        avg_q_emb = torch.stack(q_embs).mean(dim=0).to(torch.float32)
        avg_q_emb /= (avg_q_emb.norm() + 1e-8)

        all_results = {}
        for cat in target_categories:
            candidates = []
            for item_key, data in self.item_db.items():
                if data['category'] != cat: continue

                text_sim = float((avg_q_emb * data['embedding']).sum())

                # 스타일 점수 (고도화 버전)
                style_final = 0.5
                if target_style in self.style_profiles:
                    style_scores = []
                    item_emb_f32 = data['embedding'].to(torch.float32)
                    for s_name, s_embs in self.style_profiles.items():
                        sims = torch.matmul(s_embs.to(torch.float32), item_emb_f32)
                        style_scores.append((s_name, float(sims.max())))

                    style_scores.sort(key=lambda x: x[1], reverse=True)
                    top_3 = [s[0] for s in style_scores[:3]]
                    avg_all = np.mean([s[1] for s in style_scores])

                    # 페널티 로직
                    penalty = 1.0
                    if target_style not in top_3: penalty *= 0.1
                    if avg_all > 0.30: penalty *= 0.2

                    t_sim = next((s[1] for s in style_scores if s[0] == target_style), 0.0)
                    z = (t_sim - avg_all) / (np.std([s[1] for s in style_scores]) + 1e-8)
                    style_final = (1 / (1 + np.exp(-z))) * penalty

                combined = 0.3 * text_sim + 0.7 * style_final
                candidates.append({
                    **data, 
                    "score": combined, 
                    "text_sim": text_sim, 
                    "style_sim": style_final
                })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            all_results[cat] = candidates[:top_k]

        return all_results
