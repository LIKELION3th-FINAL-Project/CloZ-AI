import sqlite3
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
import chromadb
from collections import defaultdict
from typing import Dict, List, Optional
from .config import FashionConfig
from .encoder import CLIPEncoder

class FashionRecommender:
    """사용자 쿼리와 스타일 기반 아이템 추천 엔진 클래스 (실제 폴더명 tops/bottoms/outers 대응)"""
    def __init__(self, config: FashionConfig, encoder: CLIPEncoder):
        self.config = config
        self.encoder = encoder
        self.item_db = {} 
        self.style_profiles = {}

    def load_user_wardrobe(self, collection_name="mycloset-embedding"):
        """ChromaDB 캐시를 활용하여 옷장 아이템 로드"""
        # [수정] 실제 폴더명(스크린샷 기준) -> 내부 카테고리 한글 명칭
        folder_map = {
            "tops": "상의",
            "bottoms": "하의",
            "outers": "아우터",
            "상의": "상의",
            "하의": "하의",
            "아우터": "아우터"
        }
        # 한글 명칭 -> DB 저장용 영문 ID 접두사
        cat_to_db = {"상의": "shirt", "하의": "pant", "아우터": "outer"}

        try:
            client = chromadb.PersistentClient(path=self.config.chromadb_war_dir)
            col = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"[Error] ChromaDB 로드 실패: {e}")
            return {}

        self.item_db = {}
        loaded_from_chroma = 0
        computed_fallback = 0

        # base_dir 안의 실제 폴더 검색
        if not os.path.exists(self.config.base_dir):
            print(f"[Error] 경로를 찾을 수 없습니다: {self.config.base_dir}")
            return {}

        actual_dirs = os.listdir(self.config.base_dir)
        print(f"[Debug] 스캔한 폴더 목록: {actual_dirs}")

        for folder_name in actual_dirs:
            # 매핑 테이블에 존재하는 폴더만 처리
            norm_folder = folder_name.lower()
            if norm_folder not in folder_map:
                continue

            kor_cat = folder_map[norm_folder]
            db_prefix = cat_to_db[kor_cat]
            path = os.path.join(self.config.base_dir, folder_name)

            for f in os.listdir(path):
                if not f.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
                    continue

                full_path = os.path.join(path, f)
                db_id = f"{db_prefix}/{f}"
                chroma_id = f"item:{db_id}"

                emb_tensor = None

                # 1) ChromaDB 캐시 확인
                try:
                    res = col.get(ids=[chroma_id], include=["embeddings"])
                    if len(res.get("ids", [])) > 0:
                        emb = res["embeddings"][0]
                        emb_tensor = torch.tensor(emb, dtype=torch.float32)
                        emb_tensor = emb_tensor / (emb_tensor.norm() + 1e-8)
                        loaded_from_chroma += 1
                except:
                    pass

                # 2) 캐시 없으면 실시간 인코딩
                if emb_tensor is None:
                    try:
                        emb_tensor = self.encoder.encode_image(full_path).to(torch.float32)
                        emb_tensor = emb_tensor / (emb_tensor.norm() + 1e-8)
                        computed_fallback += 1

                        col.upsert(
                            ids=[chroma_id],
                            embeddings=[emb_tensor.cpu().tolist()],
                            metadatas={"item_key": db_id, "broad_cat": db_prefix, "source": "local"}
                        )
                    except Exception as e:
                        print(f"[Error] {f} 처리 중 오류: {e}")
                        continue

                # 3) 메모리 저장 (Simulation 코드와 호환을 위해 한글 카테고리 유지)
                self.item_db[f] = {
                    "id": db_id,
                    "category": kor_cat,
                    "path": full_path,
                    "embedding": emb_tensor.cpu(),
                }

        print(f"[Wardrobe] Loaded {len(self.item_db)} items. (Chroma: {loaded_from_chroma}, New: {computed_fallback})")
        return self.item_db

    def load_styles(self):
        """ChromaDB에서 스타일별 레퍼런스 임베딩 로드 (첫 번째 컬렉션 자동 선택)"""
        try:
            client = chromadb.PersistentClient(path=self.config.chromadb_ref_dir)
            colls = client.list_collections()
            if not colls:
                print("[WARN] 스타일 컬렉션이 없습니다.")
                return

            # 이름에 'reference'가 들어간 컬렉션 우선 선택, 없으면 첫 번째 선택
            target_coll = next((c for c in colls if "reference" in c.name), colls[0])
            print(f"[Info] 스타일 컬렉션 '{target_coll.name}' 로드 중...")
            
            collection = client.get_collection(name=target_coll.name)
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
        """원본 수식 기반 추천"""
        intent = agent_json.get("analyzed_intent", {})
        target_style = unicodedata.normalize('NFC', intent.get("style", "캐주얼"))
        target_categories = intent.get("categories", ["상의", "하의", "아우터"])
        
        q_embs = [self.encoder.encode_text(kw) for kw in agent_json.get("expanded_keywords", [agent_json.get("original_query", "")])]
        avg_q_emb = torch.stack(q_embs).mean(dim=0).to(torch.float32)
        avg_q_emb /= (avg_q_emb.norm() + 1e-8)

        all_results = {}
        for cat in target_categories:
            candidates = []
            for f_name, data in self.item_db.items():
                if data['category'] != cat: continue

                text_sim = float((avg_q_emb * data['embedding']).sum())
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

                    penalty = 1.0
                    if target_style not in top_3: penalty *= 0.1
                    if avg_all > 0.30: penalty *= 0.2

                    t_sim = next((s[1] for s in style_scores if s[0] == target_style), 0.0)
                    z = (t_sim - avg_all) / (np.std([s[1] for s in style_scores]) + 1e-8)
                    style_final = (1 / (1 + np.exp(-z))) * penalty

                combined = 0.3 * text_sim + 0.7 * style_final
                candidates.append({**data, "score": combined, "text_sim": text_sim, "style_sim": style_final})

            candidates.sort(key=lambda x: x['score'], reverse=True)
            all_results[cat] = candidates[:top_k]
        return all_results
