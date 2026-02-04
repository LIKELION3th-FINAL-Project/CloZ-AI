import sqlite3
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List
from .config import FashionConfig
from .encoder import CLIPEncoder
from .db_manager import FashionDBManager

class FashionRecommender:
    """사용자 쿼리와 스타일 기반 아이템 추천 엔진 클래스"""
    def __init__(self, config: FashionConfig, encoder: CLIPEncoder, db: FashionDBManager):
        self.config = config
        self.encoder = encoder
        self.db = db
        self.wardrobe = [] # 로드된 아이템 리스트
        self.style_profiles = {} # 스타일별 레퍼런스 임베딩

    def load_user_items(self):
        """DB에 등록된 모든 아이템의 이미지를 인코딩하여 메모리에 로드"""
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        rows = cur.execute("SELECT id, broad_cat FROM items").fetchall()
        conn.close()
        
        self.wardrobe = []
        for db_id, cat in rows:
            path = self.db.get_path_from_id(db_id)
            if os.path.exists(path):
                self.wardrobe.append({
                    "id": db_id, "category": cat, "path": path, 
                    "embedding": self.encoder.encode_image(path)
                })
        print(f"[INFO] 로드된 아이템 수: {len(self.wardrobe)}")

    def load_styles(self):
        """ChromaDB에서 스타일별 대표 임베딩 로드"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.config.chromadb_dir)
            collection = client.get_collection(name="reference_embeddings")
            data = collection.get(include=["embeddings", "metadatas"])
            
            style_dict = defaultdict(list)
            for emb, meta in zip(data['embeddings'], data['metadatas']):
                style_name = unicodedata.normalize("NFC", meta['style_cat'])
                style_dict[style_name].append(torch.tensor(emb))
            
            self.style_profiles = {k: torch.stack(v) for k, v in style_dict.items()}
            
            for style_name, embeddings in self.style_profiles.items():
                print(f"[ChromaDB] Loaded {style_name}: {embeddings.shape[0]} embeddings")
            print(f"\n[ChromaDB] Total styles loaded: {len(self.style_profiles)}")
            
        except Exception as e:
            print(f"스타일 로드 실패: {e}")

    def recommend(self, query_data: Dict, top_k: int = 5) -> Dict[str, List[Dict]]:
        """텍스트 쿼리와 타겟 스타일에 가장 적합한 아이템 검색 (원본 가중치 및 페널티 로직 복구)"""
        intent = query_data.get("analyzed_intent", {})
        target_style = unicodedata.normalize('NFC', intent.get("style", "캐주얼"))
        cat_map = {"top": "shirt", "bottom": "pant", "outer": "outer"}
        target_cats = [cat_map.get(c, c) for c in intent.get("categories", [])] or ["shirt", "pant", "outer"]

        # 쿼리 임베딩 생성
        keywords = query_data.get("expanded_keywords", [query_data.get("original_query", "")])
        q_embs = [self.encoder.encode_text(kw) for kw in keywords]
        avg_q_emb = F.normalize(torch.stack(q_embs).mean(dim=0), dim=-1)

        results = defaultdict(list)
        for item in self.wardrobe:
            if item['category'] not in target_cats: continue
            
            # 1. 텍스트 유사도
            text_sim = float((avg_q_emb * item['embedding']).sum())
            
            # 2. 정교한 스타일 점수 계산 (원본 그대로 복구)
            style_final = 0.5
            if target_style in self.style_profiles:
                style_scores = []
                item_emb_f32 = item['embedding'].to(torch.float32)
                
                # 모든 스타일과의 유사도 비교
                for s_name, s_embs in self.style_profiles.items():
                    # 스타일 임베딩 정규화 후 최대 유사도 추출
                    s_embs_norm = F.normalize(s_embs.to(torch.float32), dim=-1)
                    sims = torch.matmul(s_embs_norm, item_emb_f32)
                    style_scores.append((s_name, float(sims.max())))

                style_scores.sort(key=lambda x: x[1], reverse=True)
                top_3 = [s[0] for s in style_scores[:3]]
                avg_all = np.mean([s[1] for s in style_scores])

                # 페널티 로직
                penalty = 1.0
                if target_style not in top_3: penalty *= 0.1
                if avg_all > 0.30: penalty *= 0.2

                # Z-Score 및 Sigmoid 적용
                t_sim = next((s[1] for s in style_scores if s[0] == target_style), 0.0)
                z = (t_sim - avg_all) / (np.std([s[1] for s in style_scores]) + 1e-8)
                style_final = (1 / (1 + np.exp(-z))) * penalty

            # 최종 결합 점수: 텍스트 30% + 스타일 70%
            combined = 0.3 * text_sim + 0.7 * style_final
            results[item['category']].append({
                **item, 
                "score": combined, 
                "text_sim": text_sim, 
                "style_sim": style_final
            })

        for cat in results:
            results[cat].sort(key=lambda x: x['score'], reverse=True)
            results[cat] = results[cat][:top_k]
        
        return dict(results)
