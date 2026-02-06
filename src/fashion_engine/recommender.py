import sqlite3
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
import chromadb
from collections import defaultdict
from typing import Dict, List, Optional
from .encoder import CLIPEncoder
from utils.load import load_config
from loguru import logger
from pathlib import Path

class FashionRecommender:
    """고도화된 멀티 팩터 추천 엔진 (멀티 스타일, 색상, 계절, 무드 반영)"""
    def __init__(self, encoder: CLIPEncoder):
        self.config_path = Path(__file__).resolve().parents[1] / "configs" / "generation_model.yaml"
        self.config = load_config(self.config_path)
        self.folder_map = self.config["folder_map"]
        self.cat_to_db = self.config["cat_to_db"]
        self.encoder = encoder
        self.item_db = {} 
        self.style_profiles = {}

    def load_user_wardrobe(self, collection_name="mycloset-embedding"):
        """ChromaDB 캐시를 활용하여 옷장 아이템 로드 (로컬 폴더 tops/bottoms/outers 대응)"""
        try:
            client = chromadb.PersistentClient(path=self.config.chromadb_war_dir)
            col = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.error(f"ChromaDB 로드 실패: {e}")
            return {}

        self.item_db = {}
        loaded_from = 0
        new_encoded = 0

        if not os.path.exists(self.config.base_dir): 
            return {}
        actual_dirs = os.listdir(self.config.base_dir)

        for folder_name in actual_dirs:
            norm_folder = folder_name.lower()
            if norm_folder not in self.folder_map: 
                continue

            kor_cat = self.folder_map[norm_folder]
            db_prefix = self.cat_to_db[kor_cat]
            path = os.path.join(self.config.base_dir, folder_name)

            for f in os.listdir(path):
                if not f.lower().endswith((".jpg", ".png", ".jpeg", ".webp")): 
                    continue
                full_path = os.path.join(path, f)
                db_id = f"{db_prefix}/{f}"
                chroma_id = f"item:{db_id}"

                emb_tensor = None
                try:
                    res = col.get(ids=[chroma_id], include=["embeddings"])
                    if len(res.get("ids", [])) > 0:
                        emb = res["embeddings"][0]
                        emb_tensor = torch.tensor(emb, dtype=torch.float32)
                        emb_tensor /= (emb_tensor.norm() + 1e-8)
                        loaded_from += 1
                except: pass

                if emb_tensor is None:
                    try:
                        emb_tensor = self.encoder.encode_image(full_path).to(torch.float32)
                        emb_tensor /= (emb_tensor.norm() + 1e-8)
                        new_encoded += 1
                        col.upsert(ids=[chroma_id], embeddings=[emb_tensor.cpu().tolist()], metadatas={"item_key": db_id, "broad_cat": db_prefix})
                    except: continue

                self.item_db[f] = {"id": db_id, "category": kor_cat, "path": full_path, "embedding": emb_tensor.cpu()}

        print(f"[Wardrobe] Loaded {len(self.item_db)} items. (Chroma: {loaded_from}, New: {new_encoded})")
        return self.item_db

    def load_styles(self):
        """ChromaDB에서 스타일별 레퍼런스 임베딩 로드"""
        try:
            client = chromadb.PersistentClient(path=self.config.chromadb_ref_dir)
            colls = client.list_collections()
            if not colls: return
            target_coll = next((c for c in colls if "reference" in c.name), colls[0])
            collection = client.get_collection(name=target_coll.name)
            data = collection.get(include=["embeddings", "metadatas"])
            
            style_dict = defaultdict(list)
            for emb, meta in zip(data['embeddings'], data['metadatas']):
                style_name = unicodedata.normalize("NFC", meta['style_cat'])
                style_dict[style_name].append(torch.tensor(emb))
            self.style_profiles = {k: torch.stack(v) for k, v in style_dict.items()}
            print(f"[ChromaDB] Total styles loaded: {len(self.style_profiles)}")
        except Exception as e: print(f"스타일 로드 실패: {e}")

    # ========== 신규 헬퍼 함수들 (고도화 로직) ==========
    def _build_context_text(self, agent_json: Dict) -> str:
        """JSON 데이터에서 문맥 텍스트 생성"""
        parts = []
        for field in ["color", "size_fit", "mood", "location", "season"]:
            val = agent_json.get(field, {}).get("value")
            if val: parts.append(f"{field}: {', '.join(val)}")
        return " | ".join(parts) if parts else "casual fashion"

    def _calculate_multi_style_similarity(self, item_emb: torch.Tensor, styles: List[str]) -> float:
        """여러 스타일에 대한 가중 유사도 계산"""
        if not styles or not self.style_profiles: return 0.5
        sims, weights = [], [0.5, 0.3, 0.2]
        item_emb_f32 = item_emb.to(torch.float32)

        for idx, target_style in enumerate(styles[:3]):
            norm_style = unicodedata.normalize('NFC', target_style)
            if norm_style not in self.style_profiles: continue
            
            all_scores = []
            for s_name, s_embs in self.style_profiles.items():
                s_max = float(torch.matmul(s_embs.to(torch.float32), item_emb_f32).max())
                all_scores.append((s_name, s_max))
            
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top_3, avg_all = [s[0] for s in all_scores[:3]], np.mean([s[1] for s in all_scores])
            penalty = 1.0
            if norm_style not in top_3: penalty *= 0.1
            if avg_all > 0.30: penalty *= 0.2

            t_sim = next((s[1] for s in all_scores if s[0] == norm_style), 0.0)
            z = (t_sim - avg_all) / (np.std([s[1] for s in all_scores]) + 1e-8)
            style_final = (1 / (1 + np.exp(-z))) * penalty
            sims.append(style_final * (weights[idx] if idx < len(weights) else 0.1))

        return sum(sims) / (sum(weights[:len(sims)]) or 1.0)

    def _calculate_color_similarity(self, item_path: str, colors: List[str]) -> float:
        """색상 키워드와 이미지 유사도 (텍스트 기반)"""
        if not colors: return 0.5
        color_text = ", ".join(colors) + " clothing"
        try:
            item_emb = self.encoder.encode_image(item_path).to(torch.float32)
            c_emb = self.encoder.encode_text(color_text).to(torch.float32)
            return float((item_emb * c_emb).sum())
        except: return 0.5

    def _calculate_confidence_weight(self, agent_json: Dict) -> float:
        """분석 결과의 신뢰도를 점수에 반영"""
        confs = [
                    agent_json[f]["confidence"] for f in ["style", "color", "mood", "location", "season", "size_fit"] 
                    if f in agent_json and "confidence" in agent_json[f]
                    ]
        return 0.5 + (np.mean(confs) * 0.5) if confs else 0.7

    # ========== 메인 추천 함수 (고도화 버전) ==========

    def recommend_from_agent(self, agent_json: Dict, top_k: int = 5) -> Dict[str, List[Dict]]:
        """멀티 팩터 기반 추천 (신규/기존 형식 모두 지원)"""
        is_new = "style" in agent_json and isinstance(agent_json["style"], dict)

        if is_new:
            styles = agent_json.get("style", {}).get("value", ["캐주얼"])
            colors = agent_json.get("color", {}).get("value", [])
            target_cats = ["상의", "하의", "아우터"]
            q_emb = self.encoder.encode_text(self._build_context_text(agent_json)).to(torch.float32)
            conf_weight = self._calculate_confidence_weight(agent_json)
        else:
            intent = agent_json.get("analyzed_intent", {})
            styles = [intent.get("style", "캐주얼")]
            target_cats = intent.get("categories", ["상의", "하의", "아우터"])
            q_embs = [self.encoder.encode_text(kw) for kw in agent_json.get("expanded_keywords", [agent_json.get("original_query", "")])]
            q_emb = torch.stack(q_embs).mean(dim=0).to(torch.float32)
            conf_weight = 1.0

        q_emb /= (q_emb.norm() + 1e-8)
        all_results = {}

        for cat in target_cats:
            candidates = []
            for f_name, data in self.item_db.items():
                if data['category'] != cat: continue

                text_sim = float((q_emb * data['embedding']).sum())
                style_sim = self._calculate_multi_style_similarity(data['embedding'], styles)
                
                # 새로운 팩터들 (계절, 무드 등)
                color_sim, season_sim, mood_sim = 0.5, 0.5, 0.5
                if is_new:
                    color_sim = self._calculate_color_similarity(data['path'], colors)
                    for field, sim_attr in [("season", "season_sim"), ("mood", "mood_sim")]:
                        vals = agent_json.get(field, {}).get("value")
                        if vals:
                            field_emb = self.encoder.encode_text(", ".join(vals)).to(torch.float32)
                            globals()[sim_attr] = float((data['embedding'].to(torch.float32) * field_emb).sum())

                    final_score = (0.2 * text_sim + 0.5 * style_sim + 0.15 * color_sim + 0.1 * season_sim + 0.05 * mood_sim) * conf_weight
                else:
                    final_score = 0.3 * text_sim + 0.7 * style_sim

                candidates.append({**data, "score": final_score, "text_sim": text_sim, "style_sim": style_sim, "color_sim": color_sim})

            candidates.sort(key=lambda x: x['score'], reverse=True)
            all_results[cat] = candidates[:top_k]

        return all_results
