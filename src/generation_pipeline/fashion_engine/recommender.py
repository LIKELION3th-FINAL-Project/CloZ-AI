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
from ..utils.load import load_config
from loguru import logger
from pathlib import Path

class FashionRecommender:
    """고도화된 멀티 팩터 추천 엔진 (멀티 스타일, 색상, 계절, 무드 반영)"""
    def __init__(self, encoder: CLIPEncoder):
        self.config_path = Path(__file__).resolve().parents[3] / "configs" / "generation_model.yaml"
        self.config = load_config(self.config_path)
        self.folder_map = self.config["folder_map"]
        self.cat_to_db = self.config["cat_to_db"]
        self.encoder = encoder
        self.item_db = {} 
        self.style_profiles = {}

    def _resolve_user_clothes_dir(self) -> Optional[str]:
        configured = self.config.get("user_clothes_dir", "")
        candidates = []
        if configured:
            candidates.append(configured)

        project_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                str(project_root / "closet_mj"),
                str(project_root / "data" / "closet_mj"),
                str(Path.cwd() / "closet_mj"),
                str(Path.cwd() / "data" / "closet_mj"),
            ]
        )

        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                if candidate != configured:
                    logger.warning(
                        f"user_clothes_dir 경로를 자동 보정합니다: {configured} -> {candidate}"
                    )
                return candidate
        return None

    def load_user_wardrobe(self, collection_name = "wardrobe"):
        """ChromaDB 캐시를 활용하여 옷장 아이템 로드 (로컬 폴더 tops/bottoms 대응)"""
        try:
            client = chromadb.PersistentClient(path = self.config["chromadb_user_war_embedding_dir"])
            col = client.get_or_create_collection(name = collection_name)
        except Exception as e:
            logger.error(f"ChromaDB 로드 실패: {e}")
            return {}

        self.item_db = {}
        loaded_from = 0
        new_encoded = 0

        clothes_dir = self._resolve_user_clothes_dir()
        if not clothes_dir:
            logger.error(f"user_clothes_dir 경로가 존재하지 않습니다: {self.config['user_clothes_dir']}")
            return {}
        actual_dirs = os.listdir(clothes_dir)

        for folder_name in actual_dirs:
            norm_folder = folder_name.lower()
            if norm_folder not in self.folder_map: 
                continue

            kor_cat = self.folder_map[norm_folder]
            db_prefix = self.cat_to_db[kor_cat]
            path = os.path.join(clothes_dir, folder_name)
            if not os.path.isdir(path):
                continue

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
                except Exception as e:
                    logger.debug(f"Chroma 조회 실패({chroma_id}): {e}")

                if emb_tensor is None:
                    try:
                        emb_tensor = self.encoder.encode_image(full_path).to(torch.float32)
                        emb_tensor /= (emb_tensor.norm() + 1e-8)
                        new_encoded += 1
                        try:
                            col.upsert(
                                ids=[chroma_id],
                                embeddings=[emb_tensor.cpu().tolist()],
                                metadatas=[{"item_key": db_id, "broad_cat": db_prefix}],
                            )
                        except Exception as e:
                            logger.warning(f"Chroma upsert 실패({chroma_id}): {e}")
                    except Exception as e:
                        logger.warning(f"이미지 임베딩 실패({full_path}): {e}")
                        continue

                self.item_db[f] = {"id": db_id, "category": kor_cat, "path": full_path, "embedding": emb_tensor.cpu()}

        logger.info(f"Loaded {len(self.item_db)} items. (Chroma: {loaded_from}, New: {new_encoded})")
        if not self.item_db:
            logger.error(
                "옷장 아이템이 0개입니다. user_clothes_dir 내부에 tops/bottoms(또는 상의/하의) 폴더와 이미지 파일이 있는지 확인하세요."
            )
        return self.item_db

    def load_styles(self):
        """ChromaDB에서 스타일별 레퍼런스 임베딩 로드"""
        try:
            client = chromadb.PersistentClient(path = self.config["chromadb_ref_embedding_dir"])
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
            logger.info(f"[ChromaDB] Total styles loaded: {len(self.style_profiles)}")
        except Exception as e: logger.error(f"스타일 로드 실패: {e}")

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

    def _resolve_reference_styles(self, styles: List[str], top_n: int = 3) -> Dict[str, List[str]]:
        """요청 스타일별 실제 참고 ref 스타일 후보(top-n)를 반환"""
        if not styles or not self.style_profiles:
            return {}

        resolved = {}
        for style in styles[:3]:
            style_text = str(style or "").strip()
            if not style_text:
                continue
            norm_style = unicodedata.normalize("NFC", style_text)
            if norm_style in self.style_profiles:
                resolved[style_text] = [norm_style]
                continue

            query_emb = self.encoder.encode_text(style_text).to(torch.float32).cpu()
            query_emb /= (query_emb.norm() + 1e-8)

            scored = []
            for ref_name, ref_embs in self.style_profiles.items():
                ref_mean = ref_embs.to(torch.float32).mean(dim=0).cpu()
                ref_mean /= (ref_mean.norm() + 1e-8)
                sim = float((query_emb * ref_mean).sum())
                scored.append((ref_name, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            resolved[style_text] = [name for name, _ in scored[:top_n]]
        return resolved

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
        is_new = any(
            isinstance(agent_json.get(k), dict)
            for k in ["style", "color", "mood", "location", "season", "size_fit"]
        )

        if is_new:
            styles = agent_json.get("style", {}).get("value", ["캐주얼"])
            colors = agent_json.get("color", {}).get("value", [])
            target_cats = ["상의", "하의"]  # [원래 코드: 아우터 포함]
            # target_cats = ["상의", "하의", "아우터"]
            q_emb = self.encoder.encode_text(self._build_context_text(agent_json)).to(torch.float32).cpu()
            conf_weight = self._calculate_confidence_weight(agent_json)
        else:
            intent = agent_json.get("analyzed_intent", {})
            styles = [intent.get("style", "캐주얼")]
            target_cats = intent.get("categories", ["상의", "하의"])  # [원래 코드: 아우터 포함]
            # target_cats = intent.get("categories", ["상의", "하의", "아우터"])
            q_embs = [self.encoder.encode_text(kw) for kw in agent_json.get("expanded_keywords", [agent_json.get("original_query", "")])]
            q_emb = torch.stack(q_embs).mean(dim=0).to(torch.float32).cpu()
            conf_weight = 1.0

        q_emb /= (q_emb.norm() + 1e-8)
        all_results = {}
        ref_map = self._resolve_reference_styles(styles, top_n=3)
        logger.info(
            f"[RECOMMEND][ref_db] style_query={styles}, "
            f"target_categories={target_cats}, ref_candidates={ref_map if ref_map else 'none'}, top_k={top_k}"
        )

        for cat in target_cats:
            candidates = []
            for f_name, data in self.item_db.items():
                if data['category'] != cat: 
                    continue

                item_emb = data['embedding'].to(torch.float32).cpu()
                text_sim = float((q_emb * item_emb).sum())
                style_sim = self._calculate_multi_style_similarity(item_emb, styles)
                
                # 새로운 팩터들 (계절, 무드 등)
                color_sim, season_sim, mood_sim = 0.5, 0.5, 0.5
                if is_new:
                    color_sim = self._calculate_color_similarity(data['path'], colors)
                    for field in ["season", "mood"]:
                        vals = agent_json.get(field, {}).get("value")
                        if vals:
                            field_emb = self.encoder.encode_text(", ".join(vals)).to(torch.float32).cpu()
                            sim_value = float((item_emb * field_emb).sum())
                            if field == "season":
                                season_sim = sim_value
                            elif field == "mood":
                                mood_sim = sim_value

                    final_score = (0.2 * text_sim + 0.5 * style_sim + 0.15 * color_sim + 0.1 * season_sim + 0.05 * mood_sim) * conf_weight
                else:
                    final_score = 0.3 * text_sim + 0.7 * style_sim

                candidates.append({**data, "score": final_score, "text_sim": text_sim, "style_sim": style_sim, "color_sim": color_sim})

            candidates.sort(key=lambda x: x['score'], reverse=True)
            all_results[cat] = candidates[:top_k]

        return all_results
