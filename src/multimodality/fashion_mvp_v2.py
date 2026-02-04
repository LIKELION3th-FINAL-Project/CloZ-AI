# -*- coding: utf-8 -*-
import os
import sqlite3
import itertools
import unicodedata
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# --- Configuration ---

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 경로 관리 클래스"""
    base_dir: str = "/content/drive/MyDrive/LikeLion/최종 프로젝트 개인 임시/테스트용"
    db_path: str = "/content/fashion_items.db"
    chromadb_dir: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb"
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True

# --- Core Modules ---

class CLIPEncoder:
    """CLIP 모델을 이용한 이미지 및 텍스트 임베딩 추출 클래스"""
    def __init__(self, config: FashionConfig):
        self.config = config
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model = CLIPModel.from_pretrained(config.model_name).to(config.device).eval()
        if config.use_fp16 and config.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def _extract_features(self, outputs: Any) -> torch.Tensor:
        """모델 결과에서 공통적으로 특성 텐서를 추출하고 정규화함"""
        if hasattr(outputs, "image_embeds"):
            features = outputs.image_embeds
        elif hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        if features.dim() == 3: features = features.mean(dim=1)
        if features.dim() == 2 and features.size(0) == 1: features = features[0]
        
        return F.normalize(features.float(), dim=-1).cpu()

    def encode_image(self, path: str) -> torch.Tensor:
        """이미지 파일을 읽어 임베딩 벡터 생성"""
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.config.device)
        if self.config.use_fp16 and self.config.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        outputs = self.model.get_image_features(**inputs)
        return self._extract_features(outputs)

    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트 쿼리를 임베딩 벡터 생성"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)
        outputs = self.model.get_text_features(**inputs)
        return self._extract_features(outputs)


class FashionDBManager:
    """SQLite 데이터베이스 관리 및 아이템 정보 조회 클래스"""
    def __init__(self, config: FashionConfig):
        self.config = config
        self.categories_map = {"하의": "pant", "아우터": "outer", "상의": "shirt"}

    def initialize_db(self):
        """이미지 폴더를 스캔하여 DB 초기화 및 데이터 삽입"""
        if os.path.exists(self.config.db_path):
            os.remove(self.config.db_path)
            print(f"[DB] Removed existing database: {self.config.db_path}")
            
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id TEXT PRIMARY KEY, broad_cat TEXT NOT NULL, detail_cat TEXT)")
        cur.execute("CREATE INDEX idx_items_broad_cat ON items(broad_cat)")
        
        items = []
        for kor_name, eng_name in self.categories_map.items():
            folder = os.path.join(self.config.base_dir, kor_name)
            if not os.path.isdir(folder): 
                print(f"[WARN] category dir not found: {folder}")
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff')):
                    db_id = f"{eng_name}/{f}"
                    items.append((db_id, eng_name, None))
        
        cur.executemany("INSERT INTO items VALUES (?, ?, ?)", items)
        conn.commit()
        conn.close()
        print(f"[DB] Created new table with {len(items)} items")

    def get_path_from_id(self, db_id: str) -> str:
        """DB ID(예: pant/1.jpg)를 실제 파일 경로로 변환"""
        eng_cat, filename = db_id.split('/', 1)
        inv_map = {v: k for k, v in self.categories_map.items()}
        return os.path.join(self.config.base_dir, inv_map[eng_cat], filename)


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


class OutfitPlanner:
    """아이템 조합 생성 및 세트 코디네이션 평가 클래스"""
    def __init__(self, encoder: CLIPEncoder):
        self.encoder = encoder

    def generate_combinations(self, recommendations: Dict[str, List[Dict]], top_n: int = 3) -> List[Tuple]:
        """카테고리별 상위 아이템들로 가능한 모든 조합 생성"""
        pants = recommendations.get("pant", [])[:top_n]
        outers = recommendations.get("outer", [])[:top_n]
        shirts = recommendations.get("shirt", [])[:top_n]# -*- coding: utf-8 -*-
import os
import sqlite3
import itertools
import unicodedata
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# --- Configuration ---

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 경로 관리 클래스"""
    base_dir: str = "/content/drive/MyDrive/LikeLion/최종 프로젝트 개인 임시/테스트용"
    db_path: str = "/content/fashion_items.db"
    chromadb_dir: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb"
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True

# --- Core Modules ---

class CLIPEncoder:
    """CLIP 모델을 이용한 이미지 및 텍스트 임베딩 추출 클래스"""
    def __init__(self, config: FashionConfig):
        self.config = config
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model = CLIPModel.from_pretrained(config.model_name).to(config.device).eval()
        if config.use_fp16 and config.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def _extract_features(self, outputs: Any) -> torch.Tensor:
        """모델 결과에서 공통적으로 특성 텐서를 추출하고 정규화함"""
        if hasattr(outputs, "image_embeds"):
            features = outputs.image_embeds
        elif hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        if features.dim() == 3: features = features.mean(dim=1)
        if features.dim() == 2 and features.size(0) == 1: features = features[0]
        
        return F.normalize(features.float(), dim=-1).cpu()

    def encode_image(self, path: str) -> torch.Tensor:
        """이미지 파일을 읽어 임베딩 벡터 생성"""
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.config.device)
        if self.config.use_fp16 and self.config.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        outputs = self.model.get_image_features(**inputs)
        return self._extract_features(outputs)

    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트 쿼리를 임베딩 벡터 생성"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)
        outputs = self.model.get_text_features(**inputs)
        return self._extract_features(outputs)


class FashionDBManager:
    """SQLite 데이터베이스 관리 및 아이템 정보 조회 클래스"""
    def __init__(self, config: FashionConfig):
        self.config = config
        self.categories_map = {"하의": "pant", "아우터": "outer", "상의": "shirt"}

    def initialize_db(self):
        """이미지 폴더를 스캔하여 DB 초기화 및 데이터 삽입"""
        if os.path.exists(self.config.db_path):
            os.remove(self.config.db_path)
            print(f"[DB] Removed existing database: {self.config.db_path}")
            
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id TEXT PRIMARY KEY, broad_cat TEXT NOT NULL, detail_cat TEXT)")
        cur.execute("CREATE INDEX idx_items_broad_cat ON items(broad_cat)")
        
        items = []
        for kor_name, eng_name in self.categories_map.items():
            folder = os.path.join(self.config.base_dir, kor_name)
            if not os.path.isdir(folder): 
                print(f"[WARN] category dir not found: {folder}")
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff')):
                    db_id = f"{eng_name}/{f}"
                    items.append((db_id, eng_name, None))
        
        cur.executemany("INSERT INTO items VALUES (?, ?, ?)", items)
        conn.commit()
        conn.close()
        print(f"[DB] Created new table with {len(items)} items")

    def get_path_from_id(self, db_id: str) -> str:
        """DB ID(예: pant/1.jpg)를 실제 파일 경로로 변환"""
        eng_cat, filename = db_id.split('/', 1)
        inv_map = {v: k for k, v in self.categories_map.items()}
        return os.path.join(self.config.base_dir, inv_map[eng_cat], filename)


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


class OutfitPlanner:
    """아이템 조합 생성 및 세트 코디네이션 평가 클래스"""
    def __init__(self, encoder: CLIPEncoder):
        self.encoder = encoder

    def generate_combinations(self, recommendations: Dict[str, List[Dict]], top_n: int = 3) -> List[Tuple]:
        """카테고리별 상위 아이템들로 가능한 모든 조합 생성"""
        pants = recommendations.get("pant", [])[:top_n]
        outers = recommendations.get("outer", [])[:top_n]
        shirts = recommendations.get("shirt", [])[:top_n]
        
        if not (pants and outers and shirts): 
            print("[WARN] 조합 생성에 필요한 카테고리가 부족합니다.")
            return []
        
        combinations = list(itertools.product(pants, outers, shirts))
        print(f"총 {len(combinations)}개의 조합이 생성되었습니다.")
        print(f"- pant: {len(pants)}개")
        print(f"- outer: {len(outers)}개")
        print(f"- shirt: {len(shirts)}개")
        
        return combinations

    def evaluate_outfits(self, combinations: List[Tuple], style_profiles: Dict[str, torch.Tensor], 
                        target_style: str, overall_weight: float = 0.7) -> List[Dict]:
        """조합된 세트의 전체 조화도 및 스타일 부합도 평가 (가중치 파라미터 추가)"""
        target_style = unicodedata.normalize("NFC", target_style)
        if target_style not in style_profiles: 
            print(f"[WARN] 타겟 스타일 '{target_style}'이 프로필에 없습니다.")
            return []
        
        # 스타일 레퍼런스 정규화
        style_embs = F.normalize(style_profiles[target_style].to(torch.float32), dim=-1)
        results = []
        
        print(f"총 {len(combinations)}개 조합 평가 중...")
        
        for i, combo in enumerate(combinations):
            if (i + 1) % 5 == 0:
                print(f"  진행: {i + 1}/{len(combinations)}")
            
            # 3장 평균 임베딩 (전체 조화)
            embs = torch.stack([item['embedding'] for item in combo]).to(torch.float32)
            combo_emb = F.normalize(embs.mean(dim=0), dim=-1)
            harmony_sim = torch.max(combo_emb @ style_embs.T).item()
            
            # 개별 아이템 vs 스타일 평균 (개별 품질)
            indiv_sims = []
            for item in combo:
                item_emb = F.normalize(item['embedding'].to(torch.float32), dim=-1)
                sim = torch.max(item_emb @ style_embs.T).item()
                indiv_sims.append(sim)
            indiv_avg = np.mean(indiv_sims)
            
            # 최종 점수 (overall_weight 파라미터 적용)
            final_score = overall_weight * harmony_sim + (1 - overall_weight) * indiv_avg
            
            results.append({
                "combination": combo,
                "combination_idx": i,
                "final_score": final_score,
                "harmony_score": harmony_sim,
                "individual_avg": indiv_avg,
                "pant": combo[0]['path'].split('/')[-1],
                "outer": combo[1]['path'].split('/')[-1],
                "shirt": combo[2]['path'].split('/')[-1]
            })
            
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 상위 결과 출력
        print(f"\n평가 완료! 상위 3개 조합:")
        for rank, result in enumerate(results[:3], 1):
            print(f"\n{rank}위 (조합 #{result['combination_idx'] + 1}):")
            print(f"  최종 점수: {result['final_score']:.4f}")
            print(f"  전체 조화 유사도: {result['harmony_score']:.4f}")
            print(f"  개별 평균 유사도: {result['individual_avg']:.4f}")
            print(f"  - pant: {result['pant']}")
            print(f"  - outer: {result['outer']}")
            print(f"  - shirt: {result['shirt']}")
        
        return results


class VTONManager:
    """가상 피팅(Virtual Try-On) 실행 및 이미지 생성 클래스"""
    def __init__(self, weights_dir: str = "./weights"):
        self.pipeline = None
        try:
            from fashn_vton import TryOnPipeline
            self.pipeline = TryOnPipeline(weights_dir=weights_dir)
            print("✅ Fashion-VTON 파이프라인 로드 완료! (가중치는 재사용됩니다)")
        except ImportError:
            print("[WARN] fashn_vton 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print(f"[WARN] Fashion-VTON 로드 실패: {e}")

    def try_on(self, person_img_path: str, outfit: Tuple, output_prefix: str, idx: int = 0) -> Optional[Dict]:
        """상의, 하위 순차적으로 가상 피팅 적용"""
        if not self.pipeline: 
            print("[WARN] VTON 파이프라인이 로드되지 않았습니다.")
            return None
        
        # outfit: (pant, outer, shirt) 순서라고 가정
        pants, outers, shirt = outfit
        person_img = Image.open(person_img_path).convert("RGB")
        
        print(f"\n🎯 조합 #{idx + 1}")
        print(f" - shirt: {shirt['path']}")
        print(f" - pant : {pants['path']}")
        print(f" - outer: {outers['path']}")
        
        try:
            # 1. 상의 적용
            res_top = self.pipeline(
                person_image=person_img, 
                garment_image=Image.open(shirt['path']).convert("RGB"), 
                category="tops"
            )
            top_path = f"{output_prefix}_q{idx}_top.png"
            res_top.images[0].save(top_path)
            
            # 2. 하의 적용
            res_final = self.pipeline(
                person_image=res_top.images[0], 
                garment_image=Image.open(pants['path']).convert("RGB"), 
                category="bottoms"
            )
            final_path = f"{output_prefix}_q{idx}_top_bottom.png"
            res_final.images[0].save(final_path)
            
            print("✅ Saved:", top_path, final_path)
            
            return {
                "top_path": top_path,
                "final_path": final_path
            }
        except Exception as e:
            print(f"[ERROR] VTON 이미지 생성 실패: {e}")
            return None


class Visualizer:
    """결과 시각화(이미지 출력) 유틸리티 클래스"""
    
    @staticmethod
    def show_recommendations(results: Dict[str, List[Dict]], top_k: int = 3):
        """추천 아이템들을 격자 형태로 출력"""
        cats = [c for c in results if results[c]]
        if not cats: return
        
        fig, axes = plt.subplots(len(cats), top_k, figsize=(4*top_k, 4*len(cats)))
        if len(cats) == 1: axes = np.expand_dims(axes, axis=0)
        
        for i, cat in enumerate(cats):
            for j in range(top_k):
                ax = axes[i, j]
                if j < len(results[cat]):
                    item = results[cat][j]
                    ax.imshow(Image.open(item['path']))
                    score = item.get('score', 0.0)
                    ax.set_title(f"{cat} Rank {j+1}\nSc: {score:.4f}", fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_top_combinations(top_combinations: List[Dict], num_to_show: int = 3):
        """상위 조합들을 시각화합니다."""
        for rank, result in enumerate(top_combinations[:num_to_show], 1):
            combo = result["combination"]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 각 카테고리별 이미지 표시
            categories = ["pant", "outer", "shirt"]
            for idx, (item, cat) in enumerate(zip(combo, categories)):
                try:
                    img = Image.open(item['path']).convert("RGB")
                    axes[idx].imshow(img)
                    axes[idx].axis("off")
                    axes[idx].set_title(f"{cat}\n{item['path'].split('/')[-1]}", fontsize=10)
                except Exception as e:
                    axes[idx].text(0.5, 0.5, f"Error loading\n{item['path'].split('/')[-1]}",
                                ha='center', va='center')
                    axes[idx].axis("off")

            plt.suptitle(
                f"순위 #{rank} (조합 #{result['combination_idx'] + 1})\n"
                f"최종: {result['final_score']:.4f} | "
                f"전체조화: {result['harmony_score']:.4f} | "
                f"개별평균: {result['individual_avg']:.4f}",
                fontsize=12,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def show_vton_result(vton_result: Dict, query_text: str, idx: int):
        """VTON 생성 이미지 시각화"""
        if not vton_result or 'final_path' not in vton_result:
            return
        
        final_img = Image.open(vton_result["final_path"])
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img)
        plt.axis("off")
        plt.title(f"Query {idx+1} - Fashion-VTON 최종 결과\n{query_text}",
                 fontsize=12, fontweight='bold')
        plt.show()


# --- Main Pipeline ---

def run_fashion_system():
    """전체 패션 추천 및 코디 파이프라인 시뮬레이션 코드"""
    print("=" * 60)
    print("1. FashionExpert 초기화")
    print("=" * 60)
    
    config = FashionConfig()
    encoder = CLIPEncoder(config)
    db = FashionDBManager(config)
    recommender = FashionRecommender(config, encoder, db)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    
    # 데이터 준비
    print("\n" + "=" * 60)
    print("2. 데이터베이스 및 아이템 로드")
    print("=" * 60)
    # db.initialize_db() # 최초 1회 실행 필요 (주석 해제하여 사용)
    recommender.load_user_items()
    
    print("\n" + "=" * 60)
    print("3. ⭐ChromaDB에서 레퍼런스 임베딩 로드")
    print("=" * 60)
    recommender.load_styles()
    
    # 테스트 케이스 (outer 추가)
    test_cases = [
        {
            "original_query": "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?",
            "analyzed_intent": {"style": "스트릿", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["oversized black hoodie", "wide cargo pants", "baggy street fit", "urban style"]
        },
        {
            "original_query": "소개팅 나가는데 깔끔하게 입고 싶어",
            "analyzed_intent": {"style": "미니멀", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["white linen shirt", "beige straight slacks", "clean minimalist style", "neat fit"]
        },
        {
            "original_query": "날씨 좋은데 가볍게 산책할 때 입을 만한 거",
            "analyzed_intent": {"style": "캐주얼", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["vibrant blue knit", "light blue denim", "fresh daily look", "bright casual"]
        },
    ]
    
    all_results = {}
    
    for idx, test in enumerate(test_cases):
        print("\n" + "=" * 60)
        print(f"4-{idx+1}. 쿼리 처리: {test['original_query']}")
        print("=" * 60)
        
        # 1. 추천 아이템 검색
        recs = recommender.recommend(test, top_k=3)
        
        # 결과 출력
        for cat, items in recs.items():
            print(f"\n  [{cat}] Top 3 Results:")
            for j, item in enumerate(items, 1):
                f_score = item.get('score', 0.0)
                t_sim = item.get('text_sim', 0.0)
                s_sim = item.get('style_sim', 0.0)
                print(f"    {j}. {item['id']} -> Total: {f_score:.4f} (Text: {t_sim:.2f}, Style: {s_sim:.2f})")
        
        # 2. 추천 결과 시각화
        Visualizer.show_recommendations(recs, top_k=3)
        
        # 3. 조합 생성 및 평가
        print(f"\n5-{idx+1}. 조합 생성 및 평가")
        print("-" * 60)
        
        combos = planner.generate_combinations(recs, top_n=3)
        
        if combos:
            best_outfits = planner.evaluate_outfits(
                combos, 
                recommender.style_profiles, 
                test['analyzed_intent']['style'],
                overall_weight=0.7
            )
            
            if best_outfits:
                # 4. 조합 시각화
                print(f"\n6-{idx+1}. 조합 시각화")
                print("-" * 60)
                Visualizer.show_top_combinations(best_outfits, num_to_show=3)
                
                # 5. VTON 이미지 생성
                print(f"\n7-{idx+1}. Fashion-VTON 이미지 생성")
                print("-" * 60)
                
                vton_result = vton.try_on(
                    person_img_path="/content/model.webp",
                    outfit=best_outfits[0]['combination'],
                    output_prefix=f"/content/output",
                    idx=idx
                )
                
                if vton_result:
                    Visualizer.show_vton_result(vton_result, test['original_query'], idx)
                    print(f"✅ Query {idx+1} VTON 이미지 생성 완료!")
                
                all_results[idx] = {
                    "query": test,
                    "recommendations": recs,
                    "combinations": best_outfits[:3],
                    "vton_result": vton_result
                }
    
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    all_results = run_fashion_system()
        
        if not (pants and outers and shirts): 
            print("[WARN] 조합 생성에 필요한 카테고리가 부족합니다.")
            return []
        
        combinations = list(itertools.product(pants, outers, shirts))
        print(f"총 {len(combinations)}개의 조합이 생성되었습니다.")
        print(f"- pant: {len(pants)}개")
        print(f"- outer: {len(outers)}개")
        print(f"- shirt: {len(shirts)}개")
        
        return combinations

    def evaluate_outfits(self, combinations: List[Tuple], style_profiles: Dict[str, torch.Tensor], 
                        target_style: str, overall_weight: float = 0.7) -> List[Dict]:
        """조합된 세트의 전체 조화도 및 스타일 부합도 평가 (가중치 파라미터 추가)"""
        target_style = unicodedata.normalize("NFC", target_style)
        if target_style not in style_profiles: 
            print(f"[WARN] 타겟 스타일 '{target_style}'이 프로필에 없습니다.")
            return []
        
        # 스타일 레퍼런스 정규화
        style_embs = F.normalize(style_profiles[target_style].to(torch.float32), dim=-1)
        results = []
        
        print(f"총 {len(combinations)}개 조합 평가 중...")
        
        for i, combo in enumerate(combinations):
            if (i + 1) % 5 == 0:
                print(f"  진행: {i + 1}/{len(combinations)}")
            
            # 3장 평균 임베딩 (전체 조화)
            embs = torch.stack([item['embedding'] for item in combo]).to(torch.float32)
            combo_emb = F.normalize(embs.mean(dim=0), dim=-1)
            harmony_sim = torch.max(combo_emb @ style_embs.T).item()
            
            # 개별 아이템 vs 스타일 평균 (개별 품질)
            indiv_sims = []
            for item in combo:
                item_emb = F.normalize(item['embedding'].to(torch.float32), dim=-1)
                sim = torch.max(item_emb @ style_embs.T).item()
                indiv_sims.append(sim)
            indiv_avg = np.mean(indiv_sims)
            
            # 최종 점수 (overall_weight 파라미터 적용)
            final_score = overall_weight * harmony_sim + (1 - overall_weight) * indiv_avg
            
            results.append({
                "combination": combo,
                "combination_idx": i,
                "final_score": final_score,
                "harmony_score": harmony_sim,
                "individual_avg": indiv_avg,
                "pant": combo[0]['path'].split('/')[-1],
                "outer": combo[1]['path'].split('/')[-1],
                "shirt": combo[2]['path'].split('/')[-1]
            })
            
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 상위 결과 출력
        print(f"\n평가 완료! 상위 3개 조합:")
        for rank, result in enumerate(results[:3], 1):
            print(f"\n{rank}위 (조합 #{result['combination_idx'] + 1}):")
            print(f"  최종 점수: {result['final_score']:.4f}")
            print(f"  전체 조화 유사도: {result['harmony_score']:.4f}")
            print(f"  개별 평균 유사도: {result['individual_avg']:.4f}")
            print(f"  - pant: {result['pant']}")
            print(f"  - outer: {result['outer']}")
            print(f"  - shirt: {result['shirt']}")
        
        return results


class VTONManager:
    """가상 피팅(Virtual Try-On) 실행 및 이미지 생성 클래스"""
    def __init__(self, weights_dir: str = "./weights"):
        self.pipeline = None
        try:
            from fashn_vton import TryOnPipeline
            self.pipeline = TryOnPipeline(weights_dir=weights_dir)
            print("✅ Fashion-VTON 파이프라인 로드 완료! (가중치는 재사용됩니다)")
        except ImportError:
            print("[WARN] fashn_vton 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print(f"[WARN] Fashion-VTON 로드 실패: {e}")

    def try_on(self, person_img_path: str, outfit: Tuple, output_prefix: str, idx: int = 0) -> Optional[Dict]:
        """상의, 하위 순차적으로 가상 피팅 적용"""
        if not self.pipeline: 
            print("[WARN] VTON 파이프라인이 로드되지 않았습니다.")
            return None
        
        # outfit: (pant, outer, shirt) 순서라고 가정
        pants, outers, shirt = outfit
        person_img = Image.open(person_img_path).convert("RGB")
        
        print(f"\n🎯 조합 #{idx + 1}")
        print(f" - shirt: {shirt['path']}")
        print(f" - pant : {pants['path']}")
        print(f" - outer: {outers['path']}")
        
        try:
            # 1. 상의 적용
            res_top = self.pipeline(
                person_image=person_img, 
                garment_image=Image.open(shirt['path']).convert("RGB"), 
                category="tops"
            )
            top_path = f"{output_prefix}_q{idx}_top.png"
            res_top.images[0].save(top_path)
            
            # 2. 하의 적용
            res_final = self.pipeline(
                person_image=res_top.images[0], 
                garment_image=Image.open(pants['path']).convert("RGB"), 
                category="bottoms"
            )
            final_path = f"{output_prefix}_q{idx}_top_bottom.png"
            res_final.images[0].save(final_path)
            
            print("✅ Saved:", top_path, final_path)
            
            return {
                "top_path": top_path,
                "final_path": final_path
            }
        except Exception as e:
            print(f"[ERROR] VTON 이미지 생성 실패: {e}")
            return None


class Visualizer:
    """결과 시각화(이미지 출력) 유틸리티 클래스"""
    
    @staticmethod
    def show_recommendations(results: Dict[str, List[Dict]], top_k: int = 3):
        """추천 아이템들을 격자 형태로 출력"""
        cats = [c for c in results if results[c]]
        if not cats: return
        
        fig, axes = plt.subplots(len(cats), top_k, figsize=(4*top_k, 4*len(cats)))
        if len(cats) == 1: axes = np.expand_dims(axes, axis=0)
        
        for i, cat in enumerate(cats):
            for j in range(top_k):
                ax = axes[i, j]
                if j < len(results[cat]):
                    item = results[cat][j]
                    ax.imshow(Image.open(item['path']))
                    score = item.get('score', 0.0)
                    ax.set_title(f"{cat} Rank {j+1}\nSc: {score:.4f}", fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_top_combinations(top_combinations: List[Dict], num_to_show: int = 3):
        """상위 조합들을 시각화합니다."""
        for rank, result in enumerate(top_combinations[:num_to_show], 1):
            combo = result["combination"]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 각 카테고리별 이미지 표시
            categories = ["pant", "outer", "shirt"]
            for idx, (item, cat) in enumerate(zip(combo, categories)):
                try:
                    img = Image.open(item['path']).convert("RGB")
                    axes[idx].imshow(img)
                    axes[idx].axis("off")
                    axes[idx].set_title(f"{cat}\n{item['path'].split('/')[-1]}", fontsize=10)
                except Exception as e:
                    axes[idx].text(0.5, 0.5, f"Error loading\n{item['path'].split('/')[-1]}",
                                ha='center', va='center')
                    axes[idx].axis("off")

            plt.suptitle(
                f"순위 #{rank} (조합 #{result['combination_idx'] + 1})\n"
                f"최종: {result['final_score']:.4f} | "
                f"전체조화: {result['harmony_score']:.4f} | "
                f"개별평균: {result['individual_avg']:.4f}",
                fontsize=12,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def show_vton_result(vton_result: Dict, query_text: str, idx: int):
        """VTON 생성 이미지 시각화"""
        if not vton_result or 'final_path' not in vton_result:
            return
        
        final_img = Image.open(vton_result["final_path"])
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img)
        plt.axis("off")
        plt.title(f"Query {idx+1} - Fashion-VTON 최종 결과\n{query_text}",
                 fontsize=12, fontweight='bold')
        plt.show()


# --- Main Pipeline ---

def run_fashion_system():
    """전체 패션 추천 및 코디 파이프라인 시뮬레이션 코드"""
    print("=" * 60)
    print("1. FashionExpert 초기화")
    print("=" * 60)
    
    config = FashionConfig()
    encoder = CLIPEncoder(config)
    db = FashionDBManager(config)
    recommender = FashionRecommender(config, encoder, db)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    
    # 데이터 준비
    print("\n" + "=" * 60)
    print("2. 데이터베이스 및 아이템 로드")
    print("=" * 60)
    # db.initialize_db() # 최초 1회 실행 필요 (주석 해제하여 사용)
    recommender.load_user_items()
    
    print("\n" + "=" * 60)
    print("3. ⭐ChromaDB에서 레퍼런스 임베딩 로드")
    print("=" * 60)
    recommender.load_styles()
    
    # 테스트 케이스 (outer 추가)
    test_cases = [
        {
            "original_query": "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?",
            "analyzed_intent": {"style": "스트릿", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["oversized black hoodie", "wide cargo pants", "baggy street fit", "urban style"]
        },
        {
            "original_query": "소개팅 나가는데 깔끔하게 입고 싶어",
            "analyzed_intent": {"style": "미니멀", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["white linen shirt", "beige straight slacks", "clean minimalist style", "neat fit"]
        },
        {
            "original_query": "날씨 좋은데 가볍게 산책할 때 입을 만한 거",
            "analyzed_intent": {"style": "캐주얼", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["vibrant blue knit", "light blue denim", "fresh daily look", "bright casual"]
        },
    ]
    
    all_results = {}
    
    for idx, test in enumerate(test_cases):
        print("\n" + "=" * 60)
        print(f"4-{idx+1}. 쿼리 처리: {test['original_query']}")
        print("=" * 60)
        
        # 1. 추천 아이템 검색
        recs = recommender.recommend(test, top_k=3)
        
        # 결과 출력
        for cat, items in recs.items():
            print(f"\n  [{cat}] Top 3 Results:")
            for j, item in enumerate(items, 1):
                f_score = item.get('score', 0.0)
                t_sim = item.get('text_sim', 0.0)
                s_sim = item.get('style_sim', 0.0)
                print(f"    {j}. {item['id']} -> Total: {f_score:.4f} (Text: {t_sim:.2f}, Style: {s_sim:.2f})")
        
        # 2. 추천 결과 시각화
        Visualizer.show_recommendations(recs, top_k=3)
        
        # 3. 조합 생성 및 평가
        print(f"\n5-{idx+1}. 조합 생성 및 평가")
        print("-" * 60)
        
        combos = planner.generate_combinations(recs, top_n=3)
        
        if combos:
            best_outfits = planner.evaluate_outfits(
                combos, 
                recommender.style_profiles, 
                test['analyzed_intent']['style'],
                overall_weight=0.7
            )
            
            if best_outfits:
                # 4. 조합 시각화
                print(f"\n6-{idx+1}. 조합 시각화")
                print("-" * 60)
                Visualizer.show_top_combinations(best_outfits, num_to_show=3)
                
                # 5. VTON 이미지 생성
                print(f"\n7-{idx+1}. Fashion-VTON 이미지 생성")
                print("-" * 60)
                
                vton_result = vton.try_on(
                    person_img_path="/content/model.webp",
                    outfit=best_outfits[0]['combination'],
                    output_prefix=f"/content/output",
                    idx=idx
                )
                
                if vton_result:
                    Visualizer.show_vton_result(vton_result, test['original_query'], idx)
                    print(f"✅ Query {idx+1} VTON 이미지 생성 완료!")
                
                all_results[idx] = {
                    "query": test,
                    "recommendations": recs,
                    "combinations": best_outfits[:3],
                    "vton_result": vton_result
                }
    
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    all_results = run_fashion_system()