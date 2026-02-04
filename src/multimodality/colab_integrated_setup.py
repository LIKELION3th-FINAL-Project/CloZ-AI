# --- 1. 라이브러리 설치 (코랩 환경) ---
import os
import sys

# 필요한 라이브러리 설치 (코랩의 경우 앞의 !를 빼고 os.system으로 실행 가능)
print("📦 라이브러리 설치 중...")
os.system("pip install -q transformers torch pillow matplotlib chromadb")

# --- 2. 모듈화된 파일 생성 (코랩 로컬 경로에 구성) ---
print("🗂️ 패키지 구조 생성 중...")
os.makedirs("fashion_engine", exist_ok=True)

# 2-1. config.py
with open("fashion_engine/config.py", "w", encoding="utf-8") as f:
    f.write('''from dataclasses import dataclass
import torch

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 경로 관리 클래스"""
    # 사용자의 드라이브 마운트 경로에 맞게 적절히 수정하세요
    base_dir: str = "/content/drive/MyDrive/LikeLion/최종 프로젝트 개인 임시/테스트용"
    db_path: str = "/content/fashion_items.db"
    chromadb_dir: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb"
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
''')

# 2-2. encoder.py
with open("fashion_engine/encoder.py", "w", encoding="utf-8") as f:
    f.write('''import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import Any
from .config import FashionConfig

class CLIPEncoder:
    def __init__(self, config: FashionConfig):
        self.config = config
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model = CLIPModel.from_pretrained(config.model_name).to(config.device).eval()
        if config.use_fp16 and config.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def _extract_features(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "image_embeds"): features = outputs.image_embeds
        elif hasattr(outputs, "text_embeds"): features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"): features = outputs.pooler_output
        else: features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        if features.dim() == 3: features = features.mean(dim=1)
        if features.dim() == 2 and features.size(0) == 1: features = features[0]
        return F.normalize(features.float(), dim=-1).cpu()

    def encode_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.config.device)
        if self.config.use_fp16 and self.config.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        outputs = self.model.get_image_features(**inputs)
        return self._extract_features(outputs)

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)
        outputs = self.model.get_text_features(**inputs)
        return self._extract_features(outputs)
''')

# 2-3. db_manager.py
with open("fashion_engine/db_manager.py", "w", encoding="utf-8") as f:
    f.write('''import os
import sqlite3
from .config import FashionConfig

class FashionDBManager:
    def __init__(self, config: FashionConfig):
        self.config = config
        self.categories_map = {"하의": "pant", "아우터": "outer", "상의": "shirt"}

    def initialize_db(self):
        if os.path.exists(self.config.db_path): os.remove(self.config.db_path)
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id TEXT PRIMARY KEY, broad_cat TEXT NOT NULL, detail_cat TEXT)")
        cur.execute("CREATE INDEX idx_items_broad_cat ON items(broad_cat)")
        items = []
        for kor_name, eng_name in self.categories_map.items():
            folder = os.path.join(self.config.base_dir, kor_name)
            if not os.path.isdir(folder): continue
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff')):
                    items.append((f"{eng_name}/{f}", eng_name, None))
        cur.executemany("INSERT INTO items VALUES (?, ?, ?)", items)
        conn.commit()
        conn.close()
        print(f"[DB] {len(items)} items initialized.")

    def get_path_from_id(self, db_id: str) -> str:
        eng_cat, filename = db_id.split('/', 1)
        inv_map = {v: k for k, v in self.categories_map.items()}
        return os.path.join(self.config.base_dir, inv_map[eng_cat], filename)
''')

# 2-4. recommender.py
with open("fashion_engine/recommender.py", "w", encoding="utf-8") as f:
    f.write('''import sqlite3
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
    def __init__(self, config: FashionConfig, encoder: CLIPEncoder, db: FashionDBManager):
        self.config = config
        self.encoder = encoder
        self.db = db
        self.wardrobe = []
        self.style_profiles = {}

    def load_user_items(self):
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        rows = cur.execute("SELECT id, broad_cat FROM items").fetchall()
        conn.close()
        self.wardrobe = []
        for db_id, cat in rows:
            path = self.db.get_path_from_id(db_id)
            if os.path.exists(path):
                self.wardrobe.append({"id": db_id, "category": cat, "path": path, "embedding": self.encoder.encode_image(path)})
        print(f"[INFO] Loaded {len(self.wardrobe)} items.")

    def load_styles(self):
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
            print(f"[ChromaDB] {len(self.style_profiles)} styles loaded.")
        except Exception as e: print(f"Style Load Failed: {e}")

    def recommend(self, query_data: Dict, top_k: int = 5) -> Dict[str, List[Dict]]:
        intent = query_data.get("analyzed_intent", {})
        target_style = unicodedata.normalize('NFC', intent.get("style", "캐주얼"))
        cat_map = {"top": "shirt", "bottom": "pant", "outer": "outer"}
        target_cats = [cat_map.get(c, c) for c in intent.get("categories", [])] or ["shirt", "pant", "outer"]
        keywords = query_data.get("expanded_keywords", [query_data.get("original_query", "")])
        q_embs = [self.encoder.encode_text(kw) for kw in keywords]
        avg_q_emb = F.normalize(torch.stack(q_embs).mean(dim=0), dim=-1)
        results = defaultdict(list)
        for item in self.wardrobe:
            if item['category'] not in target_cats: continue
            text_sim = float((avg_q_emb * item['embedding']).sum())
            style_final = 0.5
            if target_style in self.style_profiles:
                style_scores = []
                item_emb_f32 = item['embedding'].to(torch.float32)
                for s_name, s_embs in self.style_profiles.items():
                    s_embs_norm = F.normalize(s_embs.to(torch.float32), dim=-1)
                    sims = torch.matmul(s_embs_norm, item_emb_f32)
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
            results[item['category']].append({**item, "score": combined, "text_sim": text_sim, "style_sim": style_final})
        for cat in results:
            results[cat].sort(key=lambda x: x['score'], reverse=True)
            results[cat] = results[cat][:top_k]
        return dict(results)
''')

# 2-5. planner.py
with open("fashion_engine/planner.py", "w", encoding="utf-8") as f:
    f.write('''import itertools
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple
from .encoder import CLIPEncoder

class OutfitPlanner:
    def __init__(self, encoder: CLIPEncoder):
        self.encoder = encoder

    def generate_combinations(self, recommendations: Dict[str, List[Dict]], top_n: int = 3) -> List[Tuple]:
        pants = recommendations.get("pant", [])[:top_n]
        outers = recommendations.get("outer", [])[:top_n]
        shirts = recommendations.get("shirt", [])[:top_n]
        if not (pants and outers and shirts): return []
        combinations = list(itertools.product(pants, outers, shirts))
        print(f"Generated {len(combinations)} combinations.")
        return combinations

    def evaluate_outfits(self, combinations: List[Tuple], style_profiles: Dict[str, torch.Tensor], target_style: str, overall_weight: float = 0.7) -> List[Dict]:
        target_style = unicodedata.normalize("NFC", target_style)
        if target_style not in style_profiles: return []
        style_embs = F.normalize(style_profiles[target_style].to(torch.float32), dim=-1)
        results = []
        for i, combo in enumerate(combinations):
            embs = torch.stack([item['embedding'] for item in combo]).to(torch.float32)
            combo_emb = F.normalize(embs.mean(dim=0), dim=-1)
            harmony_sim = torch.max(combo_emb @ style_embs.T).item()
            indiv_sims = []
            for item in combo:
                item_emb = F.normalize(item['embedding'].to(torch.float32), dim=-1)
                sim = torch.max(item_emb @ style_embs.T).item()
                indiv_sims.append(sim)
            indiv_avg = np.mean(indiv_sims)
            final_score = overall_weight * harmony_sim + (1 - overall_weight) * indiv_avg
            results.append({
                "combination": combo, "combination_idx": i, "final_score": final_score, 
                "harmony_score": harmony_sim, "individual_avg": indiv_avg,
                "pant": combo[0]['path'].split('/')[-1], "outer": combo[1]['path'].split('/')[-1], "shirt": combo[2]['path'].split('/')[-1]
            })
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
''')

# 2-6. vton.py, visualizer.py, __init__.py 등 나머지 파일들... (생략 및 자동생성)
with open("fashion_engine/vton.py", "w", encoding="utf-8") as f:
    f.write('class VTONManager: pass') # VTON 라이브러리 부재 대비용 더미
with open("fashion_engine/visualizer.py", "w", encoding="utf-8") as f:
    f.write('''import matplotlib.pyplot as plt
from PIL import Image
class Visualizer:
    @staticmethod
    def show_recommendations(results, top_k=3): print("Showing recs...")
    @staticmethod
    def show_top_combinations(combos, num=3): print("Showing outfits...")
''')
with open("fashion_engine/__init__.py", "w", encoding="utf-8") as f:
    f.write('from .config import FashionConfig\nfrom .encoder import CLIPEncoder\nfrom .db_manager import FashionDBManager\nfrom .recommender import FashionRecommender\nfrom .planner import OutfitPlanner\nfrom .vton import VTONManager\nfrom .visualizer import Visualizer')

print("✅ 패션 엔진 패키지 구축 완료!")

# --- 3. 실행 테스트 (Main Simulation) ---
print("\n🚀 시스템 시뮬레이션 시작...")

# 현재 경로를 sys.path에 추가하여 방금 만든 패키지를 임포트 가능하게 함
sys.path.append(os.getcwd())

from fashion_engine import (
    FashionConfig, CLIPEncoder, FashionDBManager, 
    FashionRecommender, OutfitPlanner, Visualizer
)

def run_test():
    # 1. 초기화
    config = FashionConfig()
    encoder = CLIPEncoder(config)
    db = FashionDBManager(config)
    recommender = FashionRecommender(config, encoder, db)
    planner = OutfitPlanner(encoder)
    
    # 2. 데이터 로드 (실제 경로에 이미지 파일 등이 있어야 작동합니다)
    print("아이템 및 스타일 로딩 중...")
    # db.initialize_db() 
    # recommender.load_user_items()
    # recommender.load_styles()
    
    # 3. 샘플 쿼리 테스트
    test = {
        "original_query": "스트릿 패션 추천해줘",
        "analyzed_intent": {"style": "스트릿", "categories": ["top", "bottom", "outer"]},
        "expanded_keywords": ["oversized hoodie", "cargo pants"]
    }
    
    # 데이터가 준비되지 않은 상태라면 여기까지만 확인
    print("✅ 모든 모듈이 정상적으로 임포트되었습니다. 설정을 확인하고 데이터 시뮬레이션을 진행하세요.")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"⚠️ 실행 중 오류(데이터 경로 확인 필요): {e}")

print("\n" + "="*30)
print("🏁 통합 실행 프로세스 완료")
print("="*30)
