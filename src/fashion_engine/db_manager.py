import os
import sqlite3
from .config import FashionConfig

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
