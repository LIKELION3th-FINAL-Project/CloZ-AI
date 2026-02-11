import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Dict, List

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
