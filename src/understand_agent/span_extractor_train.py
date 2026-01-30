# import torch
from utils.load import load_config
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset
import torch
import yaml
import sys

config = load_config("../../configs/span_extractor.yaml")

class SpanExtractor():
    def __init__(self):
        self.model_name = config["train"]["model_name"]
        self.data
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.datasets = load_dataset()
    
    def encoding(self):
        context = "'근대적 경영학' 또는 '고전적 경영학'에서 현대적 경영학으로 전환되는 시기는 1950년대이다. 2차 세계대전을 마치고, 6.25전쟁의 시기로 유럽은 전후 재건에 집중하고, 유럽 제국주의의 식민지가 독립하여 아프리카, 아시아, 아메리카 대륙에서 신생국가가 형성되는 시기였고, 미국은 전쟁 이후 경제적 변화에 기업이 적응을 해야 하던 시기였다. 특히 1954년 피터 드러커의 저서 《경영의 실제》는 현대적 경영의 기준을 제시하여서, 기존 근대적 인사조직관리를 넘어선 현대적 인사조직관리의 전환점이 된다. 드러커는 경영자의 역할을 강조하며 경영이 현시대 최고의 예술이자 과학이라고 주장하였고 , 이 주장은 21세기 인사조직관리의 역할을 자리매김했다.\n\n현대적 인사조직관리와 근대 인사조직관리의 가장 큰 차이는 통합이다. 19세기의 영향을 받던 근대적 경영학(고전적 경영)의 흐름은 기능을 강조하였지만, 1950년대 이후의 현대 경영학은 통합을 강조하였다. 기능이 분화된 '기계적인 기업조직' 이해에서 다양한 기능을 인사조직관리의 목적, 경영의 목적을 위해서 다양한 분야를 통합하여 '유기적 기업 조직' 이해로 전환되었다. 이 통합적 접근방식은 과정, 시스템, 상황을 중심으로 하는 인사조직관리 방식을 형성했다."
        question ="현대적 인사 조직 관리의 시발점이 된 책은?"
        encodings = self.tokenizer(
            context, 
            question,
            max_length = 512,
            truncation = True,
            padding = "max_length"
        )
        return encodings
    
    def calc(self):
        encodings = self.encoding()
        encodings = {key: torch.tensor([val]) for key, val in encodings.items()}
        input_ids = encodings["input_ids"]
        attn_mask = encodings["attention_mask"]
        pred = self.model(input_ids, attention_mask = attn_mask)
        


if __name__ == "__main__":
    extractor = SpanExtractor()
    en = extractor.encoding()
    print(en)