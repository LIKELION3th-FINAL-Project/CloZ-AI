from utils.load import load_json, load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import torch
import ast
import json
import re
import sys

class UnderstandNL:
    def __init__(self):
        self.json_file = load_json("json_template.json")
        self.config_file = load_config("../../configs/llm_base_understand.yaml")
        self.model_name = self.config_file["model"]["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype = torch.bfloat16,
            trust_remote_code = True,
            device_map = "auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def extract_dict(self, full_text):
        match_ = re.search(r"\{.*\}", full_text, flags = re.DOTALL)
        
        if not match_:
            return None
        raw = match_.group(0)
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        
    def json_text_generator(self, prompt):
        messages = [
            {
                "role": "system",
                "content": self.config_file["model"]["sys_prompt"].format(
                    json_template = json.dumps(self.json_file, ensure_ascii = False, indent = 2)
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
        )
        input_ids = inputs["input_ids"]
        output = self.model.generate(
            input_ids.to(self.model.device),
            max_new_tokens = 256,
            do_sample = False,
        )
        
        full_decoded_text = self.tokenizer.decode(output[0])
        logger.info(f"Full decoded text: {full_decoded_text}")
        dict_result = self.extract_dict(full_decoded_text)
        return dict_result
    
    def make_json_file(self, json_format_string):
        if json_format_string is None:
            raise ValueError("모델 출력에서 JSON 스트링을 찾지 못했습니다.")
        with open("model_generated_answer.json", "w", encoding = "utf-8") as f:
            json.dump(json_format_string, f, ensure_ascii = False, indent = 2)
        return
        

if __name__ == "__main__":
    understandnl = UnderstandNL()
    texts = [
        "상의는 오버핏이고 하의는 반바지로 코디해줘",
    "아우터는 얇고 상의는 밝은 색으로 추천해줘",
    "편한 상의랑 활동적인 하의로 골라줘",
    "출근용 아우터랑 깔끔한 상의로 맞춰줘",
    "여름에 입을 시원한 상의랑 하의 추천해줘",
    "캐주얼한 아우터에 무난한 하의로 골라줘",
    "상의는 포인트 있고 하의는 심플한 걸로",
    "운동화에 어울리는 상의랑 하의 추천해줘",
    "데일리로 입기 좋은 상의랑 아우터 골라줘",
    "상의는 루즈핏, 하의는 짧은 걸로 추천해줘",
    "홍대에 갈 건데 옷 추천해줘",
    "데이트할 때 입을 만한 코디 추천해줘",
    "오늘 기분이 좀 우울해서 산뜻하게 입고 싶어",
    "친구들 만나러 갈 건데 너무 꾸민 느낌은 싫어",
    "여행 가서 편하게 입을 옷 추천해줘",
    "날씨 애매한 날에 입기 좋은 옷 골라줘",
    "사진 잘 나오는 옷으로 추천해줘",
    "요즘 감성에 맞는 코디로 골라줘",
    "회사 끝나고 바로 약속 갈 수 있는 옷 추천해줘",
    "무난한데 센스 있어 보이는 옷 골라줘",
    ]
    
    for text in texts:
        full_text = understandnl.json_text_generator(text)
    
    # text = understandnl.json_text_generator("내일 홍대 클럽 갈건데 코디 추천해줘")
    # understandnl.make_json_file(text)