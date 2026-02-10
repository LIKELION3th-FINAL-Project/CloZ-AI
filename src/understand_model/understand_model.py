from utils.load import load_json, load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import torch
import ast
import json
import re
import sys
import os

load_dotenv()

def extract_json_format(text):
    match_ = re.search(r"\{.*\}", text, flags = re.DOTALL)
        
    if not match_:
        return None
    raw = match_.group(0).strip()
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
        
def make_json_file(json_format_string):
    if json_format_string is None:
        raise ValueError("모델 출력에서 JSON 스트링을 찾지 못했습니다.")
    with open("model_generated_answer.json", "w", encoding = "utf-8") as f:
        json.dump(json_format_string, f, ensure_ascii = False, indent = 2)
    return

def build_system_prompt(sys_prompt, json_template) -> dict:
    sys_formatted_prompt = {
        "role": "system",
        "content": sys_prompt.format(
            json_template = json.dumps(
                json_template,
                ensure_ascii = False,
                indent = 2,
            ),
        )
    }
    return sys_formatted_prompt

def build_user_prompt(user_prompt) -> dict:
    user_formatted_prompt = {
        "role": "user",
        "content": user_prompt,
    }
    return user_formatted_prompt

def build_assistant_prompt(assistant_prompt) -> dict:
    assistant_formatted_prompt = {
        "role": "assistant",
        "content": assistant_prompt,
    }
    return assistant_formatted_prompt


class UnderstandModel:
    def __init__(self):
        self.json_template_path = Path(__file__).parents[2] / "configs" / "json_template.json"
        self.llm_understand_config_path = Path(__file__).parents[2] / "configs" / "llm_base_understand.yaml"
        self.json_template = load_json(self.json_template_path)
        self.config_file = load_config(self.llm_understand_config_path)
        self.sys_prompt = self.config_file["model"]["sys_prompt"]
        self.model_name = self.config_file["model"]["model_name"]
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.reasoning_effort = self.config_file["model"]["reasoning_effort"]
        self.stream = self.config_file["model"]["stream"]
        self.temperature = self.config_file["model"]["temperature"]
        self.client = OpenAI(api_key = self.api_key, base_url = self.config_file["model"]["base_url"])
        # self.max_turns = self.config_file.get("chat", {}).get("max_turns", 10)
    
    # def chat(self, messages: list[dict]) -> str:
    def chat(self, user_prompt):
        messages = [build_system_prompt(self.sys_prompt, self.json_template)]
        messages.append(build_user_prompt(user_prompt))
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            reasoning_effort = self.reasoning_effort,
        )
        logger.info(f"MODEL RESP: {response.choices[0].message.content}")
        return response.choices[0].message.content
    
    def multi_turn(self):
        messages = [build_system_prompt(self.sys_prompt, self.json_template)]
        
        for turn in range(3):
            user_input = input("사용자 입력 : ").strip()
            messages.append(build_user_prompt(user_input))
            resp = self.chat(messages = messages)
            logger.info(f"RESP: {resp}")
            messages.append(build_assistant_prompt(resp))
            logger.info(f"TURN {turn + 1} 완료.")

if __name__ == "__main__":
    test_user_prompt = "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?"
    agent = UnderstandModel()
    resp = agent.chat(test_user_prompt)
    print(resp)