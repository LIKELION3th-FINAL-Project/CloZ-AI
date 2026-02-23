from ..utils.load import load_json, load_config
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

def build_system_prompt(sys_prompt, prompt_context) -> dict:
    if isinstance(prompt_context, str):
        model_response_text = prompt_context
        json_template_text = json.dumps({}, ensure_ascii=False, indent=2)
    else:
        json_template_text = json.dumps(
            prompt_context if prompt_context is not None else {},
            ensure_ascii=False,
            indent=2,
        )
        model_response_text = json_template_text

    content = sys_prompt.format(
        json_template=json_template_text,
        model_response=model_response_text,
    )
    sys_formatted_prompt = {
        "role": "system",
        "content": content
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
        self.json_template_path = Path(__file__).parents[3] / "configs" / "json_template.json"
        self.llm_understand_config_path = Path(__file__).parents[3] / "configs" / "llm_base_understand.yaml"
        self.json_template = load_json(self.json_template_path)
        self.config_file = load_config(self.llm_understand_config_path)
        base_prompt = self.config_file["model"]["sys_prompt"]
        self.initial_sys_prompt = self.config_file["model"].get("initial_sys_prompt", base_prompt)
        self.clarify_sys_prompt = self.config_file["model"].get("clarify_sys_prompt", base_prompt)
        self.request_additional_info_sys_prompt = self.config_file["model"].get(
            "request_additional_info_sys_prompt", base_prompt
        )
        self.model_name = self.config_file["model"]["model_name"]
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.reasoning_effort = self.config_file["model"]["reasoning_effort"]
        self.stream = self.config_file["model"]["stream"]
        self.temperature = self.config_file["model"]["temperature"]
        self.client = OpenAI(api_key = self.api_key, base_url = self.config_file["model"]["base_url"])
    
    def chat(self, user_prompt: str) -> str:
        return self.initial_chat(user_prompt)

    def initial_chat(self, user_prompt):
        messages = [build_system_prompt(self.initial_sys_prompt, self.json_template)]
        messages.append(build_user_prompt(user_prompt))
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            reasoning_effort = self.reasoning_effort,
        )
        logger.info(f"MODEL RESP: {response.choices[0].message.content}")
        return response.choices[0].message.content
    
    def request_additional_info_chat(self, first_model_response, user_prompt: str = ""):
        messages = [build_system_prompt(self.request_additional_info_sys_prompt, first_model_response)]
        messages.append(build_user_prompt(user_prompt or "추가 정보를 요청하세요."))
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            reasoning_effort = self.reasoning_effort,
        )
        logger.info(f"MODEL RESP: {response.choices[0].message.content}")
        return response.choices[0].message.content
    
    def clarify_chat(self, first_model_response, user_prompt: str = ""):
        messages = [build_system_prompt(self.clarify_sys_prompt, first_model_response)]
        messages.append(build_user_prompt(user_prompt))
        
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            reasoning_effort = self.reasoning_effort,
        )
        logger.info(f"MODEL RESP: {response.choices[0].message.content}")
        return response.choices[0].message.content

# Understand model 동작 테스트용
if __name__ == "__main__":
    test_user_prompt = "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?"
    agent = UnderstandModel()
    resp = agent.chat(test_user_prompt)
    print(resp)
