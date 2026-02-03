from utils.load import load_json, load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
import torch
import ast
import json
import re
import sys
import os

load_dotenv()

def build_messages(sys_prompt, json_template, user_prompt):
    return [
        {
            "role": "system",
            "content": sys_prompt.format(
                json_template = json.dumps(
                    json_template,
                    ensure_ascii = False,
                    indent = 2,
                ),
            )
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

def extract_json_file(text):
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

class UnderstandModel:
    def __init__(self):
        self.json_template = load_json("json_template.json")
        self.config_file = load_config("../../configs/llm_base_understand.yaml")
        self.model_name = self.config_file["model"]["model_name"]
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.reasoning_effort = self.config_file["model"]["reasoning_effort"]
        self.stream = self.config_file["model"]["stream"]
        self.temperature = self.config_file["model"]["temperature"]
        sys_prompt = self.config_file["model"]["sys_prompt"].format(
            json_template = json.dumps(
                self.json_template,
                ensure_ascii = False,
                indent = 2
                )
        )
        self.messages = [
            {
                "role": "system",
                "content": sys_prompt
            }
        ]
        self.max_turns = self.config_file.get("chat", {}).get("max_turns", 10)
        
    def run_openai_stream(self, user_prompt):
        if not self.api_key:
            raise RuntimeError("UPSTAGE_API_KEY가 .env에 없습니다.")
        
        client = OpenAI(api_key = self.api_key, base_url = self.config_file["model"]["base_url"])
        messages = build_messages(
            self.config_file["model"]["sys_prompt"],
            self.json_template,
            user_prompt,
        )
        stream = client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            reasoning_effort = self.reasoning_effort,
            stream = self.stream,
            temperature = self.temperature,
        )
        full_text_chunks = []
        
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                full_text_chunks.append(delta)

        full_text = "".join(full_text_chunks)
        logger.info(f"Full response: {full_text}")

        return full_text