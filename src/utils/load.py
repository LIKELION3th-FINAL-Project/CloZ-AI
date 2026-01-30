import yaml
import json

def load_config(config_path):
    with open(config_path, "r", encoding = "utf-8") as f:
        config_file = yaml.safe_load(f)
    return config_file

def load_json(json_path):
    with open(json_path, "r", encoding = "utf-8") as f:
        json_file = json.load(f)
    return json_file