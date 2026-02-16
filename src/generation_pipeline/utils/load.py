import yaml
import json
import os
from pathlib import Path


def _resolve_path(value, base_dir: Path):
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())

def load_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding = "utf-8") as f:
        config_file = yaml.safe_load(f)

    project_root = config_path.parent.parent
    path_like_keys = {
        "chroma_db_path",
        "chromadb_ref_embedding_dir",
        "chromadb_user_war_embedding_dir",
        "user_body_image",
        "user_bottom_image",
        "user_clothes_dir",
        "user_outer_image",
        "user_top_image",
    }

    for key in path_like_keys:
        if key not in config_file:
            continue

        env_key = key.upper()
        env_override = os.getenv(env_key)
        raw_value = env_override if env_override else config_file[key]
        config_file[key] = _resolve_path(raw_value, project_root)

    return config_file

def load_json(json_path):
    with open(json_path, "r", encoding = "utf-8") as f:
        json_file = json.load(f)
    return json_file
