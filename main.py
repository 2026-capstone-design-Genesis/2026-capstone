import argparse
import yaml

from modules.auth import get_hf_token
from pipeline import run_llmvs_pipeline
from modules.model_manager import get_global_model_manager


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hf_token = get_hf_token()
    model_manager = get_global_model_manager()
    print("[PRELOAD] Loading models before video processing...")
    model_manager.preload_for_config(cfg, hf_token=hf_token)
    result = run_llmvs_pipeline(cfg, hf_token=hf_token, model_manager=model_manager)
    print(result)


if __name__ == "__main__":
    main()
