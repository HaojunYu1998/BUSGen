import os
import json
import argparse
from Train import train
from Eval import evaluation


def load_env(env_file=".env"):
    """Load environment variables from .env file if it exists."""
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def override_config_from_env(Config):
    """Override config paths with environment variables if set."""
    env_mapping = {
        "DATA_ROOT": "data_root",
        "CHECKPOINT_DIR": "save_weight_dir",
        "SAMPLED_DIR": "sampled_dir",
    }
    for env_key, config_key in env_mapping.items():
        env_val = os.environ.get(env_key)
        if env_val and config_key in Config:
            Config[config_key] = env_val
    return Config


def main(args):
    load_env()
    with open(args.config_file) as f:
        Config = json.load(f)
    Config = override_config_from_env(Config)
    func = {
        "Train": train,
        "Eval": evaluation
    }["Train" if not args.eval else "Eval"]
    func(Config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--eval", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
