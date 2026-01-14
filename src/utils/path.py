from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_CONFIG_PATH = PROJECT_ROOT / "src/inference/inference_config.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "src/training_pipeline/training_config.yaml"


