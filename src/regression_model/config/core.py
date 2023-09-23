import src.regression_model
from pathlib import Path
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel


PACKAGE_ROOT = Path(src.regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class BestXGBParams(BaseModel):

    learning_rate: float
    max_depth: int
    min_child_weight: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    subsample: float

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    missing_thres: float
    test_size: float
    random_state: int
    sleep_wake_datetime_vars: List[str]
    encode_vars: List[str]
    features_to_drop: List[str]
    remove_row_na_features: List[str]
    cca_mappings: Dict[str, str]
    tuition_mappings: Dict[str, str]
    selected_model_name: str
    best_params: Dict[str, BestXGBParams]

class Config(BaseModel):
    """Master config object."""

    app: AppConfig
    model: ModelConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> yaml:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(CONFIG_FILE_PATH, 'r') as yaml_file:
            parsed_config = yaml.safe_load(yaml_file)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: yaml = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(**parsed_config)
    return _config

config = create_and_validate_config()