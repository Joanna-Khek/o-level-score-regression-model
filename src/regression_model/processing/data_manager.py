from typing import List
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import sqlite3
from src.regression_model import __version__ as _version
from src.regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def load_dataset(file_name: str) -> pd.DataFrame:
    db = sqlite3.connect(Path(f"{DATASET_DIR}/{file_name}"))
    data = pd.read_sql_query(sql="SELECT * from score", con=db)
    
     # drop na if column is found
    try:
        data.dropna(subset=['final_test'], inplace=True)
    except:
        pass

    return data

def load_pipeline(file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def save_pipeline(pipeline_to_persist: Pipeline) -> None:
    # Prepare versioned save file name
    save_file_name = f"{config.app.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)



