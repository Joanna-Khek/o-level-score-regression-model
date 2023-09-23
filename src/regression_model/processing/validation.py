from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.regression_model.config.core import config

class ScoreDataInputSchema(BaseModel):
    index: Optional[int]
    number_of_siblings: Optional[int]
    direct_admission: Optional[str]
    CCA: Optional[str]
    learning_style: Optional[str]
    student_id: Optional[str]
    gender: Optional[str]
    tuition: Optional[str]
    final_test: Optional[float]
    n_male: Optional[float]
    n_female: Optional[float]
    age: Optional[float]
    hours_per_week: Optional[float]
    attendance_rate: Optional[float]
    sleep_time: Optional[str]
    wake_time: Optional[str]
    mode_of_transport: Optional[str]
    bag_color: Optional[str]

class MultipleScoreDataInput(BaseModel):
    inputs: List[ScoreDataInputSchema]


def drop_na_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check if missing values within specified threshold.
    If within, proceed to drop na inputs from dataset. 
    If exceed threshold, raise Exception"""
    validated_data = input_data.copy()

    # Check if missing values > threshold
    vars_with_na_above_thres = [var for var in config.model.features
                                if validated_data[var].isnull().sum()/validated_data.shape[0] > \
                                    config.model.missing_thres]

    if len(vars_with_na_above_thres) != 0:
        raise Exception(f"{vars_with_na_above_thres} have missing values above threshold of {config.model.missing_thres*100}%")

    # If all missing values <= threshold, proceed to remove missing rows from dataset
    vars_with_na_within_thres = [var for var in config.model.features
                                if 0 < validated_data[var].isnull().sum()/validated_data.shape[0] \
                                    <= config.model.missing_thres]

    validated_data.dropna(subset=vars_with_na_within_thres, inplace=True)

    return validated_data

def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values"""

    validated_data = drop_na_inputs(input_data)
    errors = None
    try:
        # pydantic can only validate if inputs is a dictionary
        MultipleScoreDataInput(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()
    return validated_data, errors