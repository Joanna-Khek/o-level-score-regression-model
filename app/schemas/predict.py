from typing import Any, List, Optional
import numpy as np
from pydantic import BaseModel
from src.regression_model.processing.validation import ScoreDataInputSchema



class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

class MultipleScoreDataInput(BaseModel):
    inputs: List[ScoreDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "index": 5,
                        "number_of_siblings": 2,
                        "direct_admission": "Yes",
                        "CCA": "Sports",
                        "learning_style": "Visual",
                        "student_id": "ACN2BE",
                        "gender": "Female",
                        "tuition": "Yes",
                        "n_male": 30.0,
                        "n_female": 20.0,
                        "age": 15.0,
                        "hours_per_week": 10.0,
                        "attendance_rate": 95.0,
                        "sleep_time": "22:00",
                        "wake_time": "6:00",
                        "mode_of_transport": "private transport",
                        "bag_color": "yellow",
                    }
                ]
            }
        }