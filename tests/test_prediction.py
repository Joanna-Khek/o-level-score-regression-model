import numpy as np
from typing import List
from src.regression_model.predict import make_prediction

def test_make_prediction(sample_input_data):

    result = make_prediction(input_data=sample_input_data)
    predictions = result.get("predictions")
    assert isinstance(predictions, List[float])
    assert isinstance(predictions[0], np.float32)