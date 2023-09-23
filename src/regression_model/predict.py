import pandas as pd

from src.regression_model import __version__ as _version
from src.regression_model.config.core import config
from src.regression_model.processing.data_manager import load_pipeline
from src.regression_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(input_data: pd.DataFrame) -> dict:
    """Make a prediction using a saved model pipeline"""

    validated_data, errors = validate_inputs(input_data)

    results = {'predictions': None,
               'version': _version,
               'errors': errors}

    if not errors:
        predictions = list(_pipe.predict(X=validated_data[config.model.features]))
        results = {
            'predictions': predictions,
            'version': _version,
            'errors': errors
    }

    return results