import pytest

from src.regression_model.config.core import config
from src.regression_model.processing.data_manager import load_dataset
from src.regression_model.processing.validation import validate_inputs

@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app.training_data_file)