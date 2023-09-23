from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pipeline import pipeline

from src.regression_model.config.core import config
from src.regression_model.processing.data_manager import load_dataset, save_pipeline
from src.regression_model.processing.validation import validate_inputs
from model_dispatcher import model_selector


def run_training(model_name: str) -> None:
    """Train the model using best params from hyperparam tuning

    Args:
        model_name (str): The chosen model to train data and perform prediction
    """

    # Read Training Data
    data = load_dataset(file_name=config.app.training_data_file)

    # Validate Data
    validated_data, errors = validate_inputs(data)

    # Split into Train and Test set
    X_train, X_test, y_train, y_test = train_test_split(
    validated_data[config.model.features],
    validated_data[config.model.target],
    test_size=config.model.test_size,
    random_state=config.model.random_state) # train test split involves randomness

    # Get best param for chosen model
    best_params = config.model.best_params[model_name].dict()

    # Configure pipeline to add model (with best params from hyperparam tuning) to the last step
    model_pipeline_step = (model_name, model_selector[model_name].set_params(**best_params))
    model_pipeline = Pipeline(pipeline.steps + [model_pipeline_step])

    # Fit training data to model pipeline
    model_pipeline.fit(X_train, y_train)

    # Persist trained model
    save_pipeline(pipeline_to_persist=model_pipeline)

if __name__ == "__main__":
    run_training(model_name=config.model.selected_model_name)