import numpy as np
import pandas as pd

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin, partial

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.regression_model.config.core import config
from src.regression_model.pipeline import pipeline
from src.regression_model.processing.data_manager import load_dataset
from src.regression_model.processing.validation import validate_inputs

from src.regression_model.model_dispatcher import model_selector, params_space


def hyperparam_search(X: pd.DataFrame, y: pd.DataFrame, model_name: str, param_space: dict):
    """Conduct experimental tracking to find best parameters using MLflow"""

    def objective(params):
        with mlflow.start_run():
        
            mlflow.log_params(params)

            k_fold = KFold(n_splits=5, shuffle=True)
            val_fold_score = []
            train_fold_score = []

            for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                # Configure pipeline to add model to the last step
                model_pipeline_step = (model_name, 
                                       model_selector[model_name].set_params(**params))
                model_pipeline = Pipeline(pipeline.steps + [model_pipeline_step])

                # Fit pipeline to training data
                model_pipeline.fit(X_train, y_train)

                # Predict
                y_val_pred = model_pipeline.predict(X_val)
                y_train_pred = model_pipeline.predict(X_train)

                # Evaluate
                score_val = mean_squared_error(y_val_pred, y_val, squared=False)
                score_train = mean_squared_error(y_train_pred, y_train, squared=False)

                val_fold_score.append(score_val)
                train_fold_score.append(score_train)

            mlflow.log_metric("avg_training_rmse", np.mean(train_fold_score))
            mlflow.log_metric("avg_val_rmse", np.mean(val_fold_score))

        return {'loss': np.mean(val_fold_score), 'status': STATUS_OK}
    
        
    rstate = np.random.default_rng(42)

    best_result = fmin(
        fn = objective, # function to optimize
        space = param_space,
        algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
        max_evals = 50, # maximum number of iterations
        trials = Trials(), # logging
        rstate = rstate # fixing random state for the reproducibility
    )

def get_best_params(experiment_name: str):
    """Locate the run with lowest average validation MSE and get the params"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["metrics.avg_val_rmse ASC"])[0]
    
    best_params = best_run.data.params

    return best_params


if __name__ == "__main__":

    data = load_dataset(file_name=config.app.training_data_file)
    validated_data, errors = validate_inputs(data)

    X_train, X_test, y_train, y_test = train_test_split(
    validated_data[config.model.features],
    validated_data[config.model.target],
    test_size=config.model.test_size,
    random_state=config.model.random_state # train test split involves randomness
    )
    
    EXPERIMENT_NAME = f'{config.model.selected_model_name}-hyperopt-cv'
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Search for best params
    print("Hyperparam Searching...")
    hyperparam_search(X_train,
                      y_train,
                      model_name=config.model.selected_model_name,
                      param_space=params_space[config.model.selected_model_name])
    
    print("Getting best params")
    best_params = get_best_params(experiment_name=EXPERIMENT_NAME)

    print(best_params)
                 