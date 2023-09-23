from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from hyperopt import hp
from hyperopt.pyll import scope

model_selector = {
    'linear_regression': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'xgboost': xgb.XGBRegressor()
}

params_space = {
    'xgboost': {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 2), # exp(-5), exp(2)
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500])
        },

    'random_forest': {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        }
    }