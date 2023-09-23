from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engine.selection import DropFeatures
from feature_engine.creation import RelativeFeatures
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper

from src.regression_model.processing import features as pp
from src.regression_model.config.core import config


pipeline = Pipeline([
    ('total_class_size', RelativeFeatures(variables=['n_male'],
                                          reference =['n_female'],
                                          func=['add'])),
    ('sleeping_hours', pp.DatetimeTransformer(first_value=config.model.sleep_wake_datetime_vars[0],
                                              second_value=config.model.sleep_wake_datetime_vars[1])),
    ('clean_cca', pp.CCATransformer(feature=['CCA'])),
    ('clean_age', pp.AgeTransformer(feature=['age'])),
    ('mapper_cca', pp.Mapper(feature=['CCA'],
                             mappings=config.model.cca_mappings)),
    ('mapper_tuition', pp.Mapper(feature=['tuition'],
                                 mappings=config.model.tuition_mappings)),
    ('one_hot_encode', OneHotEncoder(variables=config.model.encode_vars,
                                     drop_last_binary=True)),
    ('scale', SklearnTransformerWrapper(StandardScaler())),
    ('drop_features', DropFeatures(features_to_drop=config.model.features_to_drop))
])
