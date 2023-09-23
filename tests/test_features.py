import pandas as pd

from src.regression_model.config.core import config
from src.regression_model.processing.features import (DatetimeTransformer,
                                                      CCATransformer,
                                                      AgeTransformer,
                                                      Mapper)

def test_datetime_transformer(sample_input_data):
    transformer = DatetimeTransformer(first_value=config.model.sleep_wake_datetime_vars[0],
                                      second_value=config.model.sleep_wake_datetime_vars[1])
    
    subject = transformer.fit_transform(sample_input_data)

    assert subject["sleeping_hours"].iat[0] == 8

def test_age_transformer():
    transformer = AgeTransformer(feature=['age'])
    data = pd.DataFrame({'age': [-5, 5, 15]})
                        
    subject = transformer.fit_transform(data)

    assert (subject == 15.0).all(axis=1).all() == True

def test_cca_transformer(sample_input_data):
    transformer = CCATransformer(feature=['CCA'])
    allowed_categories = ['Sports', 'None', 'Clubs', 'Arts']

    subject = transformer.fit_transform(sample_input_data)
    assert subject.CCA.isin(allowed_categories).all() == True

def test_cca_mapper_transformer():
    data = pd.DataFrame({'CCA': ['Sports', 'Arts', 'Clubs', 'None']})
    transformer = Mapper(feature=['CCA'], mappings=config.model.cca_mappings)
    subject = transformer.fit_transform(data)

    allowed_categories = ['Yes', 'No']
    assert subject.CCA.isin(allowed_categories).all() == True

def test_tution_mapper_transformer():
    data = pd.DataFrame({'tuition': ['Y', 'Yes', 'N', 'No']})
    transformer = Mapper(feature=['tuition'], mappings=config.model.tuition_mappings)
    subject = transformer.fit_transform(data)

    allowed_categories = ['Yes', 'No']
    assert subject.tuition.isin(allowed_categories).all() == True
