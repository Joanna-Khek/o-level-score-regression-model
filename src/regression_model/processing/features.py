import numpy as np
import pandas as pd
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Get the number of sleeping hours by computing the 
    difference between sleep time and wake time
    """

    def __init__(self, first_value: str, second_value: str):
        # first_value - second_value
        self.first_value = first_value
        self.second_value = second_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.assign(sleep_datetime = pd.to_datetime(X[self.second_value], format= "%H:%M"),
                     wake_datetime = pd.to_datetime(X[self.first_value], format= "%H:%M"),
                     sleeping_hours=lambda df_: 
                     (df_.wake_datetime - df_.sleep_datetime).dt.components.hours)
        return X 
    

class Mapper(BaseEstimator, TransformerMixin):
    """Map feature to value in dictionary"""

    def __init__(self, feature: List[str], mappings: dict):

        self.feature = feature
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()

        for feature in self.feature:
            X[feature] = X[feature].map(self.mappings)

        return X
   
class CCATransformer(BaseEstimator, TransformerMixin):
    """Convert CCA string to uppercase in the first character only"""

    def __init__(self, feature: List[str]):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        X = X.copy()
        X = X.assign(CCA=lambda df_: df_.CCA.str.title())

        return X
    
class AgeTransformer(BaseEstimator, TransformerMixin):
    """Clean negative age and single digit age"""

    def __init__(self, feature: List[str]):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def add_one_if_less_than_two_digits(self, feature: pd.Series):
        feature = int(feature)
        if feature < 0: # eg: -5 -> 15
            feature = abs(feature)
            return '1' + str(feature)
        elif feature < 10: # eg: 5 -> 15
            return '1' + str(feature)
        else:
            return str(feature)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.feature:
            X = (X
                 #.query(f"{feature} >= 0")
                 .assign(age=lambda df_: df_.age.apply(self.add_one_if_less_than_two_digits).astype(float))
            )
        
        return X