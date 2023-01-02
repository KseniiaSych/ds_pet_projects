import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin    

class ExctractTitle(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'ExctractTitle'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        dataset = X.copy()
        dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        return dataset
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
class ExctractDeck(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'ExctractDeck'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        dataset = X.copy()
        dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
        dataset['Deck'] = dataset['Deck'].replace(['A', 'B', 'C'], 'ABC')
        dataset['Deck'] = dataset['Deck'].replace(['D', 'E'], 'DE')
        dataset['Deck'] = dataset['Deck'].replace(['F', 'G'], 'FG')
        return dataset
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
class FamilySize(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'FamilySize(' + str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        dataset = X.copy()
        dataset[["SibSp","Parch"]] = dataset[["SibSp","Parch"]].fillna(0)
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        return dataset
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
class FeatureIsAlone(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'FeatureIsAlone'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        dataset = X.copy()
        dataset['IsAlone'] = 1
        dataset[['SibSp','Parch']] = dataset[['SibSp', 'Parch']].fillna(0)
        dataset['tmp_FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = np.where(dataset['tmp_FamilySize'] > 1, 0, dataset['IsAlone']) 
        return dataset.drop('tmp_FamilySize', axis=1)
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
