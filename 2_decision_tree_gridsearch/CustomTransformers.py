import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import unittest

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'SelectColumns(' + str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.copy() if self.columns is None else X[self.columns]
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    

class FillNaMode(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'FillNaMode(' +  str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for col in self.columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    

class FillNaWithConst(BaseEstimator, TransformerMixin):
    def __init__(self, const, columns = None):
        self.const = const
        self.columns = columns
    
    def __repr__(self):
        return 'FillNaWithConst(' +  str(self.columns) + ',' + self.const+ ')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for col in self.columns:
            data[col] = data[col].fillna(self.const)
        return data
        
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
class ProcessCategoriesAsIndex(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
        self.columns_categories = {}
        
    def __repr__(self):
        return 'ProcessCategoriesAsIndex(' + str(self.columns) + ')'

    def fit(self, X, y = None):
        for col in self.columns:
            self.columns_categories.update({col:X[col].unique()})
        return self
            
    def transform(self, X):   
        data = X.copy()
        for col in self.columns:
            values = self.columns_categories[col]
            map_values = dict(zip(values, np.arange(len(values))))
            data[col] = data[col].map(map_values)
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    
class ProcessCategoriesOHE(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
        self.columns_categories = {}
        
    def __repr__(self):
        return 'ProcessCategoriesOHE(' + str(self.columns) + ')'

    def fit(self, X, y = None):
        for col in self.columns:
            self.columns_categories.update({col:X[col].unique()})
        return self
            
    def transform(self, X): 
        data = X.copy()
        for col in self.columns:
            data[col] = data[col].astype(pd.CategoricalDtype(categories=self.columns_categories[col]))
        data = pd.get_dummies(data, columns=self.columns, dummy_na=False)
        
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    

class ProcessBins(BaseEstimator, TransformerMixin):
    def __init__(self, columns_bins = None):
        self.columns_bins = columns_bins
        
    def __repr__(self):
        return 'ProcessBins(' + str(self.columns_bins.keys()) + ')'

    def fit(self, X, y = None):
        return self
            
    def transform(self, X): 
        data = X.copy()
        if self.columns_bins:
            for column, bins in self.columns_bins.items():
                 data[column] = pd.cut(data[column], bins=bins['bins'], labels=bins['labels'])
            data = pd.get_dummies(data, columns=self.columns_bins.keys())
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)