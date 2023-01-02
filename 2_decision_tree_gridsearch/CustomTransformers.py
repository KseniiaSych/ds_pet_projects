import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'SelectColumns(' + str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.copy() if self.columns is None else X[self.columns].copy()
    
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
            if col in X.columns:
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
            if col in X.columns:
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
            if col in X.columns:
                self.columns_categories.update({col:X[col].unique()})
        return self
            
    def transform(self, X):   
        data = X.copy()
        for col in self.columns:
            if col in data.columns:
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
            if col in X.columns:
                self.columns_categories.update({col:X[col].unique()})
        return self
            
    def transform(self, X): 
        data = X.copy()
        present_columns = []
        for col in self.columns:
            if col in data.columns:
                present_columns.append(col)
                data[col] = data[col].astype(pd.CategoricalDtype(categories=self.columns_categories[col]))
        data = pd.get_dummies(data, columns=present_columns, dummy_na=False)
        
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

    
class Pass(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Pass'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self,X,y = None):
        return self.fit(X,y).transform(X)
    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def __repr__(self):
        return 'DropColumns(' + str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.copy().drop(columns=self.columns)
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    
class ProcessTargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, columns = None,):
        self.columns = columns
        self.target_column = target_column
        self.encoder = TargetEncoder()
    
    def __repr__(self):
        return 'TargetEncoding(' + str(self.columns) +')'

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for col in self.columns:
            if col in X.columns:
                col_name = col+'_encoded'
                data[col_name] = self.encoder.fit_transform(data[col], data[self.target_column])
                data = data.drop(columns=col)
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)