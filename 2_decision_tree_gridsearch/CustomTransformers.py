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
        return 'IndexCategory'

    def fit(self, X, y = None):
        for col in self.columns:
            if col in X.columns:
                unique_values = X[col].unique()
                col_mapping =  dict(zip(unique_values, np.arange(len(unique_values))))
                self.columns_categories.update({col:col_mapping})
        return self
            
    def map_values(self, s, m):
        if s not in m.keys():
            #add missing value with next index
            m[s] = len(m.keys())
        return m[s]
                            
    def transform(self, X):   
        data = X.copy()
        for col in self.columns:
            if col in data.columns:
                mapping = self.columns_categories[col]
                data[col] = data[col].apply(lambda s: self.map_values(s,mapping))
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    
class ProcessCategoriesOHE(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
        self.columns_categories = {}
        
    def __repr__(self):
        return 'OHECategories'

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
        return 'ProcessBins'

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
    def __init__(self, encoders_map, columns = None):
        self.columns = columns
        self.encoders_map = encoders_map
        if not all(col in encoders_map for col in columns):
            raise Excpetion("Encoder for colum not found")
    
    def __repr__(self):
        return 'TargetEncoding'

    def fit(self, X, y = None):
        return self
    
    @staticmethod
    def fit_encoder(X, y, columns):
        encoders = {}
        for col in columns:
            new_encoder = TargetEncoder()
            new_encoder.fit(X[col],y)
            encoders.update({col:new_encoder})
        return encoders
    
    def transform(self, X):
        if not self.columns:
            return X
        data = X.copy()
        for col in self.columns:
            if col in X.columns:
                encoder = self.encoders_map[col]
                data[col] = encoder.transform(data[col])
        return data
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)