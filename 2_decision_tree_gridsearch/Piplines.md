---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.tracking import MlflowClient

from CustomTransformers import (SelectColumns, FillNaCommon, FillNaWithConst, ProcessCategoriesAsIndex, 
                                ProcessCategoriesOHE, ProcessBins)

```

```python
titanic_dataset = pd.read_csv("../data/titanic/train.csv")
print("Dataset size - ", len(titanic_dataset))
```

<!-- #region tags=[] -->
# Pipelines
<!-- #endregion -->

## version 1

```python
mlflow.sklearn.autolog(disable=True)
```

```python
def exctract_target(df, tagetName='Survived'):
    target = df[tagetName].copy().to_numpy()
    features = df.drop(tagetName, axis=1)
    return features, target
```

```python
train, test = train_test_split(titanic_dataset, test_size=0.2, random_state=123)
train_copy = train.copy()
```

```python
X,Y = exctract_target(train_copy)
X_test, y_test = exctract_target(test)
```

```python
columns_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
columns_with_na_category = ['Embarked']
columns_with_na_numeric = ['Age']
category_columns = ['Embarked', 'Sex']

fare_bins = [0, 20, 80, 140, 160, 180, 220,260,500]
columns_for_bins = {'Age':{
                            'bins': [0,5,10, 15, 20,60,100], 
                            'labels': ['Toddler', 'Kid', 'Preteen', 'Teen', 'Adult', 'Elderly']
                            }, 
                    'Fare':{
                        'bins': fare_bins,
                        'labels':[str(n) for n in range(len(fare_bins)-1)]
                    }}
```

```python
preprocessing_steps = [
    ('drop_columns', SelectColumns(columns_to_use)),
    ('process_na_category', FillNaCommon(columns_with_na_category)),
    ('process_na_numeric', FillNaCommon(columns_with_na_numeric)),
    ('process_categories', ProcessCategoriesOHE(category_columns)),
    ('process_bins', ProcessBins(columns_for_bins))
         ]
preprocessing_pipeline = Pipeline(preprocessing_steps)
```

```python
preprocessing_pipeline.fit(X, Y)
```

```python
preprocessing_pipeline.transform(X)
```

```python
preprocessing_pipeline.transform(X_test)
```

```python
steps = [
    *preprocessing_steps,
    ('clf', tree.DecisionTreeClassifier())
]
pipeline = Pipeline(steps)
```

```python
pipeline
```

```python
pipeline.fit(X, Y)
```

```python
pipeline.predict(X_test)
```

```python
max_depth = list(range(4, 50, 8))
max_depth.append(None)

max_features = list(range(2, X.shape[1]+1, 2))
```

```python
param_grid = dict(
    process_na_category = [FillNaCommon(columns_with_na_category), 
                           FillNaWithConst('Na', columns_with_na_category)],
    process_categories = [ProcessCategoriesAsIndex(category_columns), ProcessCategoriesOHE(category_columns)],
    process_bins = [ProcessBins(columns_for_bins), ProcessBins()],
    clf__criterion = ['gini', 'entropy', 'log_loss'],
    clf__max_depth = max_depth,
    clf__max_features = max_features,
    clf__random_state = [42],
    clf__splitter = ['best', 'random']
)
grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_jobs=-1, n_iter=300)

```

```python
EXPERIMENT_NAME = 'mlflow-decisiontree_custom_pipeline'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=500)
with mlflow.start_run() as run:
    grid_search.fit(X, Y)
```

```python
final_pipeline = grid_search.best_estimator_
```

```python
final_pipeline
```

```python
y_pred = final_pipeline.predict(X_test)
```

```python
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
print("Precision Score: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall Score: ", recall_score(y_test, y_pred, average='weighted'))
```

## version 2

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
```

```python
mlflow.sklearn.autolog(disable=True)
```

```python
train, val = train_test_split(titanic_dataset, test_size=0.2)
X_train, Y_train = exctract_target(train)
X_val, y_val = exctract_target(val)
```

```python
category_columns = ['Embarked', 'Sex', 'Pclass']
cat_transformer = Pipeline(steps=[
    ('transformer', FunctionTransformer(lambda x: x.astype("string"), validate=False)),
    ('transformer_na', FunctionTransformer(lambda x: x.replace({pd.NA: None}), validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value = 'Na')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

columns_to_pass = ['Age', 'SibSp', 'Parch', 'Fare']
pass_transformer = Pipeline(steps=[
    ('transformer_na_pass', FunctionTransformer(lambda x: x.replace({pd.NA: np.nan}), validate=False)),
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('category', cat_transformer, category_columns),
        ('rest', pass_transformer, columns_to_pass)
    ])

clf = tree.DecisionTreeClassifier()
```

```python
clf_pipeline = Pipeline(steps=[   
    ('preprocess',  preprocessor),
    ('model', clf)
])
```

```python
clf_pipeline.fit(X_train, Y_train)
```

```python
y_pred = clf_pipeline.predict(X_val)
```

```python
print("Accuracy Score: ", accuracy_score(y_val, y_pred))
print("F1 Score: ", f1_score(y_val, y_pred, average='weighted'))
print("Precision Score: ", precision_score( y_val, y_pred, average='weighted'))
print("Recall Score: ", recall_score(y_val, y_pred, average='weighted'))
```
