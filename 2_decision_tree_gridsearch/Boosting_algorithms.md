---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import mlflow
from mlflow.tracking import MlflowClient

from xgboost import XGBClassifier
import warnings

from CustomTransformers import (FillNaMode, DropColumns, ProcessTargetEncoding)
from TitanicSpecificTransformers import (ExctractTitle, FamilySize)
```

<!-- #region tags=[] -->
# Data preprocessing and utils
<!-- #endregion -->

```python
titanic_dataset = pd.read_csv("../data/titanic/train.csv")
print("Dataset size - ", len(titanic_dataset))
```

```python
titanic_dataset.head()
```

```python
def exctract_target(df, tagetName='Survived'):
    target = df[tagetName].copy().to_numpy()
    features = df.drop(tagetName, axis=1)
    return features, target

def balance_undersample(df, tagetName='Survived'):
    survived_part = df[df[tagetName]==1]
    not_survived_sample = df[df[tagetName]==0].sample(n=len(survived_part))
    data =  pd.concat([survived_part, not_survived_sample], ignore_index=True)
    return data

def remove_outliers(df, columns=[]):
    df_copy = df.copy()
    for col in columns:
        lower = df_copy[col].quantile(0.02)
        higher = df_copy[col].quantile(0.98)
        df_copy = df_copy[(df_copy[col]>=lower) & (df_copy[col]<=higher)]
    return df_copy
```

```python
def print_scores(observed, predicted):
    print("Accuracy Score: ", accuracy_score(observed, predicted))
    print("F1 Score: ", f1_score(observed, y_pred, average='weighted'))
    print("Precision Score: ", precision_score(observed, predicted, average='weighted'))
    print("Recall Score: ", recall_score(observed, predicted, average='weighted'))
```

```python
train, test = train_test_split(titanic_dataset, test_size=0.2, random_state=123)
train_balanced = balance_undersample(train)
train_copy = remove_outliers(train_balanced, columns=['Age', 'Fare'])

X,Y = exctract_target(train_copy)
x_test, y_test = exctract_target(test)
```

```python
columns_with_na_category = ['Embarked', 'Title']
columns_with_na_numeric = ['Age']
category_columns = ['Embarked', 'Sex', 'Title']
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

x_title = ExctractTitle().fit_transform(X)

warnings.simplefilter(action='ignore', category=FutureWarning)
target_encoders = ProcessTargetEncoding.fit_encoder(x_title, Y, category_columns)
```

```python
preprocessing_steps = [
   ('add_feature_title', ExctractTitle()),
    ('add_feature_family_size', FamilySize()),
    ('process_na_category', FillNaMode(columns_with_na_category)),
    ('process_na_numeric', FillNaMode(columns_with_na_numeric)),
    ('process_categories',  ProcessTargetEncoding(target_encoders, category_columns)),
    ('drop_columns', DropColumns(columns_to_drop)),
         ]
preprocessing_pipeline = Pipeline(preprocessing_steps)
```

```python
X_preprocessed = preprocessing_pipeline.fit_transform(X)
x_test_processed = preprocessing_pipeline.fit_transform(x_test)
```

# AdaBoost

```python
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_preprocessed,Y)
```

```python
y_pred = ada_clf.predict(x_test_processed)
print_scores(y_test, y_pred)
```

```python
ada_clf.score(x_test_processed, y_test)
```

```python
param_grid = dict(
    n_estimators = [50, 80, 100, 120, 200, 250, 300, 350],
    learning_rate = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    algorithm = ['SAMME', 'SAMME.R']
)

ada_grid_search = RandomizedSearchCV(ada_clf, param_grid, n_jobs=-1)
```

```python
EXPERIMENT_NAME = 'mlflow-ada_boost'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=500)
with mlflow.start_run() as run:
    ada_grid_search.fit(X_preprocessed, Y)
```

```python
ada_best = ada_grid_search.best_estimator_
```

```python
ada_best
```

```python
y_pred_a = ada_best.predict(x_test_processed)
print_scores(y_test, y_pred_a)
```

<!-- #region tags=[] -->
# Gradient boost
<!-- #endregion -->

```python
grad_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=42)
grad_clf.fit(X_preprocessed,Y)
```

```python
grad_clf.score(x_test_processed, y_test)
```

```python
y_pred = grad_clf.predict(x_test_processed)
print_scores(y_test, y_pred)
```

```python
grad_param_grid = dict(
    loss = ['log_loss', 'deviance','exponential'],
    learning_rate = [0.5, 0.25, 0.1, 0.05, 0.01],
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100],
    subsample = [0.2, 0.5, 0.8, 1.0],
    criterion = ['friedman_mse', 'squared_error'],
    min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True),
    min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True),
    min_weight_fraction_leaf = [0.1, 0.15, 0.2],
    max_depth =list(range(1,22,2)),
    max_features  = list(range(1, x_test_processed.shape[1]))
)

grad_grid_search = RandomizedSearchCV(grad_clf, grad_param_grid, n_jobs=-1)
```

```python
EXPERIMENT_NAME = 'mlflow-grad_boost'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=False, max_tuning_runs=500)
with mlflow.start_run() as run:
    grad_grid_search.fit(X_preprocessed, Y)
```

```python
grad_best = grad_grid_search.best_estimator_
```

```python
grad_best
```

```python
y_pred_grad = grad_best.predict(x_test_processed)
print_scores(y_test, y_pred_grad)
```

## BayesSearch

```python
reg_bay_param_grid = {
    'loss': Categorical(['log_loss', 'deviance','exponential']),
    'learning_rate' : Real(0.01, 0.5, prior='log-uniform'),
    'n_estimators' : Integer(1,100),
    'subsample' : Real(0.1, 1.0, prior='log-uniform'),
    'criterion' : Categorical(['friedman_mse', 'squared_error']),
    'min_samples_split' :  Real(0.1, 1.0, prior='log-uniform') ,
    'min_samples_leaf' :  Real(0.1, 0.5, prior='log-uniform'), 
    'min_weight_fraction_leaf' : [0.1, 0.15, 0.2],
    'max_depth' : Integer(1,25),
    'max_features' : Integer(1, x_test_processed.shape[1])
}
```

```python
reg_bay =  BayesSearchCV(
    estimator = grad_clf,
    search_spaces = reg_bay_param_grid,
    n_jobs=-1,
    random_state=42
)
```

```python
EXPERIMENT_NAME = 'mlflow-grad_boost_bayes_search'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=200)
with mlflow.start_run() as run:
    reg_bay.fit(X_preprocessed, Y)
```

```python
bay_best = reg_bay.best_estimator_
```

```python
bay_best
```

```python
y_pred_bay = bay_best.predict(x_test_processed)
print_scores(y_test, y_pred_bay)
```

# XGBoost

```python
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
```

```python
xgb_clf.fit(X_preprocessed, Y)
```

```python
y_pred_xgb = xgb_clf.predict(x_test_processed)
print_scores(y_test, y_pred_xgb)
```

```python
xgb_param_grid = dict(
    max_depth =list(range(1,22,2)),
)
xgb_grid_search = RandomizedSearchCV(xgb_clf, xgb_param_grid, n_jobs=-1)
```

```python
EXPERIMENT_NAME = 'mlflow-xgb_boost'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=500)
with mlflow.start_run() as run:
    xgb_grid_search.fit(X_preprocessed, Y)
```

```python
xgb_best = xgb_grid_search.best_estimator_
```

```python
xgb_best
```

```python
y_pred_xgb= xgb_best.predict(x_test_processed)
print_scores(y_test, y_pred_xgb)
```
