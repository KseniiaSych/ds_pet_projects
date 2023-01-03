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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.tracking import MlflowClient

from CustomTransformers import (FillNaMode, FillNaWithConst, ProcessCategoriesAsIndex, 
                                ProcessCategoriesOHE, ProcessBins, Pass, DropColumns, 
                                ProcessTargetEncoding)

from TitanicSpecificTransformers import (ExctractTitle, ExctractDeck, FamilySize,
                                         FeatureIsAlone)
```

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

def remove_over_95th(df, columns=[]):
    copy = df.copy()
    for col in columns:
        lower = copy[col].quantile(0.02)
        higher = copy[col].quantile(0.98)
        copy = copy[(copy[col]>=lower) & (copy[col]<=higher)]
    return copy
```

```python
train, test = train_test_split(titanic_dataset, test_size=0.2, random_state=123)
train_balanced = balance_undersample(train)
train_copy = remove_over_95th(train_balanced, columns=['Age', 'Fare'])

X,Y = exctract_target(train_copy)
x_test, y_test = exctract_target(test)
```

```python
target_name = 'Survived'
columns_with_na_category = ['Embarked', 'Title']
columns_with_na_numeric = ['Age']
category_columns = ['Embarked', 'Sex', 'Deck', 'Title']
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

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
x_title = ExctractTitle().fit_transform(X)
x_for_target_encoding = ExctractDeck().fit_transform(x_title)
target_encoders = ProcessTargetEncoding.fit_encoder(x_for_target_encoding, Y, category_columns)
```

```python
preprocessing_steps = [
   ('add_feature_title', ExctractTitle()),
    ('add_feature_deck', ExctractDeck()),
    ('add_feature_family_size', FamilySize()),
    ('add_feature_is_alone', FeatureIsAlone()),
    ('process_na_category', FillNaMode(columns_with_na_category)),
    ('process_na_numeric', FillNaMode(columns_with_na_numeric)),
    ('process_categories',  ProcessTargetEncoding(target_encoders, category_columns)),
    ('process_bins', ProcessBins(columns_for_bins)),
    ('drop_columns', DropColumns(columns_to_drop)),
         ]
preprocessing_pipeline = Pipeline(preprocessing_steps)
```

```python
preprocessing_pipeline
```

```python
res = preprocessing_pipeline.fit_transform(X, Y)
```

```python
res.head()
```

```python
steps = [
    *preprocessing_steps,
    ('clf', RandomForestClassifier(n_estimators=5, random_state=0))
]
pipeline = Pipeline(steps)
```

```python
pipeline
```

```python
#num_estimators=[50, 100]
```

```python
param_grid = dict(
    add_feature_title = [ExctractTitle(), Pass()],
    add_feature_deck = [ExctractDeck(), Pass()],
    add_feature_family_size = [FamilySize(), Pass()],
    add_feature_is_alone = [FeatureIsAlone(), Pass()],
    process_na_category = [FillNaMode(columns_with_na_category), 
                           FillNaWithConst('Na', columns_with_na_category)],
    process_categories = [ProcessCategoriesAsIndex(category_columns), 
                          ProcessCategoriesOHE(category_columns), 
                          ProcessTargetEncoding(target_encoders, category_columns),
                         ],
    process_bins = [ProcessBins(columns_for_bins), Pass()],
    
    clf__n_estimators = [100],
    clf__criterion = ['gini'],
    clf__max_depth = [4],
    clf__random_state = [42],
    clf__max_leaf_nodes = [None],
    clf__min_samples_leaf = [1],
    clf__min_samples_split = [2],
)
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1)

```

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

EXPERIMENT_NAME = 'mlflow-randomtree_pipeline'
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
mlflow.sklearn.autolog(log_models=False, silent=True, max_tuning_runs=200)
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
y_pred = final_pipeline.predict(x_test)
```

```python
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
print("Precision Score: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall Score: ", recall_score(y_test, y_pred, average='weighted'))
```

```python

```
