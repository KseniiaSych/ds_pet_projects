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
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from CustomTransformers import (FillNaMode, DropColumns, ProcessTargetEncoding)
from TitanicSpecificTransformers import (ExctractTitle, FamilySize)

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
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)

preprocessing_steps = [
   ('add_feature_title', ExctractTitle()),
    ('add_feature_family_size', FamilySize()),
    ('process_na_category', FillNaMode(columns_with_na_category)),
    ('process_na_numeric', FillNaMode(columns_with_na_numeric)),
    ('process_categories',  ProcessTargetEncoding(target_encoders, category_columns)),
    ('drop_columns', DropColumns(columns_to_drop)),
    ('clf', clf)
         ]
preprocessing_pipeline = Pipeline(preprocessing_steps)
```

```python
preprocessing_pipeline.fit(X,Y)
```

```python

```
