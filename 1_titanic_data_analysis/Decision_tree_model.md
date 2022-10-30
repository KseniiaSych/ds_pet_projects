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
```

```python
titanic_data = pd.read_csv("../data/titanic/train.csv")
print("Dataset size - ", len(titanic_data))
```

```python
def preprocess_data(df):
    X = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    X.replace(['male', 'female'],[0, 1], inplace=True)
    embarked = X['Embarked'].unique()
    map_embarked_int = {name: n for n, name in enumerate(embarked)}
    X["Embarked"] = X['Embarked'].replace(map_embarked_int)
    X=X.dropna()
    Y = X['Survived'].copy().to_numpy()
    X = X.drop('Survived', axis=1)
    return X, Y
```

```python
np.random.shuffle(titanic_data.values)
train, test = train_test_split(titanic_data, test_size=0.2)
```

```python
titanic_data, titanic_target = preprocess_data(train)
print("Train size = ", len(titanic_target))
test_data, test_target = preprocess_data(test)
print("Test size = ", len(test_target))
```

```python
clf = tree.DecisionTreeClassifier()
clf = clf.fit(titanic_data, titanic_target)
```

```python
y_pred = clf.predict(test_data)
```

```python
accuracy = accuracy_score(test_target, y_pred)
precision = precision_score(test_target, y_pred)
recall = recall_score(test_target, y_pred)
f1 = f1_score(test_target, y_pred)

print("Decision tree metrics on test data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
```

```python

```
