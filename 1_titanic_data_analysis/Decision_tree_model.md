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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from statistics import mean
```

```python
titanic_dataset = pd.read_csv("../data/titanic/train.csv")
print("Dataset size - ", len(titanic_dataset))
```

```python
titanic_dataset.head()
```

```python
titanic_dataset.describe()
```

```python
def preprocess_data(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    X['Sex'] = X['Sex'].map({'male':0, 'female':1})
    X['Embarked'] = X['Embarked'].map({'C':0, 'Q':1, 'S':2})
    X['Embarked'] = X['Embarked'].fillna(2)
    X=X.dropna()
    
    Y = X['Survived'].copy().to_numpy()
    X = X.drop('Survived', axis=1)
    return X, Y
```

```python
np.random.shuffle(titanic_dataset.values)
train, test = train_test_split(titanic_dataset, test_size=0.2)
```

```python
X_train, Y_train = preprocess_data(train)
print("Train size = ", len(X_train))
X_test, Y_test = preprocess_data(test)
print("Test size = ", len(X_test))
```

```python
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
```

```python
Y_pred = clf.predict(X_test)
```

```python
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("Decision tree metrics on test data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
```

```python tags=[] jupyter={"outputs_hidden": true}
tree.plot_tree(clf)
```

```python
forest_clf = RandomForestClassifier(n_estimators=25, random_state=0)
forest_clf.fit(X_train, Y_train)
forest_y_pred = forest_clf.predict(X_test)
```

```python tags=[]
forest_accuracy = accuracy_score(Y_test, forest_y_pred)
forest_precision = precision_score(Y_test, forest_y_pred)
forest_recall = recall_score(Y_test, forest_y_pred)
forest_f1 = f1_score(Y_test, forest_y_pred)

print("Decision tree metrics on test data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (forest_accuracy, forest_precision, forest_recall, forest_f1))
```

```python
accuracies = []
f1s = []
values = range(5,80,5)
for i in values:
    iter_acc = []
    iter_f1 = []
    for j in range(10):
        forest_with_estimators = RandomForestClassifier(n_estimators=i, random_state=0)
        forest_with_estimators.fit(X_train, Y_train)
        y_hat = forest_with_estimators.predict(X_test)
        iter_acc.append(accuracy_score(Y_test, y_hat))
        iter_f1.append(f1_score(Y_test, y_hat))
    accuracies.append(mean(iter_acc))
    f1s.append( mean(iter_f1))                 
```

```python
plt.plot(values,accuracies, 'b-', label="accuracy")
plt.plot(values,f1s, 'g-', label="f1")
plt.legend(loc="upper left")
plt.xlabel("Number of estimators")
plt.show()
```

```python

```
