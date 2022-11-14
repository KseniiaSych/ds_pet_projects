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
age_bins = [0,5,10, 15, 20,60,100]
age_labels = ['Toddler', 'Kid', 'Preteen', 'Teen', 'Adult', 'Elderly']
fare_bins = [0, 20, 80, 140, 160, 180, 220,260,500]
fare_labels = [str(n) for n in range(len(fare_bins)-1)]
```

```python
def exctract_target(df, tagetName='Survived'):
    target = df[tagetName].copy().to_numpy()
    features = df.drop(tagetName, axis=1)
    return features, target

def preprocess_dataset(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    X['Embarked'] = X['Embarked'].fillna('Na')
    X['Age'] = pd.cut(X['Age'], bins=age_bins, labels=age_labels)
    X['Fare'] = pd.cut(X['Fare'], bins=fare_bins, labels=fare_labels)
    
    transformed = pd.get_dummies(X, columns=['Embarked', 'Sex','Age', 'Fare'])
    
    X = transformed.dropna()
    return X
```

```python
np.random.shuffle(titanic_dataset.values)
processed_titanic_dataset = preprocess_dataset(titanic_dataset)
processed_titanic_dataset.head()
```

```python
print("Processed dataset columns:", *processed_titanic_dataset.columns, sep='\n')
```

```python
train, val = train_test_split(processed_titanic_dataset, test_size=0.2)
```

```python
X_train, Y_train = exctract_target(train)
print("Train size = ", len(X_train))
X_val, Y_val = exctract_target(val)
print("Validation size = ", len(X_val))
```

```python
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, Y_train)
```

```python
Y_pred = clf.predict(X_val)
```

```python
accuracy = accuracy_score(Y_val, Y_pred)
precision = precision_score(Y_val, Y_pred)
recall = recall_score(Y_val, Y_pred)
f1 = f1_score(Y_val, Y_pred)

print("Decision tree metrics on validation data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
```

```python tags=[]
plt.figure(figsize=(10,10))
tree.plot_tree(clf)
plt.show()
```

```python
text_representation = tree.export_text(clf)
print(text_representation)
```

```python
list(zip(X_train.columns, range(len(X_train.columns))))
```

```python
forest_clf = RandomForestClassifier(n_estimators=20, random_state=0)
forest_clf.fit(X_train, Y_train)
forest_y_pred = forest_clf.predict(X_val)
```

```python tags=[]
forest_accuracy = accuracy_score(Y_val, forest_y_pred)
forest_precision = precision_score(Y_val, forest_y_pred)
forest_recall = recall_score(Y_val, forest_y_pred)
forest_f1 = f1_score(Y_val, forest_y_pred)

print("Decision tree metrics on validation data:")
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
        y_hat = forest_with_estimators.predict(X_val)
        iter_acc.append(accuracy_score(Y_val, y_hat))
        iter_f1.append(f1_score(Y_val, y_hat))
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
