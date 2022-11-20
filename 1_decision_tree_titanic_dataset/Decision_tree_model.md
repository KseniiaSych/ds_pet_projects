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

def preprocess_dataset_numeric_class(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
   
    X['Sex'] = X['Sex'].map({'male':0, 'female':1})
    X['Embarked'] = X['Embarked'].fillna('Na')
    X['Embarked'] = X['Embarked'].map({'C':0, 'Q':1, 'S':2, 'Na':4})
    
    X = X.dropna()
    return X

def preprocess_dataset_one_hot_encoding(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    X['Embarked'] = X['Embarked'].fillna('Na')
    transformed = pd.get_dummies(X, columns=['Embarked', 'Sex'])
    
    X = transformed.dropna()
    return X

def preprocess_dataset_ohe_age_and_fare_bins(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    X['Embarked'] = X['Embarked'].fillna('Na')
    X['Age'] = pd.cut(X['Age'], bins=age_bins, labels=age_labels)
    X['Fare'] = pd.cut(X['Fare'], bins=fare_bins, labels=fare_labels)
    
    transformed = pd.get_dummies(X, columns=['Embarked', 'Sex','Age', 'Fare'])
    
    X = transformed.dropna()
    return X
```

```python
def fit_decision_tree(Xtrain, Ytrain, Xval, Yval, max_depth=None):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(Xtrain, Ytrain)
    
    Y_pred = clf.predict(Xval)
    accuracy = accuracy_score(Yval, Y_pred)
    precision = precision_score(Yval, Y_pred)
    recall = recall_score(Yval, Y_pred)
    f1 = f1_score(Yval, Y_pred)

    print("Decision tree metrics on validation data:")
    print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
    return clf
```

```python
def devide_and_extract(dataset):
    train, val = train_test_split(dataset, test_size=0.2)
    X_train, Y_train = exctract_target(train)
    print("Train size = ", len(X_train))
    X_val, Y_val = exctract_target(val)
    print("Validation size = ", len(X_val))
    return X_train, Y_train, X_val, Y_val 
```

```python
# Use Ohe for Embarked, Sex, Age and Fare
```

```python
np.random.shuffle(titanic_dataset.values)
processed_titanic_dataset = preprocess_dataset_ohe_age_and_fare_bins(titanic_dataset)
processed_titanic_dataset.head()
```

```python
print("Processed dataset columns:", *processed_titanic_dataset.columns, sep='\n')
```

```python
X_train, Y_train, X_val, Y_val=devide_and_extract(processed_titanic_dataset)
fit_decision_tree(X_train, Y_train, X_val, Y_val)
```

```python
# Decrease max_depth to analize tree
```

```python
X_train, Y_train, X_val, Y_val=devide_and_extract(processed_titanic_dataset)
tree_model = fit_decision_tree(X_train, Y_train, X_val, Y_val, 3)
```

```python tags=[]
plt.figure(figsize=(11,11))
tree.plot_tree(tree_model, feature_names=list(X_train.columns))
plt.show()
```

```python
text_representation = tree.export_text(tree_model, feature_names=list(X_train.columns))
print(text_representation)
```

# Compare to class as numeric value

```python
preprocessed_numeric = preprocess_dataset_numeric_class(titanic_dataset)
X_train, Y_train, X_val, Y_val=devide_and_extract(preprocessed_numeric)
_ = fit_decision_tree(X_train, Y_train, X_val, Y_val)
```

```python
preprocessed_no_bins = preprocess_dataset_one_hot_encoding(titanic_dataset)
X_train, Y_train, X_val, Y_val=devide_and_extract(preprocessed_no_bins)
_ = fit_decision_tree(X_train, Y_train, X_val, Y_val)
```

# RandomForestClassifier

```python
X_train, Y_train, X_val, Y_val=devide_and_extract(processed_titanic_dataset)


forest_clf = RandomForestClassifier(n_estimators=45, random_state=0)
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
