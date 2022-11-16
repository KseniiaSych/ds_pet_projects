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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import graphviz 

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
```

```python
titanic_dataset = pd.read_csv("../data/titanic/train.csv")
print("Dataset size - ", len(titanic_dataset))
```

```python
titanic_dataset.head()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# Preprocessing functions
<!-- #endregion -->

```python
def exctract_target(df, tagetName='Survived'):
    target = df[tagetName].copy().to_numpy()
    features = df.drop(tagetName, axis=1)
    return features, target

def preprocess_dataset(df):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    X['Embarked'] = X['Embarked'].fillna('Na')
    transformed = pd.get_dummies(X, columns=['Embarked', 'Sex'])
    
    X = transformed.dropna()
    return X
```

```python
titanic_dataset.Survived.value_counts()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# Undersample to balance target
<!-- #endregion -->

```python
survived_part = titanic_dataset[titanic_dataset.Survived==1]
not_survived_sample = titanic_dataset[titanic_dataset.Survived==0].sample(n=len(survived_part))
sampled_dataset = pd.concat([survived_part, not_survived_sample], ignore_index=True)

```

```python
sampled_dataset.Survived.value_counts()
```

```python
sampled_dataset_precessed = preprocess_dataset(sampled_dataset)
train, val = train_test_split(sampled_dataset_precessed, test_size=0.2)
X_train, Y_train = exctract_target(train)
print("Train size = ", len(X_train))
X_val, Y_val = exctract_target(val)
print("Validation size = ", len(X_val))
```

```python
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, Y_train)
    
Y_pred = clf.predict(X_val)
accuracy = accuracy_score(Y_val, Y_pred)
precision = precision_score(Y_val, Y_pred)
recall = recall_score(Y_val, Y_pred)
f1 = f1_score(Y_val, Y_pred)

print("Decision tree metrics on validation data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
```

```python
text_representation = tree.export_text(clf)
print(text_representation)
```

```python
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=X_train.columns,  
                      class_names=['0', '1'],  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
```

# Search over parameters

```python
dataset_precessed = preprocess_dataset(sampled_dataset)
X, Y = exctract_target(dataset_precessed)
```

## Exhaustive gridsearch

```python
tree_classifier = tree.DecisionTreeClassifier()

param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2,5,10,30],
              'max_depth': [5,10,20, 30, None],
              'max_features':['sqrt', 'log2']
             }
search = GridSearchCV(tree_classifier, param_grid, verbose=True)
search.fit(X, Y)
```

```python
print(search.best_params_)
print(search.score(X, Y))
```

## Randomized search

```python
distributions = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': randint(2, 50),
              'max_depth': randint(1, 100),
                'max_features':['sqrt', 'log2']}
random_search = RandomizedSearchCV(tree_classifier, distributions, random_state=0, cv=15, verbose=True)
random_search.fit(X,Y)
```

```python
print(random_search.best_params_)
print(random_search.score(X, Y))
```

# Gridsearch for random forest

<!-- #region tags=[] -->
## Exhaustive gridsearch
<!-- #endregion -->

```python
param_grid = {'n_estimators': range(5,100,10)}
forest_estimator = RandomForestClassifier(criterion='gini',
                                        max_depth=10, 
                                        min_samples_split=2,
                                        max_features='sqrt',
                                        random_state=0)
search_forest = GridSearchCV(forest_estimator, param_grid, verbose=True)
search_forest.fit(X, Y)
```

```python
print(search_forest.best_params_)
print(search_forest.score(X, Y))
```

## Random gridsearch

```python
param_grid = {'n_estimators': randint(2, 500)}
random_search = RandomizedSearchCV(forest_estimator, param_grid, random_state=0, verbose=True)
random_search.fit(X,Y)
```

```python
print(random_search.best_params_)
print(random_search.score(X, Y))
```

<!-- #region tags=[] -->
# Use mlflow with gridsearch
<!-- #endregion -->

```python
EXPERIMENT_NAME = "mlflow-decisiontree_depth"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
```

```python
parameters_depth = list(range(2,70,10))
parameters_depth.append(None)

for idx, depth in enumerate(parameters_depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(Y_val, y_pred)

    # Start MLflow
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("depth", depth)

        # Track metrics
        mlflow.log_metric("accuracy", accuracy)

        # Track model
        mlflow.sklearn.log_model(clf, "classifier")
```

# MLflow sklearn

```python
def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

# enable autologging
mlflow.sklearn.autolog()

# train a model
model =tree.DecisionTreeClassifier()
with mlflow.start_run() as run:
    model.fit(X, Y)

# fetch logged data
params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

pprint(params)
pprint(metrics)
pprint(tags)
pprint(artifacts)
```

```python

```
