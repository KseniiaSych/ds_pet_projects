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
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import graphviz 

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns

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
dataset_processed = preprocess_dataset(titanic_dataset)

train, val = train_test_split(dataset_processed, test_size=0.2)

survived_part = train[train.Survived==1]
not_survived_sample = train[train.Survived==0].sample(n=len(survived_part))
train = pd.concat([survived_part, not_survived_sample], ignore_index=True)
```

```python
train.Survived.value_counts()
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
    
Y_pred = clf.predict(X_val)
accuracy = accuracy_score(Y_val, Y_pred)
precision = precision_score(Y_val, Y_pred)
recall = recall_score(Y_val, Y_pred)
f1 = f1_score(Y_val, Y_pred)

print("Decision tree metrics on validation data:")
print("Accuracy - %f \nPrecision - %f \nRecall - %f \nF1 - %f" % (accuracy, precision, recall, f1))
```

```python
text_representation = tree.export_text(clf, feature_names=list(X_train.columns))
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

```python
clf.get_params()
```

<!-- #region tags=[] -->
# Search over parameters
<!-- #endregion -->

```python
X, Y = exctract_target(dataset_processed)
```

## Exhaustive gridsearch

```python
tree_classifier = tree.DecisionTreeClassifier()

max_depth = list(range(4, 50, 8))
max_depth.append(None)

min_samples_split = list(range(2, 30, 7))

max_features = list(range(2, X.shape[1]+1, 2))
max_features.extend(['sqrt', 'log2'])

param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': min_samples_split,
              'max_depth': max_depth,
              'max_features':max_features,
              'min_samples_leaf': list(range(1,20, 4))
             }
search = GridSearchCV(tree_classifier, param_grid, verbose=True)
search.fit(X, Y)
```

```python
search.param_grid
```

```python
print("Best parameters - ", search.best_params_)
```

```python
df_results = pd.DataFrame(search.cv_results_)
df_results = df_results.sort_values(by=["rank_test_score"])
```

```python
df_results.head()
```

```python
plt.plot(df_results.mean_test_score, '.')
plt.ylim((0,1))
plt.axhline(Y_train.mean(), color='r')
plt.show()
```

```python
df_results['param_max_depth'].fillna(value='No max depth', inplace = True)
df_results.groupby('param_max_depth').mean_test_score.mean().plot.bar()
plt.ylim((0.75, 0.8))
plt.show()
```

```python
df_results.groupby('param_criterion').mean_test_score.mean().plot.bar()
plt.ylim((0.756, 0.8))
plt.show()
```

```python
df_results.groupby('param_splitter').mean_test_score.mean().plot.bar()
```

```python
df_results.groupby('param_max_features').mean_test_score.mean().plot.bar()
plt.ylim((0.7, 0.8))
```

```python
df_results.groupby('param_min_samples_split').mean_test_score.mean().plot.bar()
plt.ylim((0.756, 0.8))
```

```python
df_results.groupby('param_min_samples_leaf').mean_test_score.mean().plot.bar()
plt.ylim((0.7, 0.8))
```

```python
df_split_leaf = df_results.groupby(['param_min_samples_split','param_min_samples_leaf']).mean_test_score.mean().reset_index()

```

```python
df_split_leaf = df_split_leaf.pivot("param_min_samples_split", "param_min_samples_leaf", "mean_test_score")
ax = sns.heatmap(df_split_leaf)
plt.show()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
## Randomized search
<!-- #endregion -->

```python
distributions = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': uniform(),
              'min_samples_leaf': randint(1, 50),
              'max_depth': randint(1, 100),
              'max_features': randint(1, X.shape[1])}
random_search = RandomizedSearchCV(tree_classifier, distributions, random_state=0, cv=100, verbose=True)
random_search.fit(X,Y)
```

```python
print(random_search.best_params_)
```

```python
df_random_results = pd.DataFrame(random_search.cv_results_)
df_random_results = df_random_results.sort_values(by=["rank_test_score"])
```

```python
plt.plot(df_random_results.mean_test_score, '.')
plt.ylim((0,1))
plt.axhline(Y_train.mean(), color='r')
plt.show()
```

```python
df_random_results['param_max_depth'].fillna(value='No max depth', inplace = True)
df_random_results.groupby('param_max_depth').mean_test_score.mean().plot.bar()
plt.ylim((0.7, 0.8))
plt.show()
```

```python
df_random_results.groupby('param_criterion').mean_test_score.mean().plot.bar()
```

```python
df_random_results.groupby('param_splitter').mean_test_score.mean().plot.bar()
```

```python
df_random_results.groupby('param_min_samples_split').mean_test_score.mean().plot.bar()
```

```python
df_random_results.groupby('param_max_features').mean_test_score.mean().plot.bar()
```

```python
df_random_results.groupby('param_min_samples_leaf').mean_test_score.mean().plot.bar()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
# Gridsearch for random forest
<!-- #endregion -->

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Random gridsearch
<!-- #endregion -->

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
X, Y = exctract_target(dataset_processed)
```

```python
EXPERIMENT_NAME = "mlflow-decisiontree_depth"
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME).name
```

```python
max_depth = list(range(4, 50, 8))
max_depth.append(None)

min_samples_split = list(range(2, 30, 7))

max_features = list(range(2, X.shape[1]+1, 2))

param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': min_samples_split,
              'max_depth': max_depth,
              'max_features':max_features,
              'min_samples_leaf': list(range(1,20, 4))
             }
clf_tree = tree.DecisionTreeClassifier()
mlf_search = GridSearchCV(clf_tree, param_grid, n_jobs=-1)

mlflow.sklearn.autolog(log_models=False, max_tuning_runs=200)
with mlflow.start_run() as run:
    mlf_search.fit(X, Y)
```
