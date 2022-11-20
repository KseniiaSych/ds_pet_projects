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
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
sns.set_theme()
%matplotlib inline
```

```python tags=[]
df = pd.read_csv("../data/titanic/train.csv")
```

```python
df.head()
```

```python
print("Features in dataset:")
print(*list(df.columns))
```

```python
print(df.dtypes)
```

```python
print("Nan by columns:")
for col in df.columns:
    print(col, " - ", df[col].isna().sum())
```

<!-- #region tags=[] -->
# Most common and most rare names
<!-- #endregion -->

```python
def get_first_second_names(full_name):
    # Format is <surname, title. name/s>
    # in case of Mrs - name is in the bracets after husband's name
    
    full_name = full_name.replace('"', ' ')
    if '.' not in full_name:
        return []
    title_name = full_name.split('.')
    title = title_name[0].split(' ')[-1]
    if title =='Mrs' and '(' in full_name:  
        wife_name = title_name[1].split('(')[1].split(' ')
        output =  wife_name[:-1] if len(wife_name)>1 else []
    else:
        output= title_name[1].split(' ')
    return [name for name in  output if len(name)>1]
```

```python
get_first_second_names('Hewlett, Mrs. (Mary D Kingcome)')
```

```python
names_df = pd.concat([df.Sex,  df['Name'].apply(get_first_second_names)], axis=1)
names_df = names_df.explode('Name')
```

```python
names_df.head()
```

```python
names_stat = names_df.groupby('Sex').value_counts()
```

```python
print("10 Most common names:")
names_stat.groupby(level=0).nlargest(10)
```

```python
df.Sex.value_counts()
```

```python

```

```python
print("10 Most common names:")
names_stat.groupby(level=0).nsmallest(10)
```

# Age distribution by class

```python
df_age_class=df.dropna(subset=['Age','Pclass'])
```

```python
rel = sns.displot(data=df_age_class, x="Age", col="Pclass", hue="Pclass", kde=True)
rel.fig.subplots_adjust(top=.85)
rel.fig.suptitle('1st = Upper, 2nd = Middle, 3rd = Lower')
```

```python
df_age_class=df.dropna(subset=['Age','Pclass', 'Sex'])
sns.violinplot(data=df_age_class, x="Pclass", y="Age", hue="Sex", split=True)
```

# Survived against separate single features


## against sex

```python
plt = df.groupby('Sex').sum().Survived.plot.bar(width=1.0)
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Count')
```

```python
plt = (df.groupby('Sex').Survived.mean()).plot.bar(width=1.0)
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Probability')
```

## against age

```python
bins= range(0,100,5)
df.groupby([ pd.cut(df.Age, bins)]).Survived.sum().plot.bar()
```

```python
bins= range(0,100,5)
df_age_survived_grouped = df.groupby([ pd.cut(df.Age, bins)])
df_age_survived = df_age_survived_grouped.Survived.mean()
df_age_survived.plot.bar()
```

```python
bins= [0,5,10, 15, 20,60,100]
df_age_survived_grouped = df.groupby([ pd.cut(df.Age, bins)])
df_age_survived = df_age_survived_grouped.Survived.mean()
df_age_survived.plot.bar()
```

```python
df_age_survived=df.dropna(subset=['Age','Survived'])
```

```python
sns.boxplot(data=df_age_class, y="Age", x="Survived").set_title("0 = No, 1 = Yes")
```

## against class

```python
plt = df[['Pclass', 'Survived']].groupby('Pclass').sum().Survived.plot.bar(width=1.0)
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Count')
```

```python
plt = (df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()).plot.bar(width=1.0)
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')
```

## against family size

```python
df_sib_survived = df.groupby(['SibSp', 'Survived']).size().unstack(fill_value=0)
df_sib_survived.plot(kind='bar')
```

```python
df_patch_survived = df.groupby(['Parch', 'Survived']).size().unstack(fill_value=0)
df_patch_survived.plot(kind='bar')
```

```python
df['FamilySize']=df['Parch']+df['SibSp']
df_family_size_survived = df.groupby(['FamilySize', 'Survived']).size().unstack(fill_value=0)
df_family_size_survived.plot(kind='bar')
```

# against embarked

```python
plt = df[['Embarked', 'Survived']].groupby('Embarked').sum().Survived.plot.bar(width=1.0)
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Count')
```

```python
plt = (df.groupby('Embarked').Survived.sum()/df.groupby('Embarked').Survived.count()).plot.bar(width=1.0)
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')
```

<!-- #region tags=[] -->
# against fare
<!-- #endregion -->

```python
bins = range(0,int(df.Fare.max()),20)
df_fare_survived_grouped = df.groupby([ pd.cut(df.Fare, bins)])
df_fare_survived = df_fare_survived_grouped.Survived.sum()/df_fare_survived_grouped.Survived.count()
df_fare_survived.plot.bar(width=1.0)
```

```python
bins = [0,20,80, 140, 160, 180, 220,260,500]
df_fare_survived_grouped = df.groupby([ pd.cut(df.Fare, bins)])
df_fare_survived = df_fare_survived_grouped.Survived.sum()/df_fare_survived_grouped.Survived.count()
df_fare_survived.plot.bar(width=1.0)
```

```python

```
