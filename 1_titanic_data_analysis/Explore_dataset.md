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
```

```python tags=[]
df = pd.read_csv("../data/titanic/train.csv")
```

```python
df.head()
```

```python
print("Features in dataset:")
for col in df.columns:
    print(col)
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
    # Format is <surname, title. names [something additional?]>
    raw_list = full_name.split('.')[1].split(' ')
    cleared  = [x for x in raw_list if x]
    return cleared[:2]
```

```python
names = []
df = df.reset_index(drop=True) 

for index, row in df.iterrows():
    names.extend(get_first_second_names(row['Name']))
```

```python
name_occurence_count = Counter(names)
```

```python
print("10 Most common names:")
name_occurence_count.most_common(10)
```

```python
print("10 Most rare names:")
name_occurence_count.most_common()[-10:]
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

# Survived against separate single features


## against age

```python
df_age_survived=df.dropna(subset=['Age','Survived'])
```

```python tags=[]
sns.boxplot(data=df_age_class, y="Age", x="Survived").set_title("0 = No, 1 = Yes")
```

```python
def percent_of(num_a, num_b):
    return (num_a / num_b) * 100
```

## against sex

```python
df_sex_survived = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
```

```python
df_sex_survived.plot(kind='bar')
```

```python
sum_survived = df_sex_survived[1].sum()
sum_not_survived = df_sex_survived[0].sum()

data = [['female', percent_of(df_sex_survived[1]['female'], sum_survived ), percent_of(df_sex_survived[0]['female'], sum_not_survived)],  
        ['male', percent_of(df_sex_survived[1]['male'], sum_survived ), percent_of(df_sex_survived[0]['male'], sum_not_survived)],
        ]
  
df_sex_survived_percent = pd.DataFrame(data, columns=['Sex', 'Survived', 'Not_Survived' ])
```

```python tags=[]
df_sex_survived_percent.plot(kind='bar', x='Sex', title='Survived against sex in percents')
```

## against age

```python
df_age_survived = df.groupby(['Survived', pd.cut(df.Age, range(0,100,10))]).size().unstack(fill_value=0)
```

```python tags=[]
df_age_survived
```

```python
df_age_survived.plot(kind='bar')
```

## against class

```python
df_class_survived = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
```

```python
df_class_survived.plot(kind='bar')
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

```python

```
