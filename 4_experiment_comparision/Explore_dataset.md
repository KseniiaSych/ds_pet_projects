---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene
```

```python
sns.set_theme()
%matplotlib inline
```

```python
df_research = pd.read_csv("../data/dental/research_group.csv")
df_research.columns = df_research.columns.str.replace(' ', '')
df_research = df_research.assign(EXPERIMENT=lambda x: 1)

df_control = pd.read_csv("../data/dental/control_group.csv")
df_control.columns = df_control.columns.str.replace(' ', '')
df_control = df_control.assign(EXPERIMENT=lambda x:0)
```

```python
df = pd.concat([df_research, df_control], ignore_index=True)
```

```python
df.info()
```

```python
df.head()
```

```python
df.describe()
```

<!-- #region tags=[] -->
# VAS comparision
<!-- #endregion -->

## Visual comparision

```python
df_vas = df.dropna(subset=['VAS1','VAS2', 'VAS7'])
df_vas = df_vas.assign(VAS_SUM = lambda x: x.VAS1 + x.VAS2 + x.VAS7)
```

```python
sns.boxplot(data=df_vas, x="EXPERIMENT", y="VAS_SUM")
plt.title("Comparision by sum of VAS")
plt.show()
```

```python
vas_len = len(df_vas)
df_vas_by_week = pd.concat([
    pd.DataFrame({"EXPERIMENT":df_vas.EXPERIMENT,
                  "VAS": df_vas.VAS1,
                  "TYPE": pd.Series('VAS1', index=range(vas_len))
                 }),
    pd.DataFrame({"EXPERIMENT":df_vas.EXPERIMENT,
                  "VAS": df_vas.VAS2,
                  "TYPE": pd.Series('VAS2', index=range(vas_len))
                 }),
    pd.DataFrame({"EXPERIMENT":df_vas.EXPERIMENT,
                  "VAS": df_vas.VAS7,
                  "TYPE": pd.Series('VAS7', index=range(vas_len))
                 })
                           ],
                           ignore_index=True)

```

```python
sns.violinplot(data=df_vas_by_week, x="TYPE", y="VAS", hue="EXPERIMENT", split=True)
plt.title("Comparision of VAS' in different weeks")
plt.show()
```

## Hypothesis Testing

```python
def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis >> The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis >> The variances of the samples are same.")
```

```python
print("Testing sum of VAS'")
check_variance_homogeneity(df_vas.query("EXPERIMENT == 1").VAS_SUM, df_vas.query("EXPERIMENT == 0").VAS_SUM)
```

```python
print("Testing sum of VAS1")
check_variance_homogeneity(df_vas.query("EXPERIMENT == 1").VAS1, df_vas.query("EXPERIMENT == 0").VAS1)

print("Testing sum of VAS2")
check_variance_homogeneity(df_vas.query("EXPERIMENT == 1").VAS2, df_vas.query("EXPERIMENT == 0").VAS2)

print("Testing sum of VAS7")
check_variance_homogeneity(df_vas.query("EXPERIMENT == 1").VAS7, df_vas.query("EXPERIMENT == 0").VAS7)
```

# Mouth opening comparision

```python
#TBD
```

# Compare VAS and mouth opening by clustered groups

```python
#TBD
```
