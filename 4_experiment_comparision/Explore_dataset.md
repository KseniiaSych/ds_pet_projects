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
from scipy.stats import levene, shapiro, ttest_ind
```

```python
sns.set_theme()
%matplotlib inline
```

```python
df_research = pd.read_csv("../data/dental/research_group.csv")
df_research.columns = df_research.columns.str.replace(' ', '')
df_research = df_research.assign(EXPERIMENT=lambda x: 1)
print(f"{len(df_research)} samples in research group")

df_control = pd.read_csv("../data/dental/control_group.csv")
df_control.columns = df_control.columns.str.replace(' ', '')
df_control = df_control.assign(EXPERIMENT=lambda x:0)
print(f"{len(df_control)} samples in control group")
```

```python
df = pd.concat([df_research, df_control], ignore_index=True)
df = df.drop(['Tragion0','Exocanthion0','Alare0','Cheilon0','Pogonion0',
              'Tragion2','Exocanthion2','Alare2','Cheilon2','Pogonion2',
              'Tragion7','Exocanthion7','Alare7','Cheilon7','Pogonion7',
             ], axis=1)
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
# Utils
<!-- #endregion -->

```python
def print_line():
    print('_'*25)
```

```python
def print_NA_count(df, columns):
    for col in columns:
        print(col, " - ", df[col].isna().sum())
```

```python
def check_normality(data, name = None):
    print_line()
    _ , p_value_normality = shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality <0.05:
        print(f"Reject null hypothesis >> The data {name} is not normally distributed")
    else:
        print(f"Fail to reject null hypothesis >> The data {name} is normally distributed")
```

```python
def check_variance_homogeneity(group1, group2, name = None):
    print_line()
    _ , p_value_var= levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        print(f"Reject null hypothesis >> The variances of the samples {name} are different")
    else:
        print(f"Fail to reject null hypothesis >> The variances of the samples {name} are same")
```

```python
def check_ttest_ind(group1, group2, equal_var = True, name = None):
    print_line()
    _, pvalue = ttest_ind(group1, group2, equal_var = equal_var)
    print("p value:%.4f" % pvalue)
    if pvalue <0.05:
        print(f"Reject null hypothesis >> Difference for {name} is not likely to be random")
    else:
        print(f"Fail to reject null hypothesis >> Difference for {name} is likely due to random")
```

```python
def print_mean_diff(research, control, name = None):
    print_line()
    diff = research.mean()-control.mean()
    print(f"Research mean value for {name} is different from control mean on {diff}")
```

<!-- #region tags=[] -->
# VAS comparision
<!-- #endregion -->

```python
print_NA_count(df,['VAS1','VAS2', 'VAS7'])
```

```python
df_vas = df.copy()
df_vas = df_vas.assign(VAS_SUM = lambda x: x.VAS1 + x.VAS2 + x.VAS7)

research_index = df['EXPERIMENT'] == 1
control_index = df['EXPERIMENT'] == 0
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

```python
df_vas_r = df_vas[research_index]
df_vas_c = df_vas[control_index]

check_variance_homogeneity(df_vas_r.VAS_SUM, df_vas_c.VAS_SUM, "VAS_sum")

check_ttest_ind(df_vas_r.VAS_SUM, df_vas_c.VAS_SUM, True, "VAS_sum")
print_mean_diff(df_vas_r.VAS_SUM, df_vas_c.VAS_SUM, "VAS_sum")
```

```python
check_variance_homogeneity(df_vas_r.VAS1, df_vas_c.VAS1,"Vas week 1")
check_variance_homogeneity(df_vas_r.VAS2, df_vas_c.VAS2, "Vas week 2")
check_variance_homogeneity(df_vas_r.VAS7, df_vas_c.VAS7, "Vas week 7")

check_ttest_ind(df_vas_r.VAS1, df_vas_c.VAS1, name = "Vas week 1")
check_ttest_ind(df_vas_r.VAS2, df_vas_c.VAS2,  name ="Vas week 2")
check_ttest_ind(df_vas_r.VAS7, df_vas_c.VAS7, False, "Vas week 7")

print_mean_diff(df_vas_r.VAS1, df_vas_c.VAS1,"Vas week 1")
print_mean_diff(df_vas_r.VAS2, df_vas_c.VAS2, "Vas week 2")
print_mean_diff(df_vas_r.VAS7, df_vas_c.VAS7, "Vas week 7")
```

# Mouth opening comparision

```python
print_NA_count(df, ['MMO0','MMO2', 'MMO7'])
```

```python
df_mmo = df.dropna(subset= ['MMO0','MMO2', 'MMO7'])
df_mmo = df.assign(MMO2_DIFF = lambda x: (x.MMO2 - x.MMO0) / x.MMO0)
df_mmo = df_mmo.assign(MMO7_DIFF = lambda x: (x.MMO7 - x.MMO0) / x.MMO0)
m_research_index = df_mmo['EXPERIMENT'] == 1
m_control_index = df_mmo['EXPERIMENT'] == 0
```

```python
df_mmo.head()
```

```python
mmo_len = len(df_mmo)
df_mmo_differences = pd.concat([
    pd.DataFrame({"EXPERIMENT":df_mmo.EXPERIMENT,
                  "MMO_DIFF": df_mmo.MMO2_DIFF,
                  "TYPE": pd.Series('MMO2_DIF', index=range(mmo_len))
                 }),
    pd.DataFrame({"EXPERIMENT":df_mmo.EXPERIMENT,
                  "MMO_DIFF": df_mmo.MMO7_DIFF,
                  "TYPE": pd.Series('MMO7_DIF', index=range(mmo_len))
                 }),
                           ],
                           ignore_index=True)

sns.violinplot(data=df_mmo_differences, x="TYPE", y="MMO_DIFF", hue="EXPERIMENT", split=True)
plt.title("Comparision of MMO' difference in different weeks")
plt.show()

```

```python
df_mmo_r = df_mmo[m_research_index]
df_mmo_c = df_mmo[m_control_index]
```

```python
check_variance_homogeneity(df_mmo_r.MMO2_DIFF, df_mmo_c.MMO2_DIFF,"percentage difference of MMO 2 week")
check_variance_homogeneity(df_mmo_r.MMO7_DIFF, df_mmo_c.MMO7_DIFF, "percentage difference of MMO 7 week")

check_ttest_ind(df_mmo_r.MMO2_DIFF, df_mmo_c.MMO2_DIFF, name = "MMO 2 week")
check_ttest_ind(df_mmo_r.MMO7_DIFF, df_mmo_c.MMO7_DIFF, name = "MMO 7 week")

print_mean_diff(df_mmo_r.MMO2_DIFF, df_mmo_c.MMO2_DIFF, "MMO 2 week")
print_mean_diff(df_mmo_r.MMO7_DIFF, df_mmo_c.MMO7_DIFF, "MMO 7 week")
```

# Face measurement comparision

```python
print_NA_count(df, ['Sum0','Sum2', 'Sum7'])
```

```python
df_sum = df.dropna(subset= ['Sum0','Sum2', 'Sum7'])
df_sum = df_sum.assign(SUM2_DIFF = lambda x: (x.Sum2 - x.Sum0) / x.Sum0)
df_sum = df_sum.assign(SUM7_DIFF = lambda x: (x.Sum7 - x.Sum0) / x.Sum0)

s_research_index = df_sum['EXPERIMENT'] == 1
s_control_index = df_sum['EXPERIMENT'] == 0
```

```python
sum_len = len(df_sum)
df_sum_differences = pd.concat([
    pd.DataFrame({"EXPERIMENT":df_sum.EXPERIMENT,
                  "SUM_DIFF": df_sum.SUM2_DIFF,
                  "TYPE": pd.Series('SUM2_DIFF', index=range(sum_len))
                 }),
    pd.DataFrame({"EXPERIMENT":df_sum.EXPERIMENT,
                  "SUM_DIFF": df_sum.SUM7_DIFF,
                  "TYPE": pd.Series('SUM7_DIFF', index=range(sum_len))
                 }),
                           ],
                           ignore_index=True)

sns.violinplot(data=df_sum_differences, x="TYPE", y="SUM_DIFF", hue="EXPERIMENT", split=True)
plt.title("Comparision of SUM' difference in different weeks")
plt.show()
```

```python
df_sum_r = df_sum[s_research_index]
df_sum_c = df_sum[s_control_index]
```

```python
check_variance_homogeneity(df_sum_r.SUM2_DIFF, df_sum_c.SUM2_DIFF," difference of SUM measurements 2 week")
check_variance_homogeneity(df_sum_r.SUM7_DIFF, df_sum_c.SUM7_DIFF," difference of SUM measurements 7 week")

check_ttest_ind(df_sum_r.SUM2_DIFF, df_sum_c.SUM2_DIFF, name = "difference of SUM measurements 2 week")
check_ttest_ind(df_sum_r.SUM7_DIFF, df_sum_c.SUM7_DIFF, name = "difference of SUM measurements 7 week")

print_mean_diff(df_sum_r.SUM2_DIFF, df_sum_c.SUM2_DIFF, "difference of SUM measurements 2 week")
print_mean_diff(df_sum_r.SUM7_DIFF, df_sum_c.SUM7_DIFF, "difference of SUM measurements 7 week")
```

# Cluster people and compare

```python
#TBD
```
