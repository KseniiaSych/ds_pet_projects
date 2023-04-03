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
from itertools import chain

from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
```

```python
sns.set_theme()
%matplotlib inline
```

```python
def read_data(path, experiment):
    df_rh = pd.read_csv(path)
    df_rh.columns = df_rh.columns.str.replace(' ', '')
    df_rh = df_rh.assign(EXPERIMENT=lambda x: experiment)
    print(f"{len(df_rh)} samples in {experiment} group")
    return df_rh
```

```python
df_research = read_data("../data/dental/research_group.csv", 1)
df_control = read_data("../data/dental/control_group.csv", 0) 
```

```python
columns_to_drop = df_research.columns[list(chain(range(1,6), range(7,12), range(13,18)))] 
df = pd.concat([df_research, df_control], ignore_index=True).drop(columns_to_drop, axis=1)
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

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# Utils
<!-- #endregion -->

```python
def print_line(count=25):
    print('_'*count)
```

```python
def print_NA_count(df, columns):
    for col in columns:
        print(col, " - ", df[col].isna().sum())
```

```python
def check_normality(data, name = None):
    print_line()
    print("Running Shapiro-Wilk test")
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
    print("Running Levene's  test")
    _ , p_value_var= levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        print(f"Reject null hypothesis >> The variances of the samples {name} are different")
        return False
    else:
        print(f"Fail to reject null hypothesis >> The variances of the samples {name} are same")
        return True
```

```python
def check_ttest_ind(group1, group2, equal_var = True, name = None):
    print_line()
    print("Running T-test")
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
    r_mean = research.mean()
    c_mean = control.mean()
    diff = r_mean - c_mean
    print(f"{name}: \n subset research mean ({r_mean}) \n is different from subset control mean ({c_mean}) \n on {diff}")
```

```python
def compare_two_groups(df1, df2, column, name):
    print_line(50)
    equal_var = check_variance_homogeneity(df1[column], df2[column], name)
    check_ttest_ind(df1[column], df2[column], equal_var, name)
    print_mean_diff(df1[column], df2[column], name)
```

```python
def plot_violin(dataset, columns, defining_col, title):
    data_to_plot = pd.melt(dataset, value_vars=columns, id_vars=defining_col)
    sns.violinplot(x='variable', y='value', hue=defining_col, data=data_to_plot, split=True)
    plt.title(title)
    plt.show()
```

<!-- #region tags=[] -->
# VAS comparision
<!-- #endregion -->

```python
print_NA_count(df,['VAS1','VAS2', 'VAS7'])
```

```python
df_vas = df.copy().assign(VAS_SUM = lambda x: x.VAS1 + x.VAS2 + x.VAS7)

df_vas_r = df_vas.query("EXPERIMENT==1")
df_vas_c = df_vas.query("EXPERIMENT==0")
```

```python
plot_violin(df_vas, ["VAS_SUM"], "EXPERIMENT", "Comparision by sum of VAS")
```

```python
sns.boxplot(data=df_vas, x="EXPERIMENT", y="VAS_SUM")
plt.title("Comparision by sum of VAS")
plt.show()
```

```python
qqplot(df_vas_r.VAS_SUM, norm, fit=True, line="45")
plt.title("Compare research group VAS SUM distribution to normal")
plt.show()
```

```python
qqplot(df_vas_c.VAS_SUM,norm,fit=True,line="45")
plt.title("Compare control group VAS SUM distribution to normal")
plt.show()
```

```python
plot_violin(df_vas, ["VAS1", "VAS2", "VAS7"], "EXPERIMENT", "Compare VAS' difference to before operation")
```

```python
check_normality(df_vas_r.VAS_SUM, "VAS_sum research")
check_normality(df_vas_c.VAS_SUM, "VAS_sum control")

compare_two_groups(df_vas_r, df_vas_c, "VAS_SUM", "VAS_sum")
```

```python
compare_two_groups(df_vas_r, df_vas_c, "VAS1", "VAS1")
compare_two_groups(df_vas_r, df_vas_c, "VAS2", "VAS2")
compare_two_groups(df_vas_r, df_vas_c, "VAS7", "VAS7")
```

# Mouth opening comparision

```python
print_NA_count(df, ['MMO0','MMO2', 'MMO7'])
```

```python
df_mmo = df.dropna(subset= ['MMO0','MMO2', 'MMO7'])
df_mmo = df_mmo.assign(MMO2_DIFF = lambda x: (x.MMO2 - x.MMO0) / x.MMO0)
df_mmo = df_mmo.assign(MMO7_DIFF = lambda x: (x.MMO7 - x.MMO0) / x.MMO0)
```

```python
df_mmo.head()
```

```python
plot_violin(df_mmo, ['MMO2_DIFF', 'MMO7_DIFF'], "EXPERIMENT", "Comparision of MMO' difference in different days")
```

```python
df_mmo_r = df_mmo.query("EXPERIMENT==1")
df_mmo_c = df_mmo.query("EXPERIMENT==0")

compare_two_groups(df_mmo_r, df_mmo_c, "MMO2_DIFF", "percentage difference of MMO 2 day")
compare_two_groups(df_mmo_r, df_mmo_c, "MMO7_DIFF", "percentage difference of MMO 7 day")
```

# Face measurement comparision

```python
print_NA_count(df, ['Sum0','Sum2', 'Sum7'])
```

```python
df_sum = df.dropna(subset= ['Sum0','Sum2', 'Sum7'])
df_sum = df_sum.assign(SUM2_DIFF = lambda x: (x.Sum2 - x.Sum0) / x.Sum0)
df_sum = df_sum.assign(SUM7_DIFF = lambda x: (x.Sum7 - x.Sum0) / x.Sum0)
```

```python
plot_violin(df_sum, ['SUM2_DIFF', 'SUM7_DIFF'], "EXPERIMENT", "Compare SUM' difference to before operation in percents")
```

```python
s_research_index = df_sum["EXPERIMENT"] == 1
s_control_index = df_sum["EXPERIMENT"] == 0

compare_two_groups(df_sum[s_research_index], df_sum[s_control_index], "SUM2_DIFF", "percantage difference of SUM measurements 2 day")
compare_two_groups(df_sum[s_research_index], df_sum[s_control_index], "SUM7_DIFF", "percantage difference of SUM measurements 7 day")
```

# Medicine taken comparision

```python
compared_columns = ['Med1','Med2', 'Med7']
```

```python
print_NA_count(df, compared_columns)
```

```python
df_meds = df.dropna(subset=compared_columns)
for col in compared_columns:
    df_meds[col] = df_meds[col].str.extract('(\d+)').astype(int)

```

```python
df_meds
```

```python
plot_violin(df_meds, compared_columns, "EXPERIMENT", "Compare Meds' taken")
```

```python
df_meds_r_index = df_meds["EXPERIMENT"] == 1
df_meds_c_index = df_meds["EXPERIMENT"] == 0

compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med1", "Meds taken 1 day")
compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med2", "Meds taken 2 day")
compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med2", "Meds taken 7 day")
```

# Cluster people and compare

```python
#TBD
```
