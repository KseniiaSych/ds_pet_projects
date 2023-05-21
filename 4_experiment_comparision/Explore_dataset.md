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
from itertools import chain
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import levene, shapiro, ttest_ind, norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

from kneed import KneeLocator
from statsmodels.graphics.gofplots import qqplot
from sklearn.decomposition import PCA
import pacmap
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
def check_ttest_ind_value(group1, group2):
    _, pvalue = ttest_ind(group1, group2, equal_var = False)
    return "not likely" if pvalue <0.05 else "likely"
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
def mean_value(research, control):
    return  research.mean() - control.mean()
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
def plot_violin(dataset, columns, defining_col, title, split = True):   
    data_to_plot = pd.melt(dataset, value_vars=columns, id_vars=defining_col)
    sns.violinplot(x='variable', y='value', hue=defining_col, data=data_to_plot, split=split)
    plt.title(title)
    plt.show()
```

```python
def preprocess_columnt_to_number(df, columns):
    for col in columns:
        df[col] = df[col].str.extract('(\d+)').astype(int)
    return df
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
med_compared_columns = ['Med1','Med2', 'Med7']
```

```python
print_NA_count(df, med_compared_columns)
```

```python
df_meds = df.dropna(subset=med_compared_columns)
df_meds = preprocess_columnt_to_number(df_meds, med_compared_columns)
```

```python
df_meds
```

```python
plot_violin(df_meds, med_compared_columns, "EXPERIMENT", "Compare Meds' taken")
```

```python
df_meds_r_index = df_meds["EXPERIMENT"] == 1
df_meds_c_index = df_meds["EXPERIMENT"] == 0

compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med1", "Meds taken 1 day")
compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med2", "Meds taken 2 day")
compare_two_groups(df_meds[df_meds_r_index], df_meds[df_meds_c_index], "Med2", "Meds taken 7 day")
```

# Control group - compare by operator

```python
prepared_control = df_control.dropna(subset=["Operator", "VAS1", "VAS2", "VAS7", "Med1", "Med2", "Med7"])
prepared_control =  preprocess_columnt_to_number(prepared_control, ["Med1", "Med2", "Med7"])
```

```python
plot_violin(prepared_control, ["VAS1", "VAS2", "VAS7"], "Operator", "Compare VAS' by operator", False)
```

```python
plot_violin(prepared_control, ["Med1", "Med2", "Med7"], "Operator", "Compare Meds taken by operator", False)
```

```python
prepared_control = prepared_control.assign(MMO2_DIFF = lambda x: (x.MMO2 - x.MMO0) / x.MMO0)
prepared_control = prepared_control.assign(MMO7_DIFF = lambda x: (x.MMO7 - x.MMO0) / x.MMO0)
prepared_control = prepared_control.assign(SUM2_DIFF = lambda x: (x.Sum2 - x.Sum0) / x.Sum0)
prepared_control = prepared_control.assign(SUM7_DIFF = lambda x: (x.Sum7 - x.Sum0) / x.Sum0)
```

```python
plot_violin(prepared_control, ["MMO2_DIFF", "MMO7_DIFF"], "Operator", "Compare MMO perentage difference", False)
```

```python
plot_violin(prepared_control, ["SUM2_DIFF", "SUM7_DIFF"], "Operator", "Compare SUM perentage difference", False)
```

# Cluster people and compare

```python
features_columns = ["VAS1", "VAS2","Med1", "Med2", "Operator"]
columns_for_calculation = ["MMO2", "MMO0", "MMO7", "Sum2", "Sum0", "Sum7"]

df_X = df[features_columns+columns_for_calculation]
df_X = df_X.dropna()

df_X = df_X.assign(MMO2_DIFF = lambda x: (x.MMO2 - x.MMO0) / x.MMO0)
df_X = df_X.assign(SUM2_DIFF = lambda x: (x.Sum2 - x.Sum0) / x.Sum0)
df_X = df_X.drop(columns = columns_for_calculation)
df_X = preprocess_columnt_to_number(df_X, ["Med1", "Med2"])

operator_Y= df_X.Operator
df_X = df_X.drop(columns = "Operator")
```

```python
df_X.head()
```

```python
def normilize_dataframe(df):
    x = df.values 
    scaler = preprocessing.PowerTransformer()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns = df.columns)
```

```python
df_X = normilize_dataframe(df_X)
```

```python
df_X.head()
```

<!-- #region tags=[] -->
## Find number of clusters
<!-- #endregion -->

```python
def run_kmeans(df, k, random_state=42, init='k-means++'):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto", init=init)
    kmeans.fit(df)
    return kmeans
```

```python
def elbow_plot(df):
    sse = {}  
    for k in range(1, 20):
        kmeans = run_kmeans(df, k)
        sse[k] = kmeans.inertia_
    
    plt.title('Elbow plot for K selection')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()),
                 y=list(sse.values()))
    plt.show()
```

```python
elbow_plot(df_X)
```

```python
def silhouette_score_plot(df, k_max=20):
    silhouette_scores = [] 
    k_range  =  range(2, k_max)
    for k in k_range:
        kmeans = run_kmeans(df, k)
        silhouette_scores.append(silhouette_score(df, kmeans.labels_))
    
    fig, ax = plt.subplots()
    ax.plot(k_range, silhouette_scores, 'bx-')
    ax.set_title('Silhouette Score Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Scores')
    plt.xticks(k_range)
    plt.tight_layout()
    plt.show()
```

```python
silhouette_score_plot(df_X)
```

```python
def find_k(df):
    sse = {}
    for k in range(1, 20):
        kmeans = run_kmeans(df, k)
        sse[k] = kmeans.inertia_
    
    kn = KneeLocator(x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing')
    return kn.knee
```

```python
print(f"Knee recommended by algorithm - {find_k(df_X)}")
```

```python
kmeans = run_kmeans(df_X, 5)
df_clustered = df_X.assign(cluster=kmeans.labels_)
```

```python
df_clustered
```

## Visualize results

```python
df_combine = df_clustered.assign(Operator=operator_Y.values)
df_combine.groupby('cluster')['Operator'].agg(['unique'])
```

```python
df_combine.groupby(['cluster','Operator']).size().unstack(level=1).plot(kind = 'bar')
plt.title("Operators distribution by clusters")
plt.show()
```

```python
cluster_colors = ['#0000FF', '#FF4040', '#7FFF00', '#006400', '#9932CC', '#FF1493']
```

```python
pca_scaled_std = PCA(n_components=2, random_state=42)
X_std_pca = pca_scaled_std.fit_transform(df_X)
```

```python
ax = plt.subplot()
y = df_clustered.cluster
y_values = df_clustered.cluster.unique()
for l, c, m in zip(y_values, cluster_colors[0:len(y_values)], ('^', 's', 'o', 'P', 'D')):
    ax.scatter(X_std_pca[y == l, 0],
                X_std_pca[y == l, 1],
                color=c,
                label='cluster %s' % l,
                alpha=0.9,
                marker=m
                )
ax.set_title("PCA Visualization by cluster")
plt.show()
```

```python
ax = plt.subplot()
y = operator_Y
y_values =y.unique()
for l, c, m in zip(y_values, cluster_colors[0:len(y_values)], ('^', 's', 'o', 'P', 'D')):
    ax.scatter(X_std_pca[y == l, 0],
                X_std_pca[y == l, 1],
                color=c,
                label='cluster %s' % l,
                alpha=0.9,
                marker=m
                )
ax.set_title("PCA Visualization by operators")
plt.show()
```

```python
os.environ["_RANDOM_STATE"] = "42"
embedding = pacmap.PaCMAP(n_components=2)
X_std_pacmap = embedding.fit_transform(df_X.to_numpy(), init="pca")
ax = plt.subplot()

y = df_clustered.cluster
y_values = df_clustered.cluster.unique()

for l, c, m in zip(y_values, cluster_colors[0:len(y_values)], ('^', 's', 'o')):
    ax.scatter(X_std_pacmap[y == l, 0],
                X_std_pacmap[y == l, 1],
                color=c,
                label='cluster %s' % l,
                alpha=0.9,
                marker=m
                )
ax.set_title("PACMAP Visualization")
plt.show()
```

```python

```
