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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from time import time
from sklearn.preprocessing import MinMaxScaler
```

```python
!pip install scikit-learn
```

```python
sns.set_theme()
%matplotlib inline
```

```python
df = pd.read_csv('../data/AirPassengers.csv', parse_dates=[0], index_col=0,  
                            date_parser=lambda s:datetime.strptime(s, '%Y-%m'))
df = df.rename(columns={"#Passengers": "Num_Passengers"})
df = df.asfreq(pd.infer_freq(df.index))
```

```python
df.head()
```

```python
print(df.dtypes)
```

```python
print("Nan by columns:")
for col in df.columns:
    print(col, " - ", df[col].isna().sum())
    
print("Minimum number of passagers overall:", df["Num_Passengers"].min())
print("Maximum number of passagers overall:", df["Num_Passengers"].max())
print("Amount of measurements:", df["Num_Passengers"].count())
```

```python
plt.title("Amount of air passagers by month")
sns.lineplot(x="Month", y="Num_Passengers",
             data=df)
```

# STL Decompozition

```python
stl = STL(df)
result = stl.fit()
```

```python
seasonal, trend, resid = result.seasonal, result.trend, result.resid
```

```python
plt.figure(figsize=(8,6))

plt.subplot(4,1,1)
plt.plot(df)
plt.title('Original Series', fontsize=16)

plt.subplot(4,1,2)
plt.plot(trend)
plt.title('Trend', fontsize=16)

plt.subplot(4,1,3)
plt.plot(seasonal)
plt.title('Seasonal', fontsize=16)

plt.subplot(4,1,4)
plt.plot(resid)
plt.title('Residual', fontsize=16)

plt.tight_layout()
```

# ACF

```python
acf_plot = plot_acf(df.Num_Passengers, lags=100)
```

# PACF

```python
pacf_plot = plot_pacf(df.Num_Passengers)
```

# Stationarity

```python
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
```

```python
adf_test(df)
```

```python
df.rolling(window = 12).mean().plot(figsize=(8,4), color="tab:red", title="Rolling Mean over 12 month period");
```

```python
df.rolling(window = 12).var().plot(figsize=(8,4), color="tab:red", title="Rolling Variance over 12 month period");
```

## Normalize

```python
avg, dev = df.mean(), df.std()
```

```python
normalized_df = (df - avg) / dev
```

```python
plt.title("Normalized data")
sns.lineplot(x="Month", y="Num_Passengers",
             data=normalized_df)
```

## Remove trend

```python
first_dif = normalized_df.diff()[1:]
```

```python
plt.title("First Difference")
sns.lineplot(x="Month", y="Num_Passengers",
             data=first_dif)
```

## Remove Seasonality

```python
yearly_dif = first_dif.diff()[12:]
```

```python
plt.title("Yearly Difference")
sns.lineplot(x="Month", y="Num_Passengers",
             data=yearly_dif)
```

## Remove increasing variance 

```python
annual_variance = yearly_dif.groupby(yearly_dif.index.year).std()
```

```python
annual_variance
```

```python
scaler = MinMaxScaler()
stationary_df = scaler.fit(yearly_dif)
```

```python
plt.title("Stationary")
sns.lineplot(x="Month", y="Num_Passengers",
             data=stationary_df)
```
