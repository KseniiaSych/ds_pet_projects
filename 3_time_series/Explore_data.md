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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from time import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# STL Decompozition
<!-- #endregion -->

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

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# ACF & PACF
<!-- #endregion -->

```python
acf_plot = plot_acf(df.Num_Passengers, lags=100)
```

```python
pacf_plot = plot_pacf(df.Num_Passengers)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# Stationarity
<!-- #endregion -->

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
sns.set(rc={'figure.figsize':(20, 4)})
sns.lineplot(x="Month", y="Num_Passengers",
             data=normalized_df)
```

## Remove trend

```python
first_dif = normalized_df.diff()[1:]
```

```python
plt.title("First Difference")
sns.lineplot(x="Month", y="Num_Passengers", data=first_dif, )
```

```python
adf_test(first_dif)
```

## Remove Seasonality

```python
yearly_dif = first_dif.diff(12)[12:]
```

```python
plt.title("Yearly Difference")
sns.lineplot(x="Month", y="Num_Passengers",
             data=yearly_dif)
```

```python
adf_test(yearly_dif)
```

<!-- #region tags=[] -->
# ACF & PACF
<!-- #endregion -->

```python
sns.set(rc={'figure.figsize':(5, 5)})
acf_plot = plot_acf(yearly_dif.Num_Passengers, lags=100)
```

```python
pacf_plot = plot_pacf(yearly_dif.Num_Passengers)
```

# Comparing different models

<!-- #region tags=[] -->
## Utils
<!-- #endregion -->

```python
preprocessed_df = yearly_dif
```

```python
def fit_model(order, dataframe, split):
    model_fit = []
    predictions =  []
    
    for cur_date in pd.date_range(split,  dataframe.index[-1], freq='MS'):
        model = ARIMA(dataframe[:cur_date - timedelta(days=1)], order = order)
        fit = model.fit()
        pred = fit.predict(start=cur_date, end=cur_date) 
        
        model_fit.append(fit)
        predictions.append(pred)
    
    predictions_df = pd.DataFrame(pd.concat(predictions, axis=0), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
def get_split_date(data, split_coef):
    if split_coef<0 or split_coef>1:
        raise ValueError("Split coefficient should be in range [0,1]")
    split = math.floor(len(data) * split_coef)
    split = split if split<len(data) else split-1
    return data.index[split]
```

<!-- #region tags=[] -->
## AR model
<!-- #endregion -->

```python
split = get_split_date(preprocessed_df, 0.95)

train_data = preprocessed_df[:split]
test_data = preprocessed_df[split + timedelta(days=1):]
```

```python
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
```

```python
ar_model = ARIMA(train_data, order=(2,0,0))
ar_fit = ar_model.fit()
```

```python
print(ar_fit.summary())
```

```python
predictions = ar_fit.predict(start=pred_start_date, end=pred_end_date)
```

```python
plt.figure(figsize=(10,4))

plt.plot(test_data)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)
```

```python
ar_split = get_split_date(preprocessed_df, 0.70)
ar_test_data = preprocessed_df[ar_split:]
ar_models, ar_predictions = fit_model((2,0,0), preprocessed_df, ar_split)
```

```python
plt.figure(figsize=(8,4))

plt.plot(ar_test_data)
plt.plot(ar_predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)
```

```python
ar_mse = mean_squared_error(ar_test_data['Num_Passengers'], ar_predictions['Num_Passengers'])  
ar_rmse = mean_squared_error(ar_test_data['Num_Passengers'], ar_predictions['Num_Passengers'],
                                      squared=False)          
ar_mae = mean_absolute_error(ar_test_data['Num_Passengers'], ar_predictions['Num_Passengers'])
```

```python
print("MSE - ", ar_mse)
print("RMSE - ", ar_rmse)
print("MAE - ", ar_mae)
```

## MA model

```python
ma_split = get_split_date(preprocessed_df, 0.70)
ma_test_data = preprocessed_df[ma_split:]
ma_models, ma_predictions = fit_model((0,0,2), preprocessed_df, ma_split)
```

```python
plt.figure(figsize=(8,4))

plt.plot(ma_test_data)
plt.plot(ma_predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)
```

```python
ma_mse = mean_squared_error(ma_test_data['Num_Passengers'], ma_predictions['Num_Passengers'])  
ma_rmse = mean_squared_error(ma_test_data['Num_Passengers'], ma_predictions['Num_Passengers'],
                                      squared=False)          
ma_mae = mean_absolute_error(ma_test_data['Num_Passengers'], ma_predictions['Num_Passengers'])
```

```python
print("MSE - ", ma_mse)
print("RMSE - ", ma_rmse)
print("MAE - ", ma_mae)
```

## ARMA model

```python
split = get_split_date(preprocessed_df, 0.70)
test_data_spl = preprocessed_df[split:]
models, predictions = fit_model((2,0,2), preprocessed_df, split)
```

```python
plt.figure(figsize=(8,4))

plt.plot(test_data_spl)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)
```

```python
mse = mean_squared_error(test_data_spl['Num_Passengers'], predictions['Num_Passengers'])  
rmse = mean_squared_error(test_data_spl['Num_Passengers'], predictions['Num_Passengers'],
                                      squared=False)          
mae = mean_absolute_error(test_data_spl['Num_Passengers'], predictions['Num_Passengers'])
```

```python
print("MSE - ", mse)
print("RMSE - ", rmse)
print("MAE - ", mae)
```

## ARIMA model

```python
isplit = get_split_date(df, 0.70)
itest_data_spl = df[isplit:]
imodels, ipredictions = fit_model((2,2,2), df, isplit)
```

```python
plt.figure(figsize=(8,4))

plt.plot(itest_data_spl)
plt.plot(ipredictions)

plt.legend(('Data', 'Predictions'), fontsize=16)
```

```python
imse = mean_squared_error(itest_data_spl['Num_Passengers'], ipredictions['Num_Passengers'])  
irmse = mean_squared_error(itest_data_spl['Num_Passengers'], ipredictions['Num_Passengers'],
                                      squared=False)          
imae = mean_absolute_error(itest_data_spl['Num_Passengers'], ipredictions['Num_Passengers'])
```

```python
print("MSE - ", imse)
print("RMSE - ", irmse)
print("MAE - ", imae)
```
