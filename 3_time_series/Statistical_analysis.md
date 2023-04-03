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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing
from time import time
from dateutil.relativedelta import relativedelta
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from scalecast.Forecaster import Forecaster
from scalecast.auxmodels import auto_arima
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
plt.show()
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
acf_plot = plot_acf(df.Num_Passengers, lags=20)
```

```python
pacf_plot = plot_pacf(df.Num_Passengers, method='ywm')
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
    print(dfoutput)
    is_not = 'is' if dfoutput[1]<0.05 else 'is not'
    print(f"Time series {is_not} stationary")
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

<!-- #region tags=[] -->
## Remove trend
<!-- #endregion -->

```python
first_dif = df.diff()[1:]
```

```python
plt.figure(figsize=(9,3))
plt.title("First Difference")
sns.lineplot(x="Month", y="Num_Passengers", data=first_dif, )
plt.show()
```

```python
adf_test(first_dif)
```

### Second option

```python
y_detrend = ((df - df.rolling(window=12).mean())/df.rolling(window=12).std())[11:]
plt.figure(figsize=(9,3))
plt.title("Detrend")
sns.lineplot(x="Month", y="Num_Passengers", data=y_detrend, )
plt.show()
```

```python
adf_test(y_detrend)
```

<!-- #region tags=[] -->
## Remove Seasonality
<!-- #endregion -->

```python
yearly_dif = first_dif.diff(12)[12:]
```

```python
plt.figure(figsize=(9,3))
plt.title("Yearly Difference")
sns.lineplot(x="Month", y="Num_Passengers",
             data=yearly_dif)
plt.show()
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
pacf_plot = plot_pacf(yearly_dif.Num_Passengers, method='ywm')
```

# Comparing different models

<!-- #region tags=[] -->
## Utils
<!-- #endregion -->

<!-- #region tags=[] -->
### Rolling forecast utility functions
<!-- #endregion -->

```python
def predict_previous_value(dataframe, split):
    return dataframe[split + relativedelta(months=-1): ][:-1].shift(1, freq='M')
```

```python
def exponentialSmoothing(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    for i, cur_date in enumerate(pd.date_range(split,  dataframe.index[-1], freq='MS')):
        if(verbose): print(i, " iteration")
        model = ExponentialSmoothing(dataframe[:cur_date - timedelta(days=1)], seasonal_periods = 12, trend='add', seasonal='add')
        fit = model.fit()
        pred = fit.predict(start=cur_date, end=cur_date) 
        
        model_fit.append(fit)
        predictions.append(pred)
    
    predictions_df = pd.DataFrame(pd.concat(predictions, axis=0), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
def linear_regression_with_season(x, y):
    if len(y)<12:
        raise ValueError("This prediction require more then year of test data")
    model = LinearRegression()
    fit = model.fit(x, y)
    lin_pred = fit.predict([x[-1,:]+1])
    pred = lin_pred[0][0] + (y[-12][0] - fit.predict([x[-12,:]])[0][0])
    return fit, pred
```

```python
def fit_linear_regression_with_season(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        data = dataframe[:cur_date - timedelta(days=1)]['Num_Passengers'].values.reshape(-1, 1)
        index = np.arange(len(data)).reshape(-1, 1)
        fit, pred = linear_regression_with_season(index, data)
        
        model_fit.append(fit)
        predictions.append(pred)
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
def linear_regression(x, y, predict):
    model = LinearRegression()
    fit = model.fit(x, y)
    pred = fit.predict(predict)[0]
    return fit, pred
```

```python
def fit_linear_regression_with_12_lag(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data = train_data[12:]
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data = train_data[1:]
        
        train_col = ["lag_1", "lag_12"]
        fit, pred = linear_regression(train_data[train_col].iloc[:-1, :], train_data["Num_Passengers"].iloc[:-1], 
                                     train_data[train_col].iloc[-1:, :])
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
def fit_linear_regression_with_diff_of_12_lag(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data = train_data[12:]
        train_data["lag_1"] =train_data["Num_Passengers"].shift(1)
        train_data = train_data[1:]
        train_data["diff_lag"] =  train_data["lag_12"]-train_data["lag_1"] 
       
        train_col = ["lag_1", "diff_lag"]
        fit, pred = linear_regression(train_data[train_col].iloc[:-1, :], train_data["Num_Passengers"].iloc[:-1], 
                                     train_data[train_col].iloc[-1:, :])
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
def fit_linear_regression_with_lag_of_12_diff(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data = train_data[1:]
        
        train_data["diff"] = train_data["Num_Passengers"].diff(1)
        train_data = train_data[1:]
        
        train_data["lag_diff_12"] = train_data["diff"].shift(12)
        train_data = train_data[12:]
       
        train_col = ["lag_1", "lag_diff_12"]
        fit, pred = linear_regression(train_data[train_col].iloc[:-1, :], train_data["Num_Passengers"].iloc[:-1], 
                                     train_data[train_col].iloc[-1:, :])
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
def fit_model_arima(order, dataframe, split, method=None, verbose=False):
    model_fit = []
    predictions =  []
    
    for i, cur_date in enumerate(pd.date_range(split,  dataframe.index[-1], freq='MS')):
        if(verbose): print(i, " iteration")
        model = ARIMA(dataframe[:cur_date - timedelta(days=1)], order = order)
        fit = model.fit(method=method)
        pred = fit.predict(start=cur_date, end=cur_date) 
        
        model_fit.append(fit)
        predictions.append(pred)
    
    predictions_df = pd.DataFrame(pd.concat(predictions, axis=0), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
def fit_model_sarimax(order, seasonal_order, dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    
    for i, cur_date in enumerate(pd.date_range(split,  dataframe.index[-1], freq='MS')):
        if(verbose): print(i, " iteration")
        model = SARIMAX(dataframe[:cur_date - timedelta(days=1)], order=order, seasonal_order=seasonal_order)
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

```python
def split_train_test(df_to_split, train_percents):
    split_date = get_split_date(df_to_split, train_percents)

    train_data = df_to_split[:split_date - timedelta(days=1)]
    test_data = df_to_split[split_date:]
    return train_data, test_data, split_date
```

```python
def split_train_test_by_date(df_to_split, split_date):
    train_data = df_to_split[:split_date - timedelta(days=1)]
    test_data = df_to_split[split_date:]
    return train_data, test_data
```

```python
def get_metrics_for_df(test_data, predictions, col='Num_Passengers'):
    return get_metrics(test_data[col], predictions[col])
```

```python
def get_metrics(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)  
    rmse = mean_squared_error(test_data, predictions, squared=False)          
    mae = mean_absolute_error(test_data, predictions)
    return mse, rmse, mae
```

```python
def print_metrics(mse, rmse, mae):
    print("MSE - ", mse)
    print("RMSE - ", rmse)
    print("MAE - ", mae)
```

```python
def plot_test_with_predictions(test, pred, name):
    plt.figure(figsize=(8,4))

    plt.plot(test)
    plt.plot(pred)
    plt.title(name)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.show()
```

```python
def inverse_diffirence(observation, diffs, periods):
    restored = observation.copy()
    restored.iloc[periods+1:] = np.nan
    for d, val in diffs.iterrows():
        restored.loc[d] = restored.loc[d - pd.DateOffset(months=periods)] + val
    return restored.iloc[periods:]
```

<!-- #region tags=[] -->
## Baseline
<!-- #endregion -->

<!-- #region tags=[] -->
### Predict previous value
<!-- #endregion -->

```python
_, b_test_data, split_date = split_train_test(df, 0.70)
b_predictons = predict_previous_value(df, split_date)
plot_test_with_predictions(b_test_data, b_predictons, "Prediction is last known value")

b_mse, b_rmse, b_mae = get_metrics(b_test_data.values, b_predictons)
print_metrics(b_mse, b_rmse, b_mae)
```

<!-- #region tags=[] -->
### Linear regression with seasonal residual
<!-- #endregion -->

```python
_, lr_test_data, lr_split_date = split_train_test(df, 0.70)
lr_models, lr_predictons = fit_linear_regression_with_season(df, lr_split_date)
plot_test_with_predictions(lr_test_data, lr_predictons, "Linear regression with lag residual")

lr_mse, lr_rmse, lr_mae = get_metrics(lr_test_data.values, lr_predictons)
print_metrics(lr_mse, lr_rmse, lr_mae)
```

## Linear regression a1 *xt + a2*xt-12 +b

```python
_, b1_test_data, b1_split_date = split_train_test(df, 0.70)
_,b1_predictons = fit_linear_regression_with_12_lag(df, b1_split_date)
plot_test_with_predictions(b1_test_data, b1_predictons, "Linear regression a1 *xt + a2*xt-12 +b")

b1_mse, b1_rmse, b1_mae = get_metrics(b1_test_data.values, b1_predictons)
print_metrics(b1_mse, b1_rmse, b1_mae)
```

## Linear regression  a1 *xt + a2*(xt-12 - xt) +b

```python
_, b2_test_data, b2_split_date = split_train_test(df, 0.70)
_,b2_predictons = fit_linear_regression_with_diff_of_12_lag(df, b2_split_date)
plot_test_with_predictions(b2_test_data, b2_predictons, "Linear regression  a1 *xt + a2*(xt-12 - xt) +b")

b2_mse, b2_rmse, b2_mae = get_metrics(b2_test_data.values, b2_predictons)
print_metrics(b2_mse, b2_rmse, b2_mae)
```

## Linear regression a1 *xt + a2*(xt-11 - xt-12) +b

```python
_, b3_test_data, b3_split_date = split_train_test(df, 0.70)
_, b3_predictons = fit_linear_regression_with_lag_of_12_diff(df, b3_split_date)
plot_test_with_predictions(b3_test_data, b3_predictons, "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")

b3_mse, b3_rmse, b3_mae = get_metrics(b3_test_data.values, b3_predictons)
print_metrics(b3_mse, b3_rmse, b3_mae)
```

<!-- #region tags=[] -->
## ExponentialSmoothing
<!-- #endregion -->

```python
_, es_test_data, es_split_date = split_train_test(df, 0.70)
es_models, es_predictons = exponentialSmoothing(df, es_split_date)
plot_test_with_predictions(es_test_data, es_predictons, "ExponentialSmoothing")

es_mse, es_rmse, es_mae = get_metrics(es_test_data.values, es_predictons)
print_metrics(es_mse, es_rmse, es_mae)
```

```python
es_models[-1].summary()
```

<!-- #region tags=[] -->
## AR model
<!-- #endregion -->

```python
train_data, test_data, _ = split_train_test(df, 0.95)
```

```python
ar_model = ARIMA(train_data, order=(2,1,0))
ar_fit = ar_model.fit()
```

```python
print(ar_fit.summary())
```

```python
predictions = ar_fit.predict(start=test_data.index[0], end=test_data.index[-1])
plot_test_with_predictions(test_data, predictions, "Predict whole test data with AR model")
```

### Predict one value then fit model again

```python
_, ar_test_data, ar_split = split_train_test(df, 0.70)
ar_models, ar_predictions = fit_model_arima((2,1, 0), df, ar_split)
plot_test_with_predictions(ar_test_data, ar_predictions, "Predict with AR model")

ar_mse, ar_rmse, ar_mae = get_metrics(ar_test_data.values, ar_predictions)
print_metrics(ar_mse, ar_rmse, ar_mae)
```

```python
ar_models[-1].summary()
```

<!-- #region tags=[] -->
## MA model
<!-- #endregion -->

```python
_, ma_test_data, ma_split = split_train_test(df, 0.70)
ma_models, ma_predictions = fit_model_arima((0,1,2), df, ma_split)
plot_test_with_predictions(ma_test_data, ma_predictions, "Predict with MA model")

ma_mse, ma_rmse, ma_mae = get_metrics(ma_test_data.values, ma_predictions)
print_metrics(ma_mse, ma_rmse, ma_mae)
```

```python
ma_models[-1].summary()
```

<!-- #region tags=[] -->
## ARIMA model
<!-- #endregion -->

```python
i_train, i_test_data, i_split = split_train_test(df, 0.70)
i_models, i_predictions = fit_model_arima((2,1,1), df, i_split)
plot_test_with_predictions(i_test_data, i_predictions, "Predict with ARIMA model")
i_mse, i_rmse, i_mae = get_metrics(i_test_data.values, i_predictions)
print_metrics(i_mse, i_rmse, i_mae)
```

```python
i_models[-1].summary()
```

<!-- #region tags=[] -->
# SARIMA
<!-- #endregion -->

```python
_, si_test_data, si_split = split_train_test(df, 0.70)
si_models, si_predictions = fit_model_sarimax((1, 1, 1), (1,1,1,12) , df, si_split)
plot_test_with_predictions(si_test_data, si_predictions, "Predict with SARIMA model")

si_mse, si_rmse, si_mae = get_metrics(si_test_data.values, si_predictions)
print_metrics(si_mse, si_rmse, si_mae)
```

# Compare models

```python
models_metric = pd.DataFrame({"MSE":[b_mse, b1_mse, b2_mse, b3_mse, lr_mse, es_mse, ar_mse, ma_mse, i_mse, si_mse],
                            "RMSE": [b_rmse, b1_rmse, b2_rmse, b3_rmse, lr_rmse, es_rmse, ar_rmse, ma_rmse, i_rmse, si_rmse],
                            "MAE": [b_mae, b1_mae, b2_mae, b3_mae, lr_mae, es_mae, ar_mae, ma_mae, i_mae, si_mae],
                             "Model":["Last_value", "Linear_regression v0", "Linear_regression v1", "Linear_regression v2" , "Linear_regression v3" , "Exponential soothing", "AR", "MA",  "ARIMA", "SARIMA" ]
                             })
```

```python
models_metric
```

```python
plt.title("Compare models by MSE")
sns.barplot(data=models_metric, x="MSE", y="Model")
plt.show()
```

```python
plt.title("Compare models by RMSE")
sns.barplot(data=models_metric, x="RMSE", y="Model")
plt.show()
```

```python
plt.title("Compare models by MAE")
sns.barplot(data=models_metric, x="MAE", y="Model")
plt.show()
```

<!-- #region tags=[] -->
# Scalecast
<!-- #endregion -->

```python
f = Forecaster(y=df['Num_Passengers'],current_dates=df.index)
```

```python
f.plot_acf()
plt.show()
f.plot_pacf()
plt.show()
```

```python
f.seasonal_decompose().plot()
plt.show()
```

```python
f.generate_future_dates(12)
f.set_test_length(.2) # 20% test set
f.set_estimator('arima')
f.manual_forecast(order=(0,1,1),seasonal_order=(1,0,1,12), call_me='arima1')
```

```python
f.plot_test_set(ci=True,models='arima1')
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()
```

```python
f.regr.summary()
```

<!-- #region tags=[] -->
# Pmdarima
<!-- #endregion -->

```python
auto_arima(
    f,
    start_P=1,
    start_q=1,
    max_p=6,
    max_q=6,
    m=20,
    seasonal=True,
    max_P=2, 
    max_D=2,
    max_Q=2,
    max_d=2,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion="bic", #{'bic', 'hqic', 'oob', 'aicc', 'aic'}
    alpha=0.05,
    scoring='mse',
    call_me='arima3',
)

f.plot_test_set(ci=True,models='arima3')
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()
```

```python
f.regr.summary()
```

```python

```
