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
import plotly.express as px
import matplotlib.pyplot as plt

from time import time
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import timedelta

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from scalecast.Forecaster import Forecaster
from scalecast.auxmodels import auto_arima
import pmdarima as pm

import warnings; warnings.simplefilter('ignore')
```

<!-- #region tags=[] -->
# Read and view data
<!-- #endregion -->

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
df.info()
```

```python
df.describe()
```

```python
plt.figure(figsize=(10,6))
plt.title("Amount of air passagers by month")
sns.lineplot(x="Month", y="Num_Passengers", data=df)
plt.show()
```

```python
piv_index = getattr(df.index, 'month')
piv_season = getattr(df.index, 'year')
    
piv = pd.pivot_table(df, index=piv_index, columns=piv_season, values='Num_Passengers')
ax =piv.plot(figsize=(12,8), title="Amount of air passagers by month.Seasonal plot")
ax.set_xticks(range(1,13))
plt.show()
```

```python
df_circular_season = df.copy()
df_circular_season['month'] = getattr(df.index, 'month')
df_circular_season['month_map'] = getattr(df.index, 'month')*(380/12)
df_circular_season['year'] = getattr(df.index, 'year')
fig = px.line_polar(df_circular_season, r="Num_Passengers", theta="month_map", color="year", line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r)
fig.show()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true tags=[] -->
# ACF & PACF
<!-- #endregion -->

```python
acf_plot = plot_acf(df.Num_Passengers, lags=40)
```

```python
pacf_plot = plot_pacf(df.Num_Passengers, method='ywm')
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# STL Decompozition
<!-- #endregion -->

```python
stl = STL(df)
result = stl.fit()
result.plot()
plt.show()
```

```python
stl_decompos_results = [result.seasonal, result.trend, result.resid]
stl_titles = ["Seasonal", "Trend", "Residual"]
plot_origin = [1,1,0]

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("STL decomposition", fontsize=18, y=0.95)

for data, name, ax, plot_o in zip(stl_decompos_results,stl_titles , axs.ravel(), plot_origin):
    data.plot(ax=ax)
    if plot_o:
        df.plot(ax=ax, style='g--')
    ax.set_title(name)
    ax.set_xlabel("")
    if legend:=ax.get_legend():
        legend.remove()

plt.tight_layout()
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# Stationarity
<!-- #endregion -->

```python
line = 28
param_names =[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ]

def adf_test(timeseries, df_name = ""):
    print(df_name, " Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    for name, value in zip(param_names, dftest[0:4]):
        print(name.ljust(line), value)
    is_not = 'is' if dftest[1]<0.05 else 'is not'
    print(f"Time series {is_not} stationary")
```

```python
adf_test(df, "Original dataset")
```

```python
ax = df.plot()
df.rolling(window = 12).mean().plot(figsize=(8,4), color="tab:red", title="Rolling Mean over 12 month period",
                                    ax=ax, legend=False)
plt.show()
```

```python
df.rolling(window = 12).var().plot(figsize=(8,4), color="tab:red", title="Rolling Variance over 12 month period", legend=False)
plt.show()
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
sns.lineplot(x="Month", y="Num_Passengers", data=first_dif)
plt.show()
adf_test(first_dif, "First Difference")
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
sns.lineplot(x="Month", y="Num_Passengers", data=yearly_dif)
plt.show()
adf_test(yearly_dif, "Yearly Difference")
```

<!-- #region tags=[] -->
## ACF & PACF
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Split
<!-- #endregion -->

```python
def get_split_date(data, split_coef):
    assert 0 < split_coef < 1 , "Split coefficient should be in range [0,1]"

    split = math.floor(len(data) * split_coef)
    split = split if split < len(data) else split-1
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
# get all train dates before test or first date in test
def get_train_data(df, date):
    date_split =  date[0] if isinstance(date, list) else date
    return df[:date_split - timedelta(days=1)]
```

```python
def chunks(l, n):
    for i in range(0, len(l)//n):
        s=i*n
        yield l[s:s+n]
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Metrics and plots
<!-- #endregion -->
```python
def get_metrics_for_df(test_data, predictions, col='Num_Passengers'):
    return get_metrics(test_data[col], predictions[col])
```

```python
def get_metrics(test_all, predictions_all):
    size = min(len(test_all), len(predictions_all))
    test_data = test_all[:size]
    predictions = predictions_all[:size]
    mse = mean_squared_error(test_data, predictions)  
    rmse = mean_squared_error(test_data, predictions, squared=False)          
    mae = mean_absolute_error(test_data, predictions)
    mape = mean_absolute_percentage_error(test_data, predictions)
    return mse, rmse, mae, mape
```

```python
METRICS_NAMES = ["MSE", "RMSE", "MAE", "MAPE"]
def print_metrics(metrics):
    for name, value in zip(METRICS_NAMES, metrics):
        print(f"{name} - {value}")
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
def model_stats(model_fit):
    print(model_fit.summary())
     
    fig, axs = plt.subplots(nrows=2, ncols=1)
    plt.subplots_adjust(hspace=0.7)
    
    model_fit.resid.plot(ax=axs[0], title = "Model residuals")
    model_fit.resid.plot(kind='kde', ax=axs[1], title = "Last model residuals distribution")
    plt.show()
```

<!-- #region tags=[] -->
### Fit common functions
<!-- #endregion -->

```python tags=[]
def linear_regression(x, y, predict):
    model = LinearRegression()
    fit = model.fit(x, y)
    pred = fit.predict(predict)
    return fit, pred
```

```python
def fit_model_arima(order, dataframe, split, method=None, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        y_train = get_train_data(dataframe, test_dates)
        model = ARIMA(y_train, order = order)
        fit = model.fit(method=method)
        pred = fit.predict(start=test_dates[0], end=test_dates[-1]) 
        pred.name = "Num_Passengers"
        model_fit.append(fit)
        predictions.append(pred) 
    predictions_df = pd.DataFrame(pd.concat(predictions), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python tags=[]
def fit_model_arima_all_test(order, test, train, method=None):
    model = ARIMA(test, order = order)
    fit = model.fit(method=method)
    pred = fit.predict(start=train.index[0], end=train.index[-1]) 
    return fit, pred
```

```python
def fit_model_sarimax(order, seasonal_order, dataframe, split, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        y_train = get_train_data(dataframe, test_dates)
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
        fit = model.fit()
        pred = fit.predict(start=test_dates[0], end=test_dates[-1]) 
        pred.name = "Num_Passengers"
        model_fit.append(fit)
        predictions.append(pred) 
    predictions_df = pd.DataFrame(pd.concat(predictions), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
def fit_model_sarimax_all_test(order, seasonal_order,test, train):
    model = SARIMAX(test, order=order, seasonal_order=seasonal_order)
    fit = model.fit()
    pred = fit.predict(start=train.index[0], end=train.index[-1]) 

    return fit, pred
```

## Dataset with metrics results

```python
models_metrics = pd.DataFrame(columns = METRICS_NAMES + ["TestName", "Model"])
```

```python
def save_metrics(models_metrics, model_name, test_name, metrics):
    models_metrics = models_metrics.drop(models_metrics[(models_metrics.TestName == test_name) & (models_metrics.Model == model_name)].index)
    
    metrics_dict = dict(zip(METRICS_NAMES, metrics))
    metrics_dict["TestName"] = test_name
    metrics_dict["Model"] = model_name
    new_row = pd.Series(metrics_dict)
    return  pd.concat([models_metrics, new_row.to_frame().T])
```

```python
def compare_model_tests(model_name):
    one_model_metrics = models_metrics[models_metrics["Model"] == model_name]
    axes = one_model_metrics.plot.bar(x="TestName",subplots=True, figsize=(10,15), title = model_name)
    for ax in axes:
        ax.get_legend().remove()
        ax.tick_params(axis='x', labelrotation=0, labeltop=True, labelbottom=False)
    
    plt.show()
```

<!-- #region tags=[] -->
## Baseline
<!-- #endregion -->

```python
test_size_precent = 0.67
```

<!-- #region tags=[] -->
### Predict previous value
<!-- #endregion -->

```python
previous_model_name = "Last_known_value"
```

<!-- #region tags=[] -->
#### Predict all test at once
<!-- #endregion -->

```python
def predict_previous_value_for_test(train, test):
    predictions = pd.DataFrame().reindex_like(test)
    predictions['Num_Passengers'] = train['Num_Passengers'][-1]
    return predictions
```

```python
train_data, test_data, _ = split_train_test(df,test_size_precent)
predictons = predict_previous_value_for_test(train_data, test_data)
plot_test_with_predictions(test_data, predictons, "Prediction is last known value all test")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, previous_model_name, "Predict_all_test", metrics)
```

<!-- #region tags=[] -->
#### Rolling presitions 12 months
<!-- #endregion -->

```python
def predict_previous_value_rolling_12(dataframe, split):
    predictions = pd.DataFrame()
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))

    for test_split in chunks(prediction_range, 12):
        train = get_train_data(dataframe, test_split) 
        test = dataframe[dataframe.index.isin(test_split)]
        prediction = predict_previous_value_for_test(train,test)
        predictions = pd.concat([predictions, prediction])
    return predictions
```

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
predictons = predict_previous_value_rolling_12(df, split_date)
plot_test_with_predictions(test_data, predictons, "Prediction is last known value")
metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, previous_model_name, "Rolling_prediction_12", metrics)
```

<!-- #region tags=[] -->
#### Rolling prediction
<!-- #endregion -->

```python
def predict_previous_value_rolling(dataframe, split):
    return dataframe[split + relativedelta(months=-1): ][:-1].shift(1, freq='M')
```

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
predictons = predict_previous_value_rolling(df, split_date)
plot_test_with_predictions(test_data, predictons, "Prediction is last known value")
metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, previous_model_name, "Rolling_prediction_1", metrics)
```

<!-- #region tags=[] -->
#### Comparision
<!-- #endregion -->

```python
compare_model_tests(previous_model_name)
```

<!-- #region tags=[] -->
### Linear regression with seasonal residual
<!-- #endregion -->

```python
lin_reg_res_name = "Lin_reg_lag_residual"
```

```python tags=[]
def linear_regression_with_season(x, y, test, predict):
    model = LinearRegression()
    fit = model.fit(x, y)
    predict_test = fit.predict(test)
    predictions = fit.predict(predict)
    for i in range(len(predictions)):
        ind_p = i-12
        predictions[i]+= (y[ind_p] - predict_test[ind_p])
    return fit, predictions
```

#### Rolling predictions

```python
def fit_linear_regression_with_lag_residual(dataframe, split, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        y_train = get_train_data(dataframe, test_dates).Num_Passengers.values.reshape(-1, 1)
        train_size = len(y_train)
        assert train_size>12, "Not enough data for training"
        x_train = np.arange(train_size).reshape(-1, 1)
        predict_start = len(y_train)+1
        predict = np.arange(predict_start, predict_start + len(test_dates)).reshape(-1, 1)
        test = np.concatenate((x_train[-12:], predict))
        
        fit, pred = linear_regression_with_season(x_train[:-12], y_train[:-12], test, predict)
        
        model_fit.append(fit)
        predictions.extend(pred)
    predictions_data = np.concatenate(predictions)
    predictions_df = pd.DataFrame({'Month': prediction_range[:len(predictions_data)], 'Num_Passengers':predictions_data })
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
models, predictons = fit_linear_regression_with_lag_residual(df, split_date)
plot_test_with_predictions(test_data, predictons, "Linear regression with lag residual")
metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_res_name, "Rolling_prediction_1", metrics)
```

<!-- #region tags=[] -->
#### Rolling predictions 12
<!-- #endregion -->

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
models, predictons = fit_linear_regression_with_lag_residual(df, split_date, batch=12)
plot_test_with_predictions(test_data, predictons, "Linear regression with lag residual")
metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_res_name, "Rolling_prediction_12", metrics)
```

<!-- #region tags=[] -->
#### Predict all test at once
<!-- #endregion -->

```python
# Can't be used as algorithm requires to use actual data a year ago
```

#### Comparision

```python
compare_model_tests(lin_reg_res_name)
```

### Linear regression a1 *xt + a2*xt-12 +b

```python
lin_reg_1_name = "Lin_reg_a1*xt+a2*xt-12+b"
```

#### Rolling predictions

```python
def fit_linear_regression_with_12_lag(dataframe, split, batch = 1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        train_data = get_train_data(dataframe, test_dates).reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data.dropna(inplace=True)
        
        train_col = ["lag_1", "lag_12"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"],
                                                       test_size = len(test_dates), shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
        model_fit.append(fit)
        predictions.extend(pred) 
    predictions_df = pd.DataFrame({'Month': prediction_range[:len(predictions)], 'Num_Passengers':  predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_12_lag(df, split_date)
plot_test_with_predictions(test_data, predictons, "Linear regression a1*xt + a2*xt-12 +b")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_1_name, "Rolling_prediction_1", metrics)
```

#### Rolling predictions 12

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_12_lag(df, split_date, batch = 12 )
plot_test_with_predictions(test_data, predictons, "Linear regression a1*xt + a2*xt-12 +b")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_1_name, "Rolling_prediction_12", metrics)
```

#### Predict all test at once

```python
def linear_regression_with_12_lag_data(df):
    lr_df = pd.DataFrame()
    lr_df['Num_Passengers_1m_ago'] = df.Num_Passengers.shift(1)
    lr_df['Num_Passengers_12m_ago'] = df.Num_Passengers.shift(12)
    lr_df['Num_Passengers']=df.Num_Passengers
    lr_df= lr_df.dropna()
    x = lr_df.loc[:, ['Num_Passengers_1m_ago', 'Num_Passengers_12m_ago']].values

    y = lr_df.loc[:, ['Num_Passengers']].values
    return x,y, lr_df.index
```

```python
def fit_linear_regression_with_12_lag_all_test(lr_train_data, lr_test_data):
    x_train, y_train, _ = linear_regression_with_12_lag_data(lr_train_data)
    reg = LinearRegression().fit(x_train, y_train)
    x_test, y_test, test_index = linear_regression_with_12_lag_data(lr_test_data)

    pred_test = reg.predict(x_test)
    predictions_df = pd.DataFrame({'Month': test_index, 'Num_Passengers': pred_test.flatten()})
    predictions_df = predictions_df.set_index('Month')
    
    return reg, predictions_df
```

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_12_lag_all_test(train_data, test_data)
plot_test_with_predictions(test_data.loc[predictons.index], predictons, "Linear regression a1*xt + a2*xt-12 +b")

metrics = list(get_metrics(test_data.loc[predictons.index].values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_1_name, "Predict_all_test", metrics)
```

#### Comparision

```python
compare_model_tests(lin_reg_1_name)
```

### Linear regression  a1*xt + a2*(xt-12 - xt) +b

```python
lin_reg_2_name = "Lin_reg_a1*xt+a2*(xt-12-xt)+b"
```

<!-- #region tags=[] -->
#### Rolling predictions
<!-- #endregion -->

```python
def fit_linear_regression_with_diff_of_12_lag(dataframe, split, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        train_data = get_train_data(dataframe, test_dates).reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data["lag_1"] =train_data["Num_Passengers"].shift(1)
        train_data.dropna(inplace=True)
        train_data["diff_lag"] =  train_data["lag_12"] - train_data["lag_1"] 
       
        train_col = ["lag_1", "diff_lag"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"], 
                                                       test_size = len(test_dates), shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
       
        model_fit.append(fit)
        predictions.extend(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range[:len(predictions)], 'Num_Passengers':  predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_diff_of_12_lag(df, split_date)
plot_test_with_predictions(test_data, predictons, "Linear regression  a1*xt + a2*(xt-12 - xt) +b")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_2_name, "Rolling_prediction_1", metrics)
```

#### Rolling predictions 12

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_diff_of_12_lag(df, split_date, batch=12)
plot_test_with_predictions(test_data, predictons, "Linear regression  a1*xt + a2*(xt-12 - xt) +b")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_2_name, "Rolling_prediction_12", metrics)
```

#### Predict all test at once

```python
def linear_regression_with_diff_of_12_data(df):
    lr_df = pd.DataFrame()
    lr_df["Num_Passengers_1m_ago"] = df.Num_Passengers.shift(1)
    lr_df["Num_Passengers_12m_ago"] = df.Num_Passengers.shift(12)
    lr_df['Num_Passengers']=df.Num_Passengers
    lr_df= lr_df.dropna()
    lr_df["Num_Passengers_diff"] = lr_df["Num_Passengers_12m_ago"] - lr_df["Num_Passengers_1m_ago"]
    
    x = lr_df.loc[:, ['Num_Passengers_1m_ago', 'Num_Passengers_diff']].values
    y = lr_df.loc[:, ['Num_Passengers']].values
    return x,y, lr_df.index
```

```python
def fit_linear_regression_with_diff_of_12_lag_all_test(lr_train_data, lr_test_data):
    x_train, y_train, _ = linear_regression_with_diff_of_12_data(lr_train_data)
    reg = LinearRegression().fit(x_train, y_train)
    x_test, y_test, test_index = linear_regression_with_diff_of_12_data(lr_test_data)

    pred_test = reg.predict(x_test)
    predictions_df = pd.DataFrame({'Month': test_index,
                   'Num_Passengers': pred_test.flatten()})
    predictions_df = predictions_df.set_index('Month')
    
    return reg, predictions_df
```

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
_,predictons = fit_linear_regression_with_diff_of_12_lag_all_test(train_data,test_data)
plot_test_with_predictions(test_data.loc[predictons.index], predictons, "Linear regression  a1 *xt + a2*(xt-12 - xt) +b")

metrics = list(get_metrics(test_data.loc[predictons.index].values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_2_name, "Predict_all_test", metrics)
```

#### Comparision

```python
compare_model_tests(lin_reg_2_name)
```

<!-- #region tags=[] -->
### Linear regression a1 *xt + a2*(xt-11 - xt-12) +b
<!-- #endregion -->

```python
lin_reg_3_name = "Lin_reg_a1*xt+a2*(xt-11-xt-12)+b"
```

#### Rolling predictions

```python
def fit_linear_regression_with_lag_of_12_diff(dataframe, split, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        train_data = get_train_data(dataframe, test_dates).reset_index(drop=True)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data["lag_diff_11"] = train_data["Num_Passengers"].shift(11)
        train_data["lag_diff_12"] = train_data["Num_Passengers"].shift(12)
        train_data.dropna(inplace=True)
        train_data["diff_lag"] =  train_data["lag_diff_11"] - train_data["lag_diff_12"] 
       
        train_col = ["lag_1", "lag_diff_12"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"],
                                                       test_size = len(test_dates), shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
        
        model_fit.append(fit)
        predictions.extend(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range[:len(predictions)], 'Num_Passengers':  predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_lag_of_12_diff(df, split_date)
plot_test_with_predictions(test_data, predictons, "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")


metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_3_name, "Rolling_prediction_1", metrics)
```

#### Rolling predictions 12

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_lag_of_12_diff(df, split_date, batch=12)
plot_test_with_predictions(test_data, predictons, "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")


metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_3_name, "Rolling_prediction_12", metrics)
```

#### Predict all test at once

```python
def linear_regression_with_lag_of_12_diff_data(df):
    lr_df = pd.DataFrame()
    lr_df["Num_Passengers_1m_ago"] = df.Num_Passengers.shift(1)
    lr_df["Num_Passengers_11m_ago"] = df.Num_Passengers.shift(11)
    lr_df["Num_Passengers_12m_ago"] = df.Num_Passengers.shift(12)
    lr_df['Num_Passengers']=df.Num_Passengers
    lr_df= lr_df.dropna()
    lr_df["Num_Passengers_diff"] = lr_df["Num_Passengers_11m_ago"] - lr_df["Num_Passengers_12m_ago"]
    
    x = lr_df.loc[:, ['Num_Passengers_1m_ago', 'Num_Passengers_diff']].values
    y = lr_df.loc[:, ['Num_Passengers']].values
    return x,y, lr_df.index
```

```python
def fit_linear_regression_with_lag_of_12_diff_all_test(lr_train_data, lr_test_data):
    x_train, y_train, _ = linear_regression_with_diff_of_12_data(lr_train_data)
    reg = LinearRegression().fit(x_train, y_train)
    x_test, y_test, test_index = linear_regression_with_diff_of_12_data(lr_test_data)

    pred_test = reg.predict(x_test)
    predictions_df = pd.DataFrame({'Month': test_index,
                   'Num_Passengers': pred_test.flatten()})
    predictions_df = predictions_df.set_index('Month')
    
    return reg, predictions_df
```

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
_, predictons = fit_linear_regression_with_lag_of_12_diff_all_test(train_data, test_data)
plot_test_with_predictions(test_data.loc[predictons.index],predictons,   "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")

metrics = list(get_metrics(test_data.loc[predictons.index].values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, lin_reg_3_name, "Predict_all_test", metrics)
```

#### Comparision

```python
compare_model_tests(lin_reg_3_name)
```

<!-- #region tags=[] -->
## ExponentialSmoothing
<!-- #endregion -->

```python
es_model_name = "Exponential_Smoothing"
```

#### Rolling predictions

```python
def exponentialSmoothing(dataframe, split, batch=1):
    model_fit = []
    predictions =  []
    prediction_range = list(pd.date_range(split, dataframe.index[-1], freq='MS'))
    
    for test_dates in chunks(prediction_range, batch):
        train_data = get_train_data(dataframe, test_dates)
        model = ExponentialSmoothing(train_data, seasonal_periods = 12, trend='add', seasonal='add')
        fit = model.fit()
        pred = fit.predict(start=test_dates[0], end=test_dates[-1]) 
        model_fit.append(fit)
        predictions.append(pred)
    predictions_df = pd.DataFrame(pd.concat(predictions), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
models, predictons = exponentialSmoothing(df, split_date)
plot_test_with_predictions(test_data, predictons, "ExponentialSmoothing")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, es_model_name, "Rolling_prediction_1", metrics)
```

```python
models[-1].summary()
```

#### Rolling predictions 12

```python
_, test_data, split_date = split_train_test(df, test_size_precent)
models, predictons = exponentialSmoothing(df, split_date,batch=12 )
plot_test_with_predictions(test_data, predictons, "ExponentialSmoothing")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, es_model_name, "Rolling_prediction_12", metrics)
```

#### Predict all test at once

```python
def exponentialSmoothing_all_test(train, test):
    model = ExponentialSmoothing(train, seasonal_periods = 12, trend='add', seasonal='add')
    fit = model.fit()
    pred = fit.predict(start=test.index[0], end=test.index[-1]) 
    return fit, pred
```

```python
train_data, test_data, split_date = split_train_test(df, test_size_precent)
models, predictons = exponentialSmoothing_all_test(train_data, test_data)
plot_test_with_predictions(test_data, predictons, "ExponentialSmoothing")

metrics = list(get_metrics(test_data.values, predictons))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, es_model_name, "Predict_all_test", metrics)
```

#### Comparision

```python
compare_model_tests(es_model_name)
```

<!-- #region tags=[] -->
## AR model
<!-- #endregion -->

```python
ar_model_name = "AR"
```

```python
pm.auto_arima(df.Num_Passengers,
    start_p=1,
    start_q=0,
    max_p=26,
    max_q=0,
    max_order = None,
    maxiter = 500,
    seasonal=False,
    trace=True,
    error_action='ignore',
    stepwise =False,
    scoring='mse',
    information_criterion ='aic',
    method = 'nm')
```

### Predict all test at once

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
```

```python
ar_model = ARIMA(train_data, order=(14,1,0))
ar_fit = ar_model.fit()
```

```python
print(ar_fit.summary())
```

```python
predictions = ar_fit.predict(start=test_data.index[0], end=test_data.index[-1])
plot_test_with_predictions(test_data, predictions, "Predict whole test data with AR model")

metrics = list(get_metrics(test_data, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ar_model_name, "Predict_all_test", metrics)
```

### Rolling predictions

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((14,1,0), df, split, method = 'statespace')
plot_test_with_predictions(test_data, predictions, "Predict with AR model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ar_model_name, "Rolling_prediction_1", metrics)
```

```python
model_stats(models[-1])
```

### Rolling predictions 12

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((14,1,0), df, split)
plot_test_with_predictions(test_data, predictions, "Predict with AR model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ar_model_name, "Rolling_prediction_12", metrics)
```

### Comparision 

```python
compare_model_tests(ar_model_name)
```

<!-- #region tags=[] -->
## MA model
<!-- #endregion -->

```python
ma_model_name = "MA"
```

```python
pm.auto_arima(df.Num_Passengers,
    start_p=0,
    start_q=1,
    max_p=0,
    max_q=20,
    max_order = None,
    maxiter = 500,
    seasonal=False,
    trace=True,
    error_action='ignore',
    stepwise =False,
    scoring='mse',
    information_criterion ='aic',
    method = 'bfgs')
```

### Rolling predictions

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((0,1,12), df, split)
plot_test_with_predictions(test_data, predictions, "Predict with MA model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ma_model_name, "Rolling_prediction_1", metrics)
```

```python
models[-1].summary()
```

<!-- #region tags=[] -->
### Rolling predictions 12
<!-- #endregion -->

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((0,1,12), df, split, batch=12)
plot_test_with_predictions(test_data, predictions, "Predict with AR model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ma_model_name, "Rolling_prediction_12", metrics)
```

### Predict all test at once

```python
train_data, test_data, _ = split_train_test(df,test_size_precent)
models, predictions = fit_model_arima_all_test((0,1,12), train_data,test_data)
plot_test_with_predictions(test_data, predictions, "Predict with MA model")

metrics = list(get_metrics(test_data.values, predictions.values))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, ma_model_name, "Predict_all_test", metrics)
```

### Comparision

```python
compare_model_tests(ma_model_name)
```

<!-- #region tags=[] -->
## ARIMA model
<!-- #endregion -->

```python
arima_model_name = "ARIMA"
```

```python
pm.auto_arima(df.Num_Passengers,
    start_p=1,
    start_q=1,
    max_p=20,
    max_q=20,
    max_order = None,
    maxiter = 500,
    seasonal=False,
    trace=True,
    error_action='ignore',
    stepwise =False,
    scoring='mse',
    information_criterion ='aic',
    method = 'nm')
```

### Rolling prediction

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((8,1,8), df, split)
plot_test_with_predictions(test_data, predictions, "Predict with ARIMA model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, arima_model_name, "Rolling_prediction_1", metrics)
```

```python
models[-1].summary()
```

### Rolling presictions 12

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima((8,1,8), df, split, batch=12)
plot_test_with_predictions(test_data, predictions, "Predict with AR model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, arima_model_name, "Rolling_prediction_12", metrics)
```

### Predict all test at once

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
models, predictions = fit_model_arima_all_test((8,1,8), train_data, test_data)
plot_test_with_predictions(test_data, predictions, "Predict with ARIMA model")

metrics = list(get_metrics(test_data.values, predictions.values))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, arima_model_name, "Predict_all_test", metrics)
```

### Comparision 

```python
compare_model_tests(arima_model_name)
```

<!-- #region tags=[] -->
# SARIMA
<!-- #endregion -->

```python
sarimax_model_name = "SARIMAX"
```

```python
pm.auto_arima(df.Num_Passengers,
    start_p=1,
    start_q=1,
    max_p=20,
    max_q=20,
    start_P = 1,
    start_D = 1,
    start_Q = 1,
    max_P = 20,
    max_D = 20,
    max_Q = 20,
    m=12,
    max_order = None,
    maxiter = 800,
    random =True,
    seasonal=True,
    trace=True,
    error_action='ignore',
    scoring='mse',
    information_criterion ='aic',
    method = 'bfgs')
```

<!-- #region tags=[] -->
### Rolling prediction
<!-- #endregion -->

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_sarimax((0,1,1),(2,1,1, 12), df, split)
plot_test_with_predictions(test_data, predictions, "Predict with SARIMAX model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, sarimax_model_name, "Rolling_prediction_1", metrics)
```

### Rolling prediction 12

```python
_, test_data, split = split_train_test(df, test_size_precent)
models, predictions = fit_model_sarimax((0,1,1),(2,1,1, 12), df, split,  batch = 12)
plot_test_with_predictions(test_data, predictions, "Predict with SARIMAX model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, sarimax_model_name, "Rolling_prediction_12", metrics)
```

### Predict all test at once

```python
train_data, test_data, _ = split_train_test(df, test_size_precent)
models, predictions = fit_model_sarimax_all_test((0,1,1),(2,1,1, 12), train_data, test_data)
plot_test_with_predictions(test_data, predictions, "Predict with SARIMA model")

metrics = list(get_metrics(test_data.values, predictions))
print_metrics(metrics)
models_metrics = save_metrics(models_metrics, sarimax_model_name, "Predict_all_test", metrics)
```

### Comparision

```python
compare_model_tests(sarimax_model_name)
```

<!-- #region tags=[] -->
# Compare models that predicts one value at a time
<!-- #endregion -->

```python
n_nest_models = 5
```

```python
for metric in METRICS_NAMES:
    models_metrics[metric] = pd.to_numeric(models_metrics[metric])
```

```python tags=[]
rolling_prediction = models_metrics[models_metrics["TestName"]=="Rolling_prediction_1"]
```

```python tags=[]
for metric in METRICS_NAMES:
    plt.title(f"Compare models by {metric}")
    sns.barplot(data=rolling_prediction, x=metric, y="Model")
    plt.show()
```

```python
for metric in METRICS_NAMES:
    plt.title(f"Compare {n_nest_models} best models by {metric}")
    sns.barplot(data=rolling_prediction.nsmallest(n_nest_models, metric), x=metric, y="Model")
    plt.show()
```

<!-- #region tags=[] -->
# Compare models that predicts 12 months
<!-- #endregion -->

```python
rolling_prediction_year = models_metrics[models_metrics["TestName"]=="Rolling_prediction_12"]
```

```python
for metric in METRICS_NAMES:
    plt.title(f"Compare models by {metric}")
    sns.barplot(data=rolling_prediction_year, x=metric, y="Model")
    plt.show()
```

```python
for metric in METRICS_NAMES:
    plt.title(f"Compare {n_nest_models} best models by {metric}")
    sns.barplot(data=rolling_prediction_year.nsmallest(n_nest_models, metric), x=metric, y="Model")
    plt.show()
```

# Compare models that predicts all test 

```python
all_test_metrics = models_metrics[models_metrics["TestName"]=="Predict_all_test"]
```

```python
for metric in METRICS_NAMES:
    plt.title(f"Compare models by {metric}")
    sns.barplot(data=all_test_metrics, x=metric, y="Model")
    plt.show()
```

```python
for metric in METRICS_NAMES:
    plt.title(f"Compare {n_nest_models} best models by {metric}")
    sns.barplot(data=all_test_metrics.nsmallest(n_nest_models, metric), x=metric, y="Model")
    plt.show()
```

<!-- #region tags=[] -->
# Scalecast
<!-- #endregion -->

```python
f = Forecaster(y=df.Num_Passengers, current_dates=df.index)
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
    start_p=1,
    start_q=1,
    max_p=20,
    max_q=20,
    m=12,
    seasonal=True,
    max_P=20, 
    max_D=20,
    max_Q=20,
    max_d=20,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion="bic", #{'bic', 'hqic', 'oob', 'aicc', 'aic'}
    alpha=0.05,
    scoring='mae',
    call_me='arima3',
)

f.plot_test_set(ci=True,models='arima3')
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()
```

```python
f.regr.summary()
```

# Misc


## Another option to detrend

```python
y_detrend = ((df - df.rolling(window=12).mean())/df.rolling(window=12).std())[11:]
plt.figure(figsize=(9,3))
plt.title("Detrend")
sns.lineplot(x="Month", y="Num_Passengers", data=y_detrend, )
plt.show()
adf_test(y_detrend)
```
