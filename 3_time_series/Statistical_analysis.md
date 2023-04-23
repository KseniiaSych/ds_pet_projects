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
from sklearn.model_selection import train_test_split

from scalecast.Forecaster import Forecaster
from scalecast.auxmodels import auto_arima
```

<!-- #region tags=[] -->
# Read and view data
<!-- #endregion -->

```python

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
plt.title("Amount of air passagers by month")
sns.lineplot(x="Month", y="Num_Passengers",
             data=df)
plt.show()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true tags=[] -->
# ACF & PACF
<!-- #endregion -->

```python
acf_plot = plot_acf(df.Num_Passengers, lags=20)
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
ln = 28
param_names =[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ]
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    for name, value in zip(param_names, dftest[0:4]):
        print(name.ljust(ln), value)
    is_not = 'is' if dftest[1]<0.05 else 'is not'
    print(f"Time series {is_not} stationary")
```

```python
adf_test(df)
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
sns.lineplot(x="Month", y="Num_Passengers", data=first_dif, )
plt.show()
adf_test(first_dif)
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
adf_test(yearly_dif)
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

<!-- #region tags=[] -->
### Fit common functions
<!-- #endregion -->

```python
def linear_regression(x, y, predict):
    model = LinearRegression()
    fit = model.fit(x, y)
    pred = fit.predict(predict)[0]
    return fit, pred
```

```python
<<<<<<< Updated upstream
def fit_linear_regression_with_12_lag(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data.dropna(inplace=True)
        
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
=======
>>>>>>> Stashed changes
def fit_model_arima(order, dataframe, split, method=None, verbose=False):
    model_fit = []
    predictions =  []
    
    for i, cur_date in enumerate(pd.date_range(split,  dataframe.index[-1], freq='MS')):
        if(verbose): print(i, " iteration")
        model = ARIMA(dataframe[:cur_date - timedelta(days=1)], order = order)
        fit = model.fit(method=method)
        pred = fit.forecast() 
        
        model_fit.append(fit)
        predictions.append(pred)
    
    predictions_df = pd.DataFrame(pd.concat(predictions, axis=0), columns=['Num_Passengers'])
    return model_fit, predictions_df
```

```python
def fit_model_arima_all_test(order, test, train, method=None):
    model = ARIMA(test, order = order)
    fit = model.fit(method=method)
    pred = fit.predict(start=train.index[0], end=train.index[-1]) 
    return fit, pred
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

### Split

```python
def fit_model_sarimax_all_test(order, seasonal_order,test, train):
    model = SARIMAX(test, order=order, seasonal_order=seasonal_order)
    fit = model.fit()
    pred = fit.predict(start=train.index[0], end=train.index[-1]) 

    return fit, pred
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Split
<!-- #endregion -->

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

<<<<<<< Updated upstream
### Metrics and plots
=======
<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Metrics and plots
<!-- #endregion -->
>>>>>>> Stashed changes

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
def model_stats(model_fit):
    print(model_fit.summary())
     
    fig, axs = plt.subplots(nrows=2, ncols=1)
    plt.subplots_adjust(hspace=0.7)
    
    model_fit.resid.plot(ax=axs[0], title = "Model residuals")
    model_fit.resid.plot(kind='kde', ax=axs[1], title = "Last model residuals distribution")
    plt.show()
```

<!-- #region tags=[] -->
## Baseline
<!-- #endregion -->

<!-- #region tags=[] -->
### Predict previous value
<!-- #endregion -->

```python
def predict_previous_value(dataframe, split):
    return dataframe[split + relativedelta(months=-1): ][:-1].shift(1, freq='M')
```

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

```python tags=[]
def linear_regression_with_season(x, y, predict):
    if len(y)<12:
        raise ValueError("This prediction require more then year of test data")
    model = LinearRegression()
    fit = model.fit(x, y)
    predict_test = fit.predict(x)
    predictions = fit.predict(predict)
    for i in range(len(predictions)):
        predictions[i]+= (y[i-12] - predict_test[i-12])
    return fit, predictions
```

#### Rolling predictions

```python
def fit_linear_regression_with_season(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        data = dataframe[:cur_date - timedelta(days=1)]['Num_Passengers'].values.reshape(-1, 1)
        index = np.arange(len(data)).reshape(-1, 1)
        fit, pred = linear_regression_with_season(index, data, [index[-1]+1])
        
        model_fit.append(fit)
        predictions.extend(pred)
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```

```python
_, lr_test_data, lr_split_date = split_train_test(df, 0.70)
lr_models, lr_predictons = fit_linear_regression_with_season(df, lr_split_date)
plot_test_with_predictions(lr_test_data, lr_predictons, "Linear regression with lag residual")

lr_mse, lr_rmse, lr_mae = get_metrics(lr_test_data.values, lr_predictons)
print_metrics(lr_mse, lr_rmse, lr_mae)
```

<<<<<<< Updated upstream
### Linear regression a1 *xt + a2*xt-12 +b
=======
<!-- #region tags=[] -->
#### Predict all test at once
<!-- #endregion -->

```python tags=[]
def fit_linear_regression_with_season_all_test(train, test):
    y = train['Num_Passengers'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    x_test =  np.arange(len(test)).reshape(-1, 1)
    fit, pred = linear_regression_with_season(x, y, x_test)
    predictions_df = pd.DataFrame({'Month': test.index,
                   'Num_Passengers':pred.flatten()})
    predictions_df = predictions_df.set_index('Month')
    return fit, predictions_df
```

```python
lr1_train_data, lr1_test_data, _ = split_train_test(df, 0.70)
lr1_models, lr1_predictons = fit_linear_regression_with_season_all_test(lr1_train_data, lr1_test_data)
plot_test_with_predictions(lr1_test_data, lr1_predictons, "Linear regression with lag residual")

lr1_mse, lr1_rmse, lr1_mae = get_metrics(lr1_test_data.values, lr1_predictons)
print_metrics(lr1_mse, lr1_rmse, lr1_mae)
```

### Linear regression a1 *xt + a2*xt-12 +b


#### Rolling predictions

```python
def fit_linear_regression_with_12_lag(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data.dropna(inplace=True)
        
        train_col = ["lag_1", "lag_12"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"], test_size = 1, shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
>>>>>>> Stashed changes

```python
_, b1_test_data, b1_split_date = split_train_test(df, 0.70)
_,b1_predictons = fit_linear_regression_with_12_lag(df, b1_split_date)
plot_test_with_predictions(b1_test_data, b1_predictons, "Linear regression a1 *xt + a2*xt-12 +b")

b1_mse, b1_rmse, b1_mae = get_metrics(b1_test_data.values, b1_predictons)
print_metrics(b1_mse, b1_rmse, b1_mae)
```

<<<<<<< Updated upstream
### Linear regression  a1 *xt + a2*(xt-12 - xt) +b
=======
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
    predictions_df = pd.DataFrame({'Month': test_index,
                   'Num_Passengers': pred_test.flatten()})
    predictions_df = predictions_df.set_index('Month')
    
    return reg, predictions_df
```

```python
b12_train_data, b12_test_data, _ = split_train_test(df, 0.70)
_, b12_predictons = fit_linear_regression_with_12_lag_all_test(b12_train_data, b12_test_data)
plot_test_with_predictions(b12_test_data.loc[b12_predictons.index], b12_predictons, "Linear regression a1 *xt + a2*xt-12 +b")

b12_mse, b12_rmse, b12_mae = get_metrics(b12_test_data.loc[b12_predictons.index].values, b12_predictons)
print_metrics(b12_mse, b12_rmse, b12_mae)
```

### Linear regression  a1 *xt + a2*(xt-12 - xt) +b

<!-- #region tags=[] -->
#### Rolling predictions
<!-- #endregion -->

```python
def fit_linear_regression_with_diff_of_12_lag(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_12"] = train_data["Num_Passengers"].shift(12)
        train_data["lag_1"] =train_data["Num_Passengers"].shift(1)
        train_data.dropna(inplace=True)
        train_data["diff_lag"] =  train_data["lag_12"] - train_data["lag_1"] 
       
        train_col = ["lag_1", "diff_lag"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"], test_size = 1, shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
       
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
>>>>>>> Stashed changes

```python
_, b2_test_data, b2_split_date = split_train_test(df, 0.70)
_,b2_predictons = fit_linear_regression_with_diff_of_12_lag(df, b2_split_date)
plot_test_with_predictions(b2_test_data, b2_predictons, "Linear regression  a1 *xt + a2*(xt-12 - xt) +b")

b2_mse, b2_rmse, b2_mae = get_metrics(b2_test_data.values, b2_predictons)
print_metrics(b2_mse, b2_rmse, b2_mae)
```

<<<<<<< Updated upstream
### Linear regression a1 *xt + a2*(xt-11 - xt-12) +b
=======
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
b22_train_data, b22_test_data, _ = split_train_test(df, 0.70)
_,b22_predictons = fit_linear_regression_with_diff_of_12_lag_all_test(b22_train_data, b22_test_data)
plot_test_with_predictions(b22_test_data.loc[b22_predictons.index], b22_predictons, "Linear regression  a1 *xt + a2*(xt-12 - xt) +b")

b22_mse, b22_rmse, b22_mae = get_metrics(b22_test_data.loc[b22_predictons.index], b22_predictons)
print_metrics(b22_mse, b22_rmse, b22_mae)
```

<!-- #region tags=[] -->
### Linear regression a1 *xt + a2*(xt-11 - xt-12) +b
<!-- #endregion -->

#### Rolling predictions

```python
def fit_linear_regression_with_lag_of_12_diff(dataframe, split, verbose=False):
    model_fit = []
    predictions =  []
    prediction_range = pd.date_range(split,  dataframe.index[-1], freq='MS')
    for i, cur_date in enumerate(prediction_range):
        if(verbose): print(i, " iteration")
        train_data = dataframe[:cur_date].reset_index(drop=True)
        train_data["lag_1"] = train_data["Num_Passengers"].shift(1)
        train_data["lag_diff_11"] = train_data["Num_Passengers"].shift(11)
        train_data["lag_diff_12"] = train_data["Num_Passengers"].shift(12)
        train_data.dropna(inplace=True)
        train_data["diff_lag"] =  train_data["lag_diff_11"] - train_data["lag_diff_12"] 
       
        train_col = ["lag_1", "lag_diff_12"]
        X_train, X_test, y_train, _ = train_test_split(train_data[train_col],  train_data["Num_Passengers"], test_size = 1, shuffle = False)
        fit, pred = linear_regression(X_train, y_train , X_test)
        
        model_fit.append(fit)
        predictions.append(pred)
        
    predictions_df = pd.DataFrame({'Month': prediction_range,
                   'Num_Passengers': predictions})
    predictions_df = predictions_df.set_index('Month')
    
    return model_fit, predictions_df
```
>>>>>>> Stashed changes

```python
_, b3_test_data, b3_split_date = split_train_test(df, 0.70)
_, b3_predictons = fit_linear_regression_with_lag_of_12_diff(df, b3_split_date)
plot_test_with_predictions(b3_test_data, b3_predictons, "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")

b3_mse, b3_rmse, b3_mae = get_metrics(b3_test_data.values, b3_predictons)
print_metrics(b3_mse, b3_rmse, b3_mae)
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
b32_train_data, b32_test_data, _ = split_train_test(df, 0.70)
_, b32_predictons = fit_linear_regression_with_lag_of_12_diff_all_test(b32_train_data, b3_test_data)
plot_test_with_predictions(b32_test_data.loc[b32_predictons.index], b32_predictons,\
                           "Linear regression a1 *xt + a2*(xt-11 - xt-12) +b")

b32_mse, b32_rmse, b32_mae = get_metrics(b32_test_data.loc[b32_predictons.index].values, b32_predictons)
print_metrics(b32_mse, b32_rmse, b32_mae)
```

<!-- #region tags=[] -->
## ExponentialSmoothing
<!-- #endregion -->

#### Rolling predictions

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
_, es_test_data, es_split_date = split_train_test(df, 0.70)
es_models, es_predictons = exponentialSmoothing(df, es_split_date)
plot_test_with_predictions(es_test_data, es_predictons, "ExponentialSmoothing")

es_mse, es_rmse, es_mae = get_metrics(es_test_data.values, es_predictons)
print_metrics(es_mse, es_rmse, es_mae)
```

```python
es_models[-1].summary()
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
es2_train_data, es2_test_data, es2_split_date = split_train_test(df, 0.70)
es2_models, es2_predictons = exponentialSmoothing_all_test(es2_train_data, es2_test_data)
plot_test_with_predictions(es2_test_data, es2_predictons, "ExponentialSmoothing")
es2_mse, es2_rmse, es2_mae = get_metrics(es2_test_data.values, es2_predictons)
print_metrics(es2_mse, es2_rmse, es2_mae)
```

<!-- #region tags=[] -->
## AR model
<!-- #endregion -->

### Predict all test at once

```python
train_data, test_data, _ = split_train_test(df, 0.7)
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

ar1_mse, ar1_rmse, ar1_mae = get_metrics(test_data.values, predictions)
print_metrics(ar1_mse, ar1_rmse, ar1_mae)
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
model_stats(ar_models[-1])
```

<!-- #region tags=[] -->
## MA model
<!-- #endregion -->

### Predict one value then fit model again

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

### Predict all test at once

```python
ma1_train_data, ma1_test_data, _ = split_train_test(df, 0.70)
ma1_models, ma1_predictions = fit_model_arima_all_test((0,1,2), ma1_train_data, ma1_test_data)
plot_test_with_predictions(ma1_test_data, ma1_predictions, "Predict with MA model")

ma1_mse, ma1_rmse, ma1_mae = get_metrics(ma1_test_data.values, ma1_predictions.values)
print_metrics(ma1_mse, ma1_rmse, ma1_mae)
```

<!-- #region tags=[] -->
## ARIMA model
<!-- #endregion -->

### Predict one value then fit model again

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

### Predict all test at once

```python
i1_train, i1_test_data, _ = split_train_test(df, 0.70)
i1_models, i1_predictions = fit_model_arima_all_test((2,1,1), i1_train, i1_test_data)
plot_test_with_predictions(i1_test_data, i1_predictions, "Predict with ARIMA model")

i1_mse, i1_rmse, i1_mae = get_metrics(i1_test_data.values, i1_predictions.values)
print_metrics(i1_mse, i1_rmse, i1_mae)
```

<!-- #region tags=[] -->
# SARIMA
<!-- #endregion -->

<!-- #region tags=[] -->
### Predict one value then fit model again
<!-- #endregion -->

```python
_, si_test_data, si_split = split_train_test(df, 0.70)
si_models, si_predictions = fit_model_sarimax((1, 1, 1), (1,1,1,12) , df, si_split)
plot_test_with_predictions(si_test_data, si_predictions, "Predict with SARIMA model")

si_mse, si_rmse, si_mae = get_metrics(si_test_data.values, si_predictions)
print_metrics(si_mse, si_rmse, si_mae)
```

### Predict all test at once

```python
si1_train_data, si1_test_data, _ = split_train_test(df, 0.70)
si1_models, si1_predictions = fit_model_sarimax_all_test((1, 1, 1), (1,1,1,12) , si1_train_data, si1_test_data)
plot_test_with_predictions(si1_test_data, si1_predictions, "Predict with SARIMA model")

si1_mse, si1_rmse, si1_mae = get_metrics(si1_test_data.values, si1_predictions.values)
print_metrics(si1_mse, si1_rmse, si1_mae)

```

# Compare models that predicts one value at a time

```python
models_metric = pd.DataFrame({"MSE":[b_mse, b1_mse, b2_mse, b3_mse, lr_mse, es_mse, ar_mse, ma_mse, i_mse, si_mse],
                            "RMSE": [b_rmse, b1_rmse, b2_rmse, b3_rmse, lr_rmse, es_rmse, ar_rmse, ma_rmse, i_rmse, si_rmse],
                            "MAE": [b_mae, b1_mae, b2_mae, b3_mae, lr_mae, es_mae, ar_mae, ma_mae, i_mae, si_mae],
                             "Model":["Last_value", "Linear_regression v0", "Linear_regression v1", "Linear_regression v2" , \
                                      "Linear_regression witt residuals" , "Exponential soothing", "AR", "MA",  "ARIMA", "SARIMA" ]
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

# Compare models that predicts all test set at once

```python
models_metric_all_test = pd.DataFrame({"MSE":[b_mse, b12_mse, b22_mse, b32_mse, lr1_mse, es2_mse, ar1_mse, ma1_mse, i1_mse, si1_mse],
                            "RMSE": [b_rmse, b12_rmse, b22_rmse, b32_rmse, lr1_rmse, es2_rmse, ar1_rmse, ma1_rmse, i1_rmse, si1_rmse],
                            "MAE": [b_mae, b12_mae, b22_mae, b32_mae, lr1_mae, es2_mae, ar1_mae, ma1_mae, i1_mae, si1_mae],
                             "Model":["Last_value", "Linear_regression v0", "Linear_regression v1", "Linear_regression v2" , \
                                     "Linear_regression witt residuals" , "Exponential soothing", "AR", "MA",  "ARIMA", "SARIMA" ]
                             })
```

```python
models_metric_all_test
```

```python
plt.title("Compare models by MSE")
sns.barplot(data=models_metric_all_test, x="MSE", y="Model")
plt.show()
```

```python
plt.title("Compare models by RMSE")
sns.barplot(data=models_metric_all_test, x="RMSE", y="Model")
plt.show()
```

```python
plt.title("Compare models by MAE")
sns.barplot(data=models_metric_all_test, x="MAE", y="Model")
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

```python

```
