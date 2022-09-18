!pip install pystan~=2.14 # Python interface to Stan for Bayesian inference
!pip install fbprophet # Making forecasts for time series datasets
!pip install openpyxl # To export dataframe to excel
from prophet.plot import plot_plotly, plot_components_plotly

from google.colab import drive # To mount google drive on google Colab
drive.mount('/content/gdrive')

from fbprophet import Prophet
import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O
from sklearn.metrics import mean_squared_error


df = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/Adult Milk FTA.csv')

df["ds"]= pd.to_datetime(df["ds"])

df.info()
df.head()
df.tail()
df.plot(x='ds',y='y',figsize=(18,6)) # Plot your data - NEED TO WORK ON THE DATASET IN EXCEL

len(df)

train = df.iloc[:len(df)-365]
test = df.iloc[len(df)-365:]

m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=365) #MS for monthly
forecast = m.predict(future)

forecast.tail()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

test.tail()

plot_plotly(m ,forecast)

plot_components_plotly(m, forecast)

from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-365:]['yhat']

print("Root Mean Squared Error between actual and  predicted values: ",rmse(predictions,test['y']))
print("Mean Value of Test Dataset:", test['y'].mean())
