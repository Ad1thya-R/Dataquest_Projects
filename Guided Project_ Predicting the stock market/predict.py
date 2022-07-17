import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
sp500=pd.read_csv('sphist.csv')
#convert date column into date type in pandas
sp500['Date']=pd.to_datetime(sp500['Date'])
sp500.sort_values(['Date'],inplace=True)
#Let us choose 3 indicators to compute
'''
1. the average price of the past 5 days
2. the standard deviation of the past 5 days 
3. ratio between average of past 5 days and average of past 365 days
'''
sp500['avg_5']=sp500['Close'].rolling(5).mean().shift().fillna(0)
sp500['std_5']=sp500['Close'].rolling(5).std().shift().fillna(0)    
avg_365=sp500['Close'].rolling(365).mean().shift() 
sp500['ratio_5_365']=sp500['avg_5']/avg_365

#2 additional features to check for improved performanc
sp500['std_365']=sp500['Close'].rolling(365).std().shift().fillna(0)
sp500['avg_365']=avg_365

'''
To avoid issues with the 365 day rolling window, remove any dates before 151-01-03
'''
sp500=sp500[sp500["Date"] > datetime(year=1951, month=1, day=2)]
sp500=sp500.dropna(axis=0)
'''
split test and train datasets
train should contain any rows in the data with a date less than 2013-01-01. test should contain any rows with a date greater than or equal to 2013-01-01.
'''
train=sp500[sp500["Date"] < datetime(year=2013, month=1, day=1)]
test=sp500[sp500["Date"] >= datetime(year=2013, month=1, day=1)]
'''
train and test an ML model
'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
train_unbiased=train.drop(['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'], axis=1)
features=['avg_365', 'std_365']
errors=[]
for n in range(3): 
    feat_slice=features[:n]
    unbiased_slice=train_unbiased.drop(feat_slice, axis=1)
    reg=LinearRegression().fit(unbiased_slice, train[['Close']])
    predictions=reg.predict(test[unbiased_slice.columns])
    error=mean_squared_error(predictions, test[['Close']])
    errors.append(error)

print('Base 3 features rmse=', np.sqrt(errors[2]))
print('4 features rmse=', np.sqrt(errors[1]))
print('5 features rmse=', np.sqrt(errors[0]))

plt.plot(test['Date'],test['Close'], color='red')
plt.plot(test['Date'], predictions)
plt.show()






























