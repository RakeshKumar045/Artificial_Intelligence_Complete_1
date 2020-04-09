# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
from pandas import datetime
from pandas import read_csv


def parser(d):
    return datetime.strptime(d, '%Y-%m')


parser("2018-08")

# In[7]:


datetime.strptime("2018-08", "%Y-%m")

# In[18]:


sales = read_csv('sales-cars.csv', parse_dates=[0], index_col=0, date_parser=parser)
plt.plot(sales)

# In[19]:


sales.shape

# In[20]:


type(sales)

# In[25]:


sales.head(5)

# In[22]:


sales.isnull().sum()

# In[24]:


sales.index[1]

# In[27]:


sales.plot()  # sales.plot and sales.plot() : both are different

# In[5]:


from pandas.plotting import autocorrelation_plot

autocorrelation_plot(sales)
plt.show()

# In[50]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

X = sales.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []

# In[51]:


X.shape

# In[52]:


for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test_Amat MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# In[ ]:


# blue line tensorbroad_pb_android data and red line train data
