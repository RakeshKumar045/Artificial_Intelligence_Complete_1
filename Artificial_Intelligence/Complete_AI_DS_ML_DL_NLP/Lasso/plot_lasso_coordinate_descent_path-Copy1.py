# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#
# # Lasso and Elastic Net
# 
# 
# Lasso and elastic net (L1 and L2 penalisation) implemented using a
# coordinate descent.
# 
# The coefficients can be forced to be positive.
# 
# 

# In[7]:


import matplotlib.pyplot as plt
# Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 10

# Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i * np.pi / 180 for i in range(60, 300, 4)])
np.random.seed(10)  # Setting seed for reproducability
y = np.sin(x) + np.random.normal(0, 0.15, len(x))
data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
plt.plot(data['x'], data['y'], '.')

# In[10]:


data.head(5)

# In[8]:


for i in range(2, 16):  # power of 1 is already there
    colname = 'x_%d' % i  # new var will be x_power
    data[colname] = data['x'] ** i
print
data.head()

# In[4]:


dataset.head

# In[13]:


# Import Linear Regression model from scikit-learn.
from sklearn.linear_model import LinearRegression


def linear_regression(data, power, models_to_plot):
    # initialize predictors:
    predictors = ['x']
    if power >= 2:
        predictors.extend(['x_%d' % i for i in range(2, power + 1)])

    # Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors], data['y'])
    y_pred = linreg.predict(data[predictors])

    # Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('Plot for power: %d' % power)

    # Return the result in pre-defined format
    rss = sum((y_pred - data['y']) ** 2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret


# In[14]:


# Initialize a dataframe to store the results:
col = ['rss', 'intercept'] + ['coef_x_%d' % i for i in range(1, 16)]
ind = ['model_pow_%d' % i for i in range(1, 16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

# Define the powers for which a plot is required:
models_to_plot = {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236}

# Iterate through all powers and assimilate results
for i in range(1, 16):
    coef_matrix_simple.iloc[i - 1, 0:i + 2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
