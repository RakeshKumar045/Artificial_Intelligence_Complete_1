# coding: utf-8

# # Logistic Regression

# In[3]:


import pandas as pd
import seaborn as sb
from pylab import rcParams
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

# In[4]:
# In[5]:
# In[30]:


# In[20]:


# Setting up the plot
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# ### Logistic regression on mtcars

# In[14]:


address = 'mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()

# In[15]:


# Locating values
cars_data = cars.iloc[:, [5, 11]].values
cars_data_names = ['drat', 'carb']
y = cars.iloc[:, 9].values

# #### Checking for independence between features

# In[24]:


sb.regplot(x='drat', y='carb', data=cars, scatter=True)

# In[17]:


# Correlation
drat = cars['drat']
carb = cars['carb']

spearmanr_coefficient, p_value = spearmanr(drat, carb)
print('Spearman Rank Correlation Coefficient %0.3f' % (spearmanr_coefficient))

# #### Checking for missing values

# In[18]:


cars.isnull().sum()

# #### Checking that your target is binary or ordinal

# In[19]:


sb.countplot(x='am', data=cars, palette='hls')

# #### Checking that your dataset_D size is sufficient

# In[11]:


cars.info()

# #### Deploying and evaluating your model

# In[27]:


# Scaling the values
X = scale(cars_data)

# In[28]:


# Defining and fitting LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X, y)

# In[31]:


# Prediction
y_pred = LogReg.predict(X)
print(classification_report(y, y_pred))
print(metrics.accuracy_score(y, y_pred))
