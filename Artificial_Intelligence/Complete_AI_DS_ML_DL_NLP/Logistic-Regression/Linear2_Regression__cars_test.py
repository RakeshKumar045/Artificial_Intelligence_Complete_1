# coding: utf-8

# In[20]:


# import the packages
# import the packages
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# In[5]:


cars = pd.read_csv('mtcars.csv')

# In[4]:


get_ipython().run_line_magic('pwd', '')

# In[13]:


# feature engineering
x = cars.loc[:, ["hp", "wt", "am"]]
y = cars.mpg

# In[11]:


x.head(3)

# In[12]:


y.head(3)

# In[114]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=10)  # stratify = y : not for regression
# x_train, x_test , y_train , y_test = train_test_split(x ,y , test_size = 0.3 , random_state = 10 , stratify = y) stratify = y : only for classifiacation


# In[98]:


model = linear_model.LinearRegression()

# In[99]:


model.fit(x_train, y_train)

# In[100]:


predict_x = model.predict(x_test)

# In[101]:


accuracy = mean_squared_error(y_test, predict_x)

# In[102]:


predict_x

# In[103]:


y_test

# In[109]:


from sklearn.metrics import r2_score, mean_squared_error

# In[111]:


accuracy = r2_score(y_test, predict_x)  # r2_score is standrad accuracy
accuracy

# In[112]:


accuracy1 = mean_squared_error(y_test, predict_x)
accuracy1
