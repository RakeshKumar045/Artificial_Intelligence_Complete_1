# coding: utf-8

# In[2]:


import pandas as pd

# In[50]:


data = pd.read_csv("mtcars.csv")

# In[51]:


data.shape

# In[52]:


data.head()

# In[53]:


data.tail()

# In[54]:


print(data.columns)

# In[58]:


data.describe()

# In[56]:


print(type(data))
print(type(data.car_model))

# In[57]:


data.info()

# In[39]:


# data.car_model


# In[61]:


# data.columns = ['hi', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs','am', 'gear', 'carb']


# In[62]:


data.head()

# In[70]:


# data.isnull() # or data.isnull


# In[69]:


ata.isnull().sum()

# In[75]:


x = data.loc[:, ['cyl', 'hp']]
y = data.mpg

# In[76]:


from sklearn.model_selection import train_test_split

# In[118]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# In[119]:


from sklearn.linear_model import LinearRegression

# In[120]:


model = LinearRegression()

# In[126]:


model.fit(x_train, y_train)

# In[128]:


save(example.ml)

# In[122]:


pred_x = model.predict(x_test)

# In[123]:


from sklearn.metrics import r2_score

# In[124]:


accuracy = r2_score(y_test, pred_x)

# In[125]:


print(accuracy)

# In[130]:


# save the model to disk
from sklearn.externals import joblib

filename = 'first_model.ml'
joblib.dump(model, filename)
