# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# In[2]:


cars = pd.read_csv('mtcars.csv')

# In[10]:


cars.head()

# In[11]:


# feature engineering
x = cars.loc[:, ["hp", "wt", "mpg"]]
y = cars.am

# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# In[13]:


model = linear_model.LogisticRegression()

# In[14]:


model.fit(x_train, y_train)

# In[16]:


predict_x = model.predict(x_test)
predict_x

# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix

# In[19]:


accuracy_score(y_test, predict_x)

# In[23]:


confusion_matrix(y_test, predict_x)
