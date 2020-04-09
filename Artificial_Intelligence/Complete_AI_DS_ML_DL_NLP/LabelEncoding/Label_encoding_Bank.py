# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


data = pd.read_csv('bank-full.csv', ';')

# In[3]:


from collections import Counter as c

c(data.y)

# In[4]:


data.shape

# In[5]:


data.head()

# In[6]:


c(data.job)

# In[7]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

# In[8]:


data.job = enc.fit_transform(data.job)
data.marital = enc.fit_transform(data.marital)
data.education = enc.fit_transform(data.education)
data.default = enc.fit_transform(data.default)
data.housing = enc.fit_transform(data.housing)

# In[18]:


data.head()

# In[9]:


data.marital
