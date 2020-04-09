# coding: utf-8

# In[14]:


import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# In[16]:


sale = [2, 1, 3]
sale_predict = [2, 1, 2]

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')

# In[35]:


r2_score(sale, sale_predict)

# In[36]:


mean_squared_error(sale, sale_predict)

# In[37]:


d = [3, 4, 3, 2, 4, 20]
columns = ["Sales"]

# In[38]:


data = pd.DataFrame(d, columns=columns)

# In[39]:


data.head

# In[40]:


data.Sales.plot(kind="box")

# In[32]:


# how to find outlier
data[data.Sales < 3 * data.Sales.std()]

# In[44]:


data.Sales.plot(kind="box")

# In[45]:


data.Sales.std

# In[46]:


data[(data.Sales < 3 * data.Sales.std()) & (
        data.Sales > -3 * data.Sales.std())]  # standrad formula for outlier with 3_dimension

# In[48]:


data[(data.Sales < 3 * data.Sales.std()) & (data.Sales > -3 * data.Sales.std())].plot(kind="box")

# In[49]:


data[(data.Sales < 2 * data.Sales.std()) & (
        data.Sales > -2 * data.Sales.std())]  # standrad formula for outlier with 2_dimension

# In[50]:


data[(data.Sales < 2 * data.Sales.std()) & (data.Sales > -2 * data.Sales.std())].plot(kind="box")
