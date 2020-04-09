# coding: utf-8

# In[32]:


import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# In[2]:


data = datasets.load_breast_cancer()

# In[4]:


data.data.shape

# In[6]:


# data.data[0:3]


# In[7]:


x = data.data
y = data.target

# In[8]:


x.shape

# In[10]:


# x[0:2]


# In[11]:


pca = PCA()

# In[12]:


x_pca = pca.fit_transform(x)

# In[15]:


pca.explained_variance_

# In[17]:


pca.explained_variance_[0] + pca.explained_variance_[1]

# In[18]:


pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]

# In[19]:


x = x_pca[:, 0:2]

# In[20]:


x.shape

# In[21]:


model_nn = MLPClassifier()

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y)

# In[24]:


model_nn.fit(x, y)

# In[26]:


y_predict = model_nn.predict(x_test)

# In[30]:


# data.target
# y


# In[34]:


accuracy = accuracy_score(y_test, y_predict)
accuracy

# In[35]:


pd.crosstab(y_test, y_predict)
