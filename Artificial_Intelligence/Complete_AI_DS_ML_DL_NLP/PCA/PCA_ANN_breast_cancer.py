# coding: utf-8

# In[26]:


import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale

# In[27]:


data = datasets.load_breast_cancer()

# In[28]:


X = data.data
y = data.target

# In[29]:


X.shape

# In[30]:


# Scaling , it will increase the accuracy
X = scale(X)

# In[31]:


X.shape

# In[32]:


pca = PCA()

# In[33]:


X_pca = pca.fit_transform(X)

# In[34]:


pca.explained_variance_ratio_

# In[60]:


pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2] + \
pca.explained_variance_ratio_[3] + pca.explained_variance_ratio_[4] + pca.explained_variance_ratio_[5] + \
pca.explained_variance_ratio_[6]

# In[61]:


X = X_pca[:, 0:6]

# In[62]:


X.shape

# In[63]:


model_nn = MLPClassifier()

# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

# In[65]:


model_nn.fit(X_train, y_train)

# In[66]:


y_predict = model_nn.predict(X_test)

# In[67]:


accuracy_score(y_test, y_predict)

# In[52]:


pd.crosstab(y_test, y_predict)
