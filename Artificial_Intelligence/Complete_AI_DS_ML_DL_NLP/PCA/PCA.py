# coding: utf-8

# # Principal component analysis (PCA)

# In[7]:


import pandas as pd

import seaborn as sb
from pylab import rcParams

from sklearn import datasets

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# ### PCA on the iris dataset_D

# In[10]:


iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names
variable_names

# In[11]:


from sklearn.decomposition import PCA

pca = PCA()

# In[12]:


X[0:5, :]

# In[13]:


iris_pca = pca.fit_transform(X)

p_df = pd.DataFrame(pca.explained_variance_ratio_)
p_df.plot(kind='bar')

# In[14]:


X[0:3, :]

# In[15]:


iris_pca[0:3, :]

# In[16]:


pca.explained_variance_ratio_

# In[17]:


X_pca = iris_pca[:, 0:2]

# In[18]:


X_pca.shape

# In[19]:


pca.explained_variance_ratio_

# In[20]:


comps = pd.DataFrame(pca.components_, index=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                     columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', ])
comps

# In[21]:


sb.heatmap(comps)
