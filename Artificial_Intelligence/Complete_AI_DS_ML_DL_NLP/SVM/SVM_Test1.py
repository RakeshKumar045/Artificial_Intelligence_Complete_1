# coding: utf-8

# In[16]:


import numpy as np
from sklearn import datasets, svm

# In[2]:


# import iris data to model Svm classifier
iris = datasets.load_iris()

# In[3]:


X = iris.data[:, 2:]  # we only take the last two features.
y = iris.target

# In[6]:


C = 10.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)

# In[7]:


# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X, y)

# In[8]:


# LinearSVC
lin_svc.score(X, y)

# In[10]:


# SVC
svc.score(X, y)

# In[11]:


rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)

# In[12]:


rbf_svc

# In[13]:


poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

# In[14]:


poly_svc

# In[17]:


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
