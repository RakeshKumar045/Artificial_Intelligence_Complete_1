# coding: utf-8

# In[15]:


from sklearn import datasets, svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# In[2]:


iris = datasets.load_iris()

# In[3]:


iris.feature_names

# In[4]:


iris.data.shape

# In[5]:


X = iris.data
y = iris.target

# In[6]:


X.shape

# <h3>Splitting the Dataset</h3>

# In[7]:


# Split the data into tensorbroad_pb_android and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# <h3>Applying Linear SVM</h3>

# In[8]:


# Linear Kernel

svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted = svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
print(accuracy_score(y_test, predicted))

# In[9]:


# In[10]:


import pandas as pd

X_df = pd.DataFrame(X_train)
X_df.head()

# In[11]:


parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]},
              {'kernel': ['rbf'], 'gamma': [.05, 0.1, 0.06, .07, .08],
               'C': [1, 10, 100, 1000]}
              ]

# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_model_svc = GridSearchCV(SVC(), parameters)
grid_model_svc.fit(X_train, y_train)

grid_model_svc.best_score_

# In[13]:


grid_model_svc.best_params_

# In[45]:


import pandas as pd

wine = pd.read_csv('/Users/ashok/Downloads/winequality-white.csv', ';')
wine.head()

# In[48]:


X = wine.loc[:, :'alcohol']
y = wine.quality

# In[49]:


# Split the data into tensorbroad_pb_android and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# In[50]:


X_test.shape

# In[51]:


# Linear Kernel
from sklearn.metrics import accuracy_score

svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted = svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
print(accuracy_score(y_test, predicted))

# In[52]:


joblib.dump(svc_linear, 'trained_model_wine.ml')

# In[43]:


get_ipython().run_line_magic('pwd', '')
