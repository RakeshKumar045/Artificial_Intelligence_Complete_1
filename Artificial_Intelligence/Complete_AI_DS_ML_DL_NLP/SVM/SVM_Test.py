# coding: utf-8

# In[10]:


from sklearn import datasets, svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# In[3]:


# import iris data to model Svm classifier
iris = datasets.load_iris()

# In[4]:


iris.feature_names

# In[5]:


iris.data.shape

# In[6]:


X = iris.data
y = iris.target

# In[7]:


X.shape

# In[8]:


# Split the data into tensorbroad_pb_android and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# In[9]:


# Linear Kernel

svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted = svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
print(accuracy_score(y_test, predicted))

# In[15]:


# y


# In[11]:


import pandas as pd

X_df = pd.DataFrame(X_train)
X_df.head()

# In[12]:


parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]},
              {'kernel': ['rbf'], 'gamma': [.05, 0.1, 0.06, .07, .08],
               'C': [1, 10, 100, 1000]}
              ]

# In[16]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_model_svc = GridSearchCV(SVC(), parameters)
grid_model_svc.fit(X_train, y_train)

grid_model_svc.best_score_

# In[17]:


grid_model_svc.best_params_

# In[18]:


import pandas as pd

wine = pd.read_csv('winequality-white.csv', ';')
wine.head()

# In[19]:


X = wine.loc[:, :'alcohol']
y = wine.quality
# Split the data into tensorbroad_pb_android and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_test.shape

# In[20]:


# Linear Kernel
from sklearn.metrics import accuracy_score

svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted = svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
print(accuracy_score(y_test, predicted))

# In[21]:


joblib.dump(svc_linear, 'trained_model_wine.ml')
