# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# In[2]:


breast_cancer = pd.read_csv('breast_cancer.csv')

# In[3]:


breast_cancer.head()

# In[4]:


breast_cancer.info()

# In[5]:


breast_cancer.shape

# In[6]:


data = breast_cancer.loc[0: 568, :]  # or  breast_cancer.dropna(inplace = true)

# In[7]:


data.tail(5)

# In[8]:


data.shape

# In[9]:


# data["outcome"].dropna()


# In[10]:


# data["mean radius"].dropna()


# In[11]:


# data["mean concave points"].dropna()


# In[12]:


# data["mean perimeter"].dropna()


# In[13]:


data.columns

# In[97]:


# feature engineering
# x = data.iloc[:,1:-1]
x = data.loc[:, ["mean radius", "mean perimeter", "mean smoothness"]]
y = data.outcome

# In[98]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# In[99]:


model = linear_model.LogisticRegression()

# In[100]:


model.fit(x_train, y_train)

# In[101]:


predict_x = model.predict(x_test)
predict_x

# In[145]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# In[146]:


accuracy_score(y_test, predict_x)

# In[147]:


confusion_matrix(y_test, predict_x)

# In[148]:


from collections import Counter as c

# In[149]:


c(y_test)

# In[150]:


c(y_train)

# In[151]:


print(classification_report)

# In[152]:


print(classification_report(y_test, predict_x))

# In[153]:


print(f1_score(y_test, predict_x))

# In[154]:


from sklearn.preprocessing import scale, StandardScaler

# In[155]:


x = scale(x)
x

# In[156]:


x[1:5]

# In[157]:


# Scaling the data is 2 types 1) normalization and 2) Standard Dev
scaling = StandardScaler()
scaling.fit(x)
x = scaling.fit_transform(x)
# x_new= scaling.fit_transform(x_new)
# x = scale(x)

x
