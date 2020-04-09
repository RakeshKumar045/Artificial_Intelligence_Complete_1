# coding: utf-8

# # Logistic Regression

# In[2]:


import pandas as pd
from collections import Counter as c
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

enc = LabelEncoder()
from imblearn.over_sampling import SMOTE

# In[3]:


# In[4]:


from pylab import rcParams
import seaborn as sb

# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# In[7]:


# CAR car acceptability 
# . PRICE overall price 
# . . buying buying price 
# . . maint price of the maintenance 
# . TECH technical characteristics 
# . . COMFORT comfort 
# . . . doors number of doors 
# . . . persons capacity in terms of persons to carry 
# . . . lug_boot the size of luggage boot 
# . . safety estimated safety of the car 


# find columns name

# Attribute Information:

# Class Values: 

# unacc, acc, good, vgood 

# Attributes: 

# buying: vhigh, high, med, low. 
# maint: vhigh, high, med, low. 
# doors: 2, 3, 4, 5more. 
# persons: 2, 4, more. 
# lug_boot: small, med, big. 
# safety: low, med, high. 


# In[9]:


# Setting up the plot
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# In[10]:


data = pd.read_csv('cars.csv')

# In[11]:


columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "outcome"]

# In[12]:


data.head(1)

# In[13]:


# cars.isnull().sum()


# In[14]:


data.columns

# In[15]:


data.columns = columns

# In[16]:


data.head(1)

# In[17]:


data.shape

# In[18]:


c(data)

# In[38]:


type(data.buying)
print(len(data.buying))


# In[39]:


def checkCounterType(listOFData):
    for i in listOFData:
        print(i, len(i))


# In[40]:


counterList = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "outcome"]
checkCounterType(data.columns)

# In[16]:


c(data.buying)

# In[17]:


c(data.maint)

# In[18]:


c(data.doors)

# In[19]:


data.doors.replace('5more', '5', inplace=True)
c(data.doors)

# In[20]:


c(data.persons)

# In[21]:


data.persons.replace('more', '5', inplace=True)
c(data.persons)

# In[22]:


c(data.lug_boot)

# In[23]:


c(data.safety)

# In[24]:


c(data.outcome)

# In[93]:


data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)
data.outcome = enc.fit_transform(data.outcome)

# In[94]:


data.shape

# In[95]:


data.head(1)

# In[96]:


x = data.iloc[:, :6]
y = data.iloc[:, -1]

# In[97]:


print(x.shape)
print(y.shape)
print(x.columns)
# print(data.iloc[:,-1])


# In[98]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# In[99]:


# Defining and fitting LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(x_train, y_train)

# In[100]:


# Prediction
y_pred = LogReg.predict(x_test)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

# In[101]:


model = KNeighborsClassifier()
model.fit(x_train, y_train)
# Prediction
y_pred1 = model.predict(x_test)
# print(classification_report(y, y_pred1))
# print(metrics.accuracy_score(y, y_pred1))


# In[102]:


pd.crosstab(y_test, y_pred1)

# In[103]:


print(classification_report(y_test, y_pred1))
print(metrics.accuracy_score(y_test, y_pred1))

# In[104]:


# !pip install imblearn


# In[ ]:


# In[105]:


smote_res = SMOTE()
x_train_res, y_train_res = smote_res.fit_sample(x_train, y_train)
c(y_train_res)

# In[106]:


x_train_res

# In[107]:


model_xgboost = XGBClassifier()

# In[108]:


x_train.columns

# In[109]:


x_train.doors.dtypes

# In[110]:


x_train.buying.dtypes

# In[111]:


x_train.maint.dtypes

# In[112]:


x_train.doors.dtypes

# In[113]:


x_train.doors = x_train.doors.astype(int)
x_train.doors.dtype

# In[114]:


x_train.persons.dtypes

# In[115]:


x_train.persons = x_train.persons.astype(int)
x_train.persons.dtypes

# In[116]:


x_train.lug_boot.dtypes

# In[117]:


x_train.safety.dtypes

# In[118]:


model_xgboost.fit(x_train, y_train)

# In[119]:


x_test.columns

# In[120]:


x_test.buying.dtypes

# In[121]:


x_test.maint.dtypes

# In[122]:


x_test.doors.dtypes

# In[123]:


x_test.doors = x_test.doors.astype(int)
x_test.doors.dtypes

# In[124]:


x_test.persons.dtypes

# In[125]:


x_test.persons = x_test.persons.astype(int)
x_test.persons.dtypes

# In[126]:


x_test.lug_boot.dtypes

# In[127]:


x_test.safety.dtypes

# In[139]:


predict_xgboost = model_xgboost.predict(x_test)

# In[140]:


print(classification_report(y_test, predict_xgboost))
print(metrics.accuracy_score(y_test, predict_xgboost))

# In[141]:


pd.crosstab(y_test, predict_xgboost)

# In[149]:


# model_xgboost_reg = XGBRegressor()
# model_xgboost_reg.fit(x_train_res , y_train_res)
# predict_smote = model_xgboost_reg.predict(x_test)


# In[147]:


# model_xgboost_reg.fit(x_train_res , y_train_res)


# In[27]:


# predict_smote = model_xgboost_reg.predict(x_test)
