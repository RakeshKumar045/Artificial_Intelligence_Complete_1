# coding: utf-8

# In[3]:


import seaborn as sb
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression

# setting configs plot size 5x4 inches and seaborn style whitegrid
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

import pandas as pd

from sklearn.metrics import confusion_matrix
from collections import Counter as c
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

# differnece sklearn.cross_validation and sklearn.model_selection

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# In[4]:


data = pd.read_csv("bank-full_25thAug.csv", ";")

# In[5]:


print(data.shape)
print(data.columns)

# In[6]:


print(data.head(2))

# In[10]:


print(data.info())

# In[12]:


# data.isnull().sum()


# In[11]:


enc = LabelEncoder()

# In[14]:


data["job"] = enc.fit_transform(data["job"])
data["marital"] = enc.fit_transform(data["marital"])
data["education"] = enc.fit_transform(data["education"])
data["default"] = enc.fit_transform(data["default"])
data["housing"] = enc.fit_transform(data["housing"])
data["loan"] = enc.fit_transform(data["loan"])
data["contact"] = enc.fit_transform(data["contact"])

data["month"] = enc.fit_transform(data["month"])
data["poutcome"] = enc.fit_transform(data["poutcome"])
data["y"] = enc.fit_transform(data["y"])

# In[11]:


c(data)

# In[18]:


(data.iloc[:, :-1])  # (data.iloc[:,:16])

# In[19]:


# data.iloc[:,-1]


# In[26]:


data.iloc[0:2, :]  # data.iloc[0:2]

# In[27]:


data.iloc[0:2, :3]

# In[28]:


data.iloc[0:2, 3]

# In[30]:


data.iloc[2:2, 3:]

# In[31]:


data.iloc[0:2, 3:3]

# In[16]:


data.head(2)

# In[32]:


data[]

# In[33]:


data[:]

# In[24]:


# plt.scatter((data.iloc[1]) ,data["y"])


# In[32]:


# feature engineering
x = data.iloc[:, :16]
y = data.iloc[:, -1]

# In[33]:


x.head(2)

# In[39]:


# print("X  :  " , x)
print()

# print("Y  :  ", y)
print()
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# print(y_train)
# print(x_train)

linear_regression_model = LogisticRegression()
# print(linear_regression_model)
# print()

linear_regression_model.fit(x_train, y_train)

predict_x = linear_regression_model.predict(x_test)
# print(" predict_x :  " ,predict_x)

accuracy = accuracy_score(y_test, predict_x)  # r2_score is standrad accuracy
print("accuracy : ", accuracy)

# In[45]:


print("Training Features Shape:", x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# In[69]:


names = ["Logistic Regression",
         "Decision Tree Classification",
         "KNN",
         "RandomForestClassifier"
         ]

# In[70]:


algorithms = [LogisticRegression(),
              DecisionTreeClassifier(),
              KNN(n_neighbors=3),
              RandomForestClassifier()
              ]

# In[71]:


columns_name = ["Model_name", "Random_state", 'accuracy_score']
random_state_list_up_to_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# In[93]:


rows = []


def addRandomStateForAlgorithm(x, y, names, algorithms, columns_name, random_state_list):
    for j in range(len(algorithms)):
        model = algorithms[j]
        for i in random_state_list:
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=i)
            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)
            row = [names[j], i, accuracy_score(y_test, pred_test)]
            rows.append(row)
    models_df = pd.DataFrame(rows)
    models_df.columns = columns_name
    print(models_df)


# In[94]:


addRandomStateForAlgorithm(x, y, names, algorithms, columns_name, random_state_list_up_to_10)

# In[95]:


c(data.y)

# In[66]:


c(data)

# In[91]:


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=3)

# In[97]:


# right format
confusion_matrix(y_test, predict_x)

# In[98]:


# wrong format
confusion_matrix(predict_x, y_test)
