# coding: utf-8

# In[29]:


import pandas as pd
import pydotplus
from IPython.display import Image
from collections import Counter as c
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# differnece sklearn.cross_validation and sklearn.model_selection


# In[2]:


# local csv file for iris dataset_D
iris = pd.read_csv("iris_dataset.csv", ";")

# In[3]:


# iris ==> DESCR , data, target, feature_names
iris.shape

# In[4]:


iris.columns

# In[5]:


iris.isnull().sum()

# In[6]:


iris.head(2)

# In[7]:


enc = LabelEncoder()

# In[8]:


iris['names_new_encode'] = enc.fit_transform(iris['names'])

# In[9]:


iris.head(2)

# In[10]:


c(iris['names'])

# In[11]:


X = iris.iloc[:, 0:4]
y = iris.iloc[:, -1]

# In[23]:


model = DecisionTreeClassifier(max_depth=3)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# In[24]:


model.fit(x_train, y_train)
x_prediect = model.predict(x_test)
accuracy = accuracy_score(y_test, x_prediect)
accuracy

# In[25]:


# !conda install pydotplus -y


# In[22]:


# !conda install graphviz -y


# In[31]:


# Visualizing
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,
                filled=True, feature_names=X, class_names=['setosa', 'versi color', 'virginca'], rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
