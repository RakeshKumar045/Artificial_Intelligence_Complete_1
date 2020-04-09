# coding: utf-8

# In[ ]:


# !conda install pydotplus -y


# In[1]:


# !conda install graphviz -y


# In[1]:


# importing necessary packages
# importing necessary packages
import pandas as pd
import pydotplus
import sklearn.datasets as datasets
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# In[2]:


# loading the data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# In[3]:


X.head()

# In[10]:


# Defining and fitting
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(X, y)

# In[11]:


# dtree.predict([[5,3,1.5,0.4]])


# In[12]:


# Visualizing
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, feature_names=iris.feature_names, class_names=['setosa', 'versi color', 'virginca'],
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
