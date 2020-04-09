# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_breast_cancer
##from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()

# In[2]:


pd.DataFrame(cancer.target).describe()

# In[3]:


cancer.feature_names

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=10)
tree = DecisionTreeClassifier(random_state=15)
###Decision trees in scikit-learn are implemented in the DecisionTreeRegressor 
##and DecisionTreeClassifier classes. Scikit-learn only implements pre-pruning, not post- pruning.
tree.fit(X_train, y_train)
print("accuracy on training set: %f" % tree.score(X_train, y_train))
print('\n'"accuracy on tensorbroad_pb_android set: %f" % tree.score(X_test, y_test))
tree

# In[5]:


###apply pre-pruning to the tree, which will stop developing the tree before we
### perfectly fit to the training data.
tree01 = DecisionTreeClassifier(max_depth=2, random_state=15)
tree01.fit(X_train, y_train)
print('\n'"accuracy on training set 01: %f" % tree01.score(X_train, y_train))
print('\n'"accuracy on tensorbroad_pb_android set 01: %f" % tree01.score(X_test, y_test))

# In[6]:


###visualize and analyze the tree model###
###build a file to visualize 
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="mytree.dot", class_names=['malignant', "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
###visualize the .dot file. Need to install graphviz seperately at first 
import graphviz

with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# In[7]:


1 - (54 / 100) ** 2 - (46 / 100) ** 2

# In[8]:


from sklearn.cross_validation import cross_val_score

# In[9]:


scores = cross_val_score(tree01, X_train, y_train, cv=5)

# In[10]:


scores

# In[11]:


scores.mean()
