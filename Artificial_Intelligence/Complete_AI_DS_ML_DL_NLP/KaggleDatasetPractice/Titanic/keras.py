# coding: utf-8

# In[1]:


import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics

# ## Load Titanic data set

# In[2]:


train_df = pd.read_csv("titanic-train.csv")

# In[4]:


train_df.isnull().sum()

# In[13]:


train_df["Age"][5]

# In[3]:


print("dataset_D shape:", train_df.shape)

# In[4]:


train_df.head(5)

# In[14]:


train_df["Cabin"]


# ### Set 'CABIN' values to 'Yes' or 'No'. If value is NAN the set CABIN is NO else Yes

# In[15]:


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


# In[18]:


train_df.Cabin.isnull()

# In[16]:


dataset = set_Cabin_type(train_df)

# ### Convert 'CABIN' to dummy variable

# In[17]:


train_df["Cabin"]

# In[7]:


dataset = pd.get_dummies(dataset, columns=["Cabin"])
dataset.drop('Cabin_No', axis=1, inplace=True)

# ### Convert 'SEX' to dummy variable

# In[8]:


dataset = pd.get_dummies(dataset, columns=["Sex"])
dataset.drop('Sex_female', axis=1, inplace=True)

# ### Fill NA data with median values

# In[9]:


dataset["Age"].fillna(dataset["Age"].median(skipna=True), inplace=True)
dataset["Embarked"].fillna(dataset['Embarked'].value_counts().idxmax(), inplace=True)

# ### Convert dummy variables for 'Embarked' and 'Pclass'

# In[10]:


dataset = pd.get_dummies(dataset, columns=["Embarked"])
dataset = pd.get_dummies(dataset, columns=["Pclass"])

# In[11]:


dataset.head(3)

# ### Drop colums which do not add any information for prediction

# In[12]:


dataset.drop('Name', axis=1, inplace=True)
dataset.drop('PassengerId', axis=1, inplace=True)
dataset.drop('Ticket', axis=1, inplace=True)

# In[13]:


X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 0].values

# ### Split train and tensorbroad_pb_android using sklearn library

# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ###  Importing the Keras libraries and packages

# In[15]:


from keras.models import Sequential
from keras.layers import Dense

# In[16]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=20, epochs=100)

kernel_initializer - Defines
the
way
to
set
the
initial
random
weights
of
Keras
layers.
activation - activation
function is used
to
intoduce
non - linear
nature
to
data.
optimizer - Optimazation
function
like
Stocastic
GD, adam, RMSprop
etc
# In[17]:


y_pred = classifier.predict(X_test)

# In[18]:


y_pred1 = (y_pred > 0.5)
score = metrics.accuracy_score(y_test, y_pred1)
score
