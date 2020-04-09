# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from matplotlib import rcParams
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# setting configs plot size 5x4 inches and seaborn style whitegrid
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

from sklearn.preprocessing import LabelEncoder

# In[2]:


# get the folder path
# %pwd


# In[3]:


hp = pd.read_csv("hp_data.csv")

# In[4]:


print("Shape is   :  ", hp.shape)
print()

print("No of cloumns  :  ", hp.columns)
print()

print("Print fisrt 1 items  :  ")
print(hp.head(1))
print()

print("Checking missing value   :  ")
print(hp.isnull().sum())

# In[5]:


enc = LabelEncoder()

# In[6]:


hp.place = enc.fit_transform(hp.place)
hp.sale = enc.fit_transform(hp.sale)
hp.built = enc.fit_transform(hp.built)

# In[7]:


print(hp.head(2))

# In[8]:


print(hp.describe)

# In[22]:


sb.regplot(x="sqft", y="price", data=hp)

# In[23]:


sb.regplot(x="yearsOld", y="price", data=hp)

# In[24]:


sb.regplot(x="floor", y="price", data=hp)

# In[25]:


sb.regplot(x="totalFloor", y="price", data=hp)

# In[26]:


sb.regplot(x="bhk", y="price", data=hp)

# In[27]:


sb.regplot(x="place", y="price", data=hp)

# In[28]:


hp.plot()

# In[29]:


hp.price.plot()

# In[30]:


# plot several variables
graph_data = hp.loc[:, ["sqft", "floor", "bhk", "totalFloor", "yearsOld"]]

color_theme = ['red', 'blue', 'yellow', 'green']
graph_data.plot(color=color_theme)

# In[9]:


print(hp.columns)

# In[10]:


plt.scatter((hp["sqft"] + hp["yearsOld"] + hp["floor"] + hp["totalFloor"] + hp["bhk"]), hp["price"])

# In[11]:


plt.scatter((hp["id"] + hp["place"] + hp["built"] + hp["sqft"] + hp["sale"] + hp["yearsOld"] + hp["floor"] + hp[
    "totalFloor"] + hp["bhk"]), hp["price"])

# In[12]:


plt.scatter((hp["place"] + hp["sqft"] + hp["yearsOld"] + hp["floor"] + hp["totalFloor"] + hp["bhk"]), hp["price"])

# In[13]:


# droping the sqft value , if sqrt > 3500
# hp = hp[hp["sqft"] < 3500]


# In[14]:


plt.scatter((hp["sqft"] + hp["yearsOld"] + hp["floor"] + hp["totalFloor"] + hp["bhk"]), hp["price"])

# In[15]:


hp.price.plot()

# In[16]:


hp.sqft.plot()

# In[19]:


print(hp.head(2))

# In[21]:


# creating simple bar using X and Y 
# plt.bar(hp.sqft,y)


# In[22]:


# feature engineering
x = hp.loc[:, ["id", "place", "built", "sqft", "sale", "yearsOld", "floor", "totalFloor", "bhk"]]
y = hp.price

# In[23]:


print(x.head(2))

# In[24]:


# print("X  :  " , x)
print()

# print("Y  :  ", y)
print()
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# print(y_train)
# print(x_train)

linear_regression_model = LinearRegression()
# print(linear_regression_model)
# print()

linear_regression_model.fit(x_train, y_train)

predict_x = linear_regression_model.predict(x_test)
# print(" predict_x :  " ,predict_x)

r2_score_accuracy = r2_score(y_test, predict_x)  # r2_score is standrad accuracy
print("linear rgression r2_score_accuracy : ", r2_score_accuracy)
# print()

mean_squared_error_accuracy = mean_squared_error(y_test, predict_x)
print("linear rgression mean_squared_error_accuracy : ", mean_squared_error_accuracy)
print()

# check the accuracy on the training set
print("Accuracy on training set: ", linear_regression_model.score(x_train, y_train))
# check the accuracy on the tensorbroad_pb_android set
print("Accuracy on tensorbroad_pb_android set: ", linear_regression_model.score(x_test, y_test))
print(x_test.head(10))

# In[25]:


decision_tree_regression_model = DecisionTreeRegressor()
decision_tree_regression_model.fit(x_train, y_train)

predict_x = decision_tree_regression_model.predict(x_test)
# print(" predict_x :  " ,predict_x)

r2_score_accuracy = r2_score(y_test, predict_x)  # r2_score is standrad accuracy
print("decision_tree_regression_model r2_score_accuracy : ", r2_score_accuracy)
print()

mean_squared_error_accuracy = mean_squared_error(y_test, predict_x)
print("decision_tree_regression_model mean_squared_error_accuracy : ", mean_squared_error_accuracy)
print()

# check the accuracy on the training set
print("Accuracy on training set: ", decision_tree_regression_model.score(x_train, y_train))
# check the accuracy on the tensorbroad_pb_android set
print("Accuracy on tensorbroad_pb_android set: ", decision_tree_regression_model.score(x_test, y_test))

# In[26]:


print(x_test.columns, decision_tree_regression_model.feature_importances_)

# In[27]:


random_forest_regression_model = RandomForestRegressor()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

random_forest_regression_model.fit(x_train, y_train)

predict_x = random_forest_regression_model.predict(x_test)
# print(" predict_x :  " ,predict_x)

r2_score_accuracy = r2_score(y_test, predict_x)  # r2_score is standrad accuracy
print("random_forest_regression_model r2_score_accuracy : ", r2_score_accuracy)
print()

mean_squared_error_accuracy = mean_squared_error(y_test, predict_x)
print("random_forest_regression_model mean_squared_error_accuracy : ", mean_squared_error_accuracy)
print()

# check the accuracy on the training set
print("Accuracy on training set: ", random_forest_regression_model.score(x_train, y_train))
# check the accuracy on the tensorbroad_pb_android set
print("Accuracy on tensorbroad_pb_android set: ", random_forest_regression_model.score(x_test, y_test))

# In[28]:


print("Training Features Shape:", x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# In[29]:


# print("x train is  :  ", x_train)
print()

# print("y train is  :  ",y_train)


# In[30]:


# print("x tensorbroad_pb_android is  :  ",x_test)
print()

# print("y tensorbroad_pb_android is  :  ",y_test)


# In[31]:


print("skew : ", stats.skew(hp.sqft))
print()

print("kurtosis : ", stats.kurtosis(hp.sqft))
print()

# print("coefficient : ",model.coef_)


# In[32]:


print("x train head : ")
print(x_train.head(1))

print("y train head :")
print(y_train.head(1))

# In[33]:


# print(model.predict([[1450, 1 , 1 ,2 , 3]]))
# print()

# print(model.predict([[1450, 1 , 0 ,2 , 1]]))
# print()

# print(model.predict([[1400, 1 , 0 ,2 , 1]]))
# print()

# print(model.predict([[1400, 1 , 1 ,2 , 3]]))


# In[34]:


names = ["Linear Regression",
         "Decision Tree Regression",
         "Random Forest Regression"
         ]

# In[35]:


algorithms = [LinearRegression(),
              DecisionTreeRegressor(),
              RandomForestRegressor()
              ]

# In[36]:


columns_name = ["Model_name", "Random_state", 'r2_score']

# In[37]:


rows = []


def addRandomStateForAlgorithm(x, y, names, algorithms, columns_name, random_state_list):
    for j in range(len(algorithms)):
        model = algorithms[j]
        for i in random_state_list:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=i)
            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)
            row = [names[j], i, r2_score(y_test, pred_test)]
            rows.append(row)
    models_df = pd.DataFrame(rows)
    models_df.columns = columns_name
    print(models_df)


# In[38]:


random_state_list_up_to_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random_state_list_10_up_to_20 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# In[39]:


addRandomStateForAlgorithm(x, y, names, algorithms, columns_name, random_state_list_up_to_10)

# In[40]:


# addRandomStateForAlgorithm(x, y,names,algorithms,columns_name,random_state_list_10_up_to_20)


# In[41]:


print(x.shape, hp.shape)

# In[42]:


print(hp.head())

# In[43]:


x = hp.drop(["id", "built", "sale", "price"], 1)
y = hp["price"]

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)

# In[45]:


print(x_test.head(10))

# In[46]:


lm1 = LinearRegression()
lm1.fit(x_train, y_train)
print(lm1.score(x_train, y_train))

# In[47]:


y_lm1 = lm1.predict(x_test)
print(r2_score(y_test, y_lm1))
