# coding: utf-8

# # 1) Straight Line Graph : (plt.plot(X,Y))

# # 2) Scatter Graph : plt.scatter(x,y)

# # 3) Bar Graph : plt.bar(x, y)
# 

# # 4) Histogram Graph : plt.hist(X, cumulative=True, bins=20) 
#     only X is required 
#     it needs to single value

# # 5 Pie chart Graph : plt.pie(X)
# It need to single value(x)
# 

# # 6) Fill Graph : plt.fill(X,Y)

# # 7) Histogram 2d Graph : plt.hist2d(X, Y) 

# # 8) Area Plot or StackPlot Graph : plt.stackplot(X, Y)
# 

# In[6]:


import numpy as np

from matplotlib import pyplot as plt

# In[43]:


plt.plot([1, 2, 3, 4, 5, 6, 7])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# In[31]:


x = [1, 2, 3, 4]
y = [5, 6, 7, 2]

plt.plot(x, y, linewidth=10.0, color="red")
plt.xlabel("X-axis", color="b")
plt.ylabel("Y-axis", color="g")
plt.title("Practice_key_points_D", color="r")
plt.show()

# In[32]:


x = [1, 2, 3, 4]
y = [5, 6, 7, 2]

plt.scatter(x, y, linewidth=1.0, color="yellow", label="example info)
plt.xlabel("X-axis", color="b")
plt.ylabel("Y-axis", color="g")
plt.title("Practice_key_points_D", color="r")
plt.show()

# In[35]:


x = [1, 2, 3, 4]
y = [5, 6, 7, 2]

plt.bar(x, y, linewidth=1.0, color="yellow")
plt.xlabel("X-axis", color="b")
plt.ylabel("Y-axis", color="g")
plt.title("Practice_key_points_D", color="r")
plt.show()

# In[36]:


x = [1, 2, 3, 4, 3, 6, 8, 9]
y = [5, 6, 7, 2, 7, 2, 4, 1]

plt.bar(x, y, linewidth=1.0, color="yellow")
plt.xlabel("X-axis", color="b")
plt.ylabel("Y-axis", color="g")
plt.title("Practice_key_points_D", color="r")
plt.show()

# # 1) Matplot Combine(X1,Y1 and X2,Y2)

# In[48]:


x = [1, 2, 3, 4]
y = [5, 6, 7, 2]

x1 = [3, 6, 9, 8]
y1 = [5, 6, 7, 2]

plt.plot(x, y, label="First", color="yellow")
plt.plot(x1, y1, label="Second", color="green")
plt.xlabel("X-axis", color="b")
plt.ylabel("Y-axis", color="g")
plt.title("Practice_key_points_D", color="r")
plt.show()

# # 2) With Line

# In[36]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
plt.plot(x, y1, label="line L")
# plt.plot(x, y2, label="line H")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()

# # Without Line

# In[4]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
plt.plot(x, y1)
# plt.plot(x, y2, label="line H")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()

# In[9]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
# plt.plot(x, y1, label="line L")
plt.plot(x, y2, label="it is testing line , if line label statement is large , so label will display in center")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()

# # 3) Bar Graph Combine(X1 , Y1 and X2, Y2)

# In[37]:


# Look at index 4 and 6, which demonstrate overlapping cases.
x1 = [1, 3, 4, 5, 6, 7, 9]
y1 = [4, 7, 2, 4, 7, 8, 3]

x2 = [2, 4, 6, 8, 10]
y2 = [5, 6, 2, 6, 2]

# Colors: https://matplotlib.org/api/colors_api.html

plt.bar(x1, y1, label="Blue Bar", color='b')
plt.bar(x2, y2, label="Green Bar", color='g')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()

# # 4) Bar Graph Single Display

# In[12]:


# Look at index 4 and 6, which demonstrate overlapping cases.
x1 = [1, 3, 4, 5, 6, 7, 9]
y1 = [4, 7, 2, 4, 7, 8, 3]

plt.bar(x1, y1, label="Blue Bar", color='r')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()

# In[13]:


# Look at index 4 and 6, which demonstrate overlapping cases.

x2 = [2, 4, 6, 8, 10]
y2 = [5, 6, 2, 6, 2]

plt.bar(x2, y2, label="Green Bar", color='g')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()

# # 5) Histograms

# In[16]:


# Use numpy to generate a bunch of random data in a bell curve around 5.
n = 5 + np.random.randn(1000)

m = [m for m in range(len(n))]
plt.bar(m, n)
plt.title("Raw Data")
plt.show()

plt.hist(n, bins=20)
plt.title("Histogram")
plt.show()

plt.hist(n)
plt.title("Histogram for without Cumulative and bins")
plt.show()

plt.hist(n, cumulative=True, bins=20)
plt.title("Cumulative Histogram for n")
plt.show()

plt.hist(m, cumulative=True, bins=20)
plt.title("Cumulative Histogram for m")
plt.show()

plt.plot(m, n)
plt.title("Plot Graph")
plt.show()

# print("M = {}\n".format(m))
# print("\nN = \n",n)


# # 6 Scatter Plots

# In[27]:


x1 = [2, 3, 4]
y1 = [5, 5, 5]

x2 = [1, 2, 3, 4, 5]
y2 = [2, 3, 2, 3, 4]
y3 = [6, 8, 7, 8, 7]

# Markers: https://matplotlib.org/api/markers_api.html

plt.scatter(x1, y1)
plt.scatter(x2, y2, marker='v', color='r')
plt.scatter(x2, y3, marker='^', color='m')
plt.title('Scatter Plot Example')
plt.show()

# # 7 Stack Plot

# In[17]:


idxes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
arr1 = [23, 40, 28, 43, 8, 44, 43, 18, 17]
arr2 = [17, 30, 22, 14, 17, 17, 29, 22, 30]
arr3 = [15, 31, 18, 22, 18, 19, 13, 32, 39]

# Adding legend for stack plots is tricky.
plt.plot([], [], color='r', label='D 1')
plt.plot([], [], color='g', label='D 2')
plt.plot([], [], color='b', label='D 3')

plt.stackplot(idxes, arr1, arr2, arr3, colors=['r', 'g', 'b'])
plt.title('Stack Plot Example')
plt.legend()
plt.show()

# In[30]:


idxes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
arr1 = [23, 40, 28, 43, 8, 44, 43, 18, 17]

# Adding legend for stack plots is tricky.
plt.plot([], [], color='yellow', label='D 1')

plt.stackplot(idxes, arr1, colors=['r'])
plt.title('Stack Plot Example')
plt.legend()
plt.show()

# In[31]:


import sys

print('Hello, Colaboratory from Python {}!'.format(sys.version_info[0]))

# In[32]:


import tensorflow as tf

tf.test.gpu_device_name()

# # 9 Pie chart

# In[33]:


import matplotlib.pyplot as plt

labels = 'S1', 'S2', 'S3'
sections = [56, 66, 24]
colors = ['c', 'g', 'y']

plt.pie(sections, labels=labels, colors=colors,
        startangle=90,
        explode=(0, 0.1, 0),
        autopct='%1.2f%%')

plt.axis('equal')  # Try commenting this out.
plt.title('Pie Chart Example')
plt.show()

# In[35]:


# import tensorflow as tf
# import numpy as np

# with tf.Session():
#     input1 = tf.constant(1.0, shape=[2, 3])
#     input2 = tf.constant(np.reshape(np.arange(1.0, 7.0, dtype=np.float32), (2, 3)))
#     output = tf.add(input1, input2)
#     result = output.eval()

# result


# In[7]:


x = np.arange(20)
y = [x_i + np.random.randn(1) for x_i in x]
a, b = np.polyfit(x, y, 1)
_ = plt.plot(x, y, 'o', np.arange(20), a * np.arange(20) + b, '-')

# In[33]:


# !pip install -q matplotlib-venn


# In[34]:


# Now the newly-installed library can be used anywhere else in the notebook.
# Only needs to be run once at the top of the notebook.
# !pip install -q matplotlib-venn
_ = venn2(subsets=(3, 2, 1))

# # 10 Fill
# You can plot multiple polygons by providing multiple x, y, [color] groups.
# 
# 

# In[39]:


x1 = [1, 3, 4, 5, 6, 7, 9]
y1 = [4, 7, 2, 4, 7, 8, 3]

x2 = [2, 4, 6, 8, 10]
y2 = [5, 6, 2, 6, 2]

plt.fill(x1, y1, label="Blue Bar", color='b')
plt.fill(x2, y2, label="Green Bar", color='g')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()

# # 11) Fill and Bar combine

# In[40]:


x1 = [1, 3, 4, 5, 6, 7, 9]
y1 = [4, 7, 2, 4, 7, 8, 3]

x2 = [2, 4, 6, 8, 10]
y2 = [5, 6, 2, 6, 2]

plt.fill(x1, y1, label="Blue Bar", color='b')
plt.bar(x2, y2, label="Green Bar", color='g')
plt.plot()

plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar Chart Example")
plt.legend()
plt.show()

# In[41]:


np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

plt.show()

# In[48]:


# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

# In[54]:


# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');

# In[55]:

# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
