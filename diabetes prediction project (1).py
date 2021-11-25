#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("diabetes.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.size


# In[5]:


data.describe()


# In[6]:


data.isnull()


# In[7]:


data.isnull().values.any()


# In[8]:


#correlation
#get correlations of each features in dataset
corrmat=data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[9]:


data.corr()


# In[10]:


data['Outcome'].value_counts()


# In[11]:


#seperating data and labels
X= data.drop(columns='Outcome',axis=1)
y= data["Outcome"]


# In[12]:


print(X)


# In[13]:


print(y)


# In[14]:


#scaling the data
from sklearn.preprocessing import scale
x=scale(X)


# In[15]:


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[16]:


#one hot encoding
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)


# In[17]:


#Definiing and desining neural network model
model=Sequential()
model.add(Dense(12, input_dim=8,activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="sigmoid"))


# In[18]:


print(model.summary())


# In[19]:


#compiling model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[20]:


#training model
history=model.fit(X_train, y_train, epochs=1000)


# In[21]:


#prediction and evalution
from sklearn.metrics import accuracy_score
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
accuracy_score(y_test,y_pred)


# In[22]:


history_dict=history.history
history_dict.keys()
np.array(history_dict["loss"]).shape
np.array(history_dict["accuracy"]).shape


# In[23]:


#ploting loss and accuracy against no. of epochs
loss=history_dict["loss"]
accuracy=history_dict["accuracy"]
epochs=np.arange(1000)
plt.plot(epochs,loss,c="r",label="Loss")
plt.plot(epochs,accuracy,c="g",label="Accur")
plt.plot()


# In[ ]:




