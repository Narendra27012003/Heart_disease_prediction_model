#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dependencies


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


#Data collection and processing 


# In[4]:


#loading the csv data into pandas Dataframe
heart_data=pd.read_csv("C:\\Users\\nb508\\Desktop\\heart_disease_data (2).csv")


# In[5]:


#print 5 rows of the dataset
heart_data.head()


# In[6]:


# print last 5 rows of the dataset
heart_data.tail()


# In[7]:


# number of rows and columns in the dataset
heart_data.shape


# In[8]:


# getting some info about the data  and  checking for missing values
heart_data.info()


# In[9]:


# statistical measures about the data
heart_data.describe()


# In[10]:


# checking the distribution of Target Variable
heart_data['target'].value_counts()


# In[11]:


#1 --> Defective Heart

#0 --> Healthy Heart


# In[12]:


#Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)


# In[13]:


print(X)


# In[14]:


Y = heart_data['target']
print(Y)


# In[15]:


#Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[16]:


print(X.shape, X_train.shape, X_test.shape)


# In[17]:


#Model Training
#Logistic Regression


# In[18]:


model = LogisticRegression()


# In[19]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# In[20]:


#Model Evaluation

#Accuracy Score


# In[21]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[22]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[23]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[24]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[26]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,13)
#or
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# In[ ]:





# In[ ]:




