#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


data = pd.read_csv("~/Desktop/PS_20174392719_1491204439457_log.csv")


# In[5]:


print(data.head())


# In[6]:


print(data.isnull().sum())


# In[7]:


print(data.type.value_counts())


# In[8]:


type = data["type"].value_counts()


# In[9]:


type


# In[10]:


transactions = type.index
transactions


# In[11]:


quantity = type.values
quantity


# In[16]:


import plotly.express as px


# In[17]:


figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()


# In[21]:


data_types = data.dtypes
data_types


# In[26]:


columns_to_omit = ['type','nameOrig','nameDest'] 
new_data = data.drop(columns=columns_to_omit)


# In[28]:


correlation = new_data.corr()
correlation


# In[29]:


print(correlation["isFraud"].sort_values(ascending=False))


# In[30]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})


# In[31]:


data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})


# In[32]:


print(data.head())


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])


# In[35]:


y = np.array(data[["isFraud"]])


# In[36]:


from sklearn.tree import DecisionTreeClassifier


# In[37]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)


# In[38]:


model = DecisionTreeClassifier()


# In[41]:


model.fit(xtrain, ytrain)


# In[40]:


print(model.score(xtest, ytest))


# In[56]:


features = np.array([[4, 9000.60, 9000.60, 0.0]])


# In[57]:


print(model.predict(features))


# In[ ]:




