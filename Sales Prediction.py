#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data=pd.read_csv("C:/Users/Dell/Desktop/intern/oasis/advertising/Advertising.csv")
data


# In[4]:


print(data.head())


# In[5]:


print(data.tail())


# In[6]:


data.isnull().sum()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[8]:


x=data[['TV','Radio','Newspaper']]
y=data['Sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65, random_state=0)


# In[9]:


print(x_train)


# In[10]:


print(y_train)


# In[11]:


print(x_test)


# In[12]:


print(y_test)


# In[13]:


import seaborn as sns


# In[14]:


sns.heatmap(data.corr(),annot=True)


# In[15]:


sns.lmplot(data=data,x="Radio",y="Sales")


# In[16]:


sns.lmplot(data=data,x="Newspaper",y="Sales")


# In[17]:


sns.lmplot(data=data,x="TV",y="Sales")


# In[18]:


model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[19]:


print(model.intercept_)
print(model.coef_)


# In[20]:


act_pred=pd.DataFrame({'Actual':y_test.values.flatten(),'Predict':y_pred.flatten()})
act_pred.head(20)


# In[21]:


sns.lmplot(data=act_pred,x='Actual',y="Predict")


# In[22]:


from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# In[23]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))


# In[24]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

