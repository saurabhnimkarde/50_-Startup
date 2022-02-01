#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


turbine=pd.read_csv("gas_turbines.csv")


# In[3]:


turbine.head()


# In[4]:


turbine.isnull().sum()


# In[5]:


turbine.dtypes


# In[6]:


turbine.shape


# In[7]:


turbine_mean = turbine.TEY.mean()
turbine_mean


# In[38]:


#Inference : As the mean is 134.188, we will round it off to 134 and categorize the column


# In[8]:


turbine['performance'] = turbine.TEY.map(lambda x: 1 if x > 134 else 0)


# In[9]:


turbine.performance


# In[10]:


turbine.performance.value_counts()


# In[11]:


turbine


# In[12]:


turbine.drop(['TEY'],inplace=True,axis=1)


# In[13]:


turbine


# In[14]:


turbine.shape


# In[15]:


X = turbine.iloc[:,0:10]
Y = turbine.iloc[:,10]


# In[16]:


X


# In[17]:


Y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)


# In[21]:


# Now apply the transformations to the data:
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[22]:


from sklearn.neural_network import MLPClassifier


# In[23]:


mlp = MLPClassifier(hidden_layer_sizes=(30,30))


# In[24]:


mlp.fit(x_train,y_train)


# In[25]:


prediction_train=mlp.predict(x_train)
prediction_test = mlp.predict(x_test)


# In[26]:


prediction_test


# In[27]:


prediction_train


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)


# In[29]:


pd.crosstab(y_test,prediction_test)


# In[ ]:




