#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


data = load_breast_cancer()


# In[4]:


data.keys()


# In[7]:


df = pd.DataFrame(data['data'], columns = data['feature_names'])


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()


# In[14]:


sc.fit(df)


# In[15]:


sc_data = sc.transform(df)


# In[19]:


from sklearn.decomposition import PCA


# In[20]:


pca = PCA(n_components=2)
pca.fit(sc_data)
final_data = pca.transform(sc_data)


# In[21]:


final_data.shape


# In[ ]:




