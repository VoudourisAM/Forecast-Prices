#!/usr/bin/env python
# coding: utf-8

# ---
# # Import Libraries

# In[1]:


import pandas as pd


# ---
# ### Function

# In[1]:


def Save_CSV_File(dataframe, name_of_csv):
    '''
    This function save a dataframe as csv file
    ------------------------------
    Parameter(dataframe): Name of dataframe
    Parameter(name_of_csv): Name of csv file
    ------------------------------
    '''
    
    dataframe.to_csv('../Forecast_Prices/Data/Binance_Data_Crypto/'+name_of_csv+'.csv') 


# ---

# In[ ]:




