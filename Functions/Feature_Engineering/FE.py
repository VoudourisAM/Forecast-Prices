#!/usr/bin/env python
# coding: utf-8

# ---
# # Import Libraries

# In[3]:


import pandas as pd
import numpy as np


# ---
# ### Feature Engineering
# * Log Returns
# * Moving Average
# * Rolling Standar Diviation
# * Spread

# In[4]:


class Feature_Engineering:
    '''
    This class generate new Dataframe of Feature Engineering
    '''
    
    #Constuctor
    def __init__(self, dataframe):
        '''
        Constructor of Feature_Engineering class
        ------------------------------
        Parameter(dataframe): DataFrame
        ------------------------------
        '''
        
        self.data = dataframe
        
#------------------------------------------------------------------------------------------------------------------------------#        
    def Log_Return(self, name_of_column):
        '''
        This Function of Feature_Engineering return Dataframe with Log Return
        ------------------------------
        Parameter(dataframe): DataFrame
        Parameter(name_of_column): LIST (One OR More) of dataframe column name 
        ------------------------------
        '''
        
        try:
            new_dataframe = self.data.copy()

            for _ in range(len(name_of_column)):
                new_dataframe[name_of_column[_]+'_Lg'] = np.log(new_dataframe[name_of_column[_]] / new_dataframe[name_of_column[_]].shift(1))
            return new_dataframe
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#    
    def Moving_Average(self, name_of_column, ma_number):
        '''
        This Function of Feature_Engineering return Dataframe with (Slow or Fast) Moving Average Return
        -Fast Moving Average (FMA): Used for short-term trends, typically with smaller periods like 9, 12, or 20.
        -Slow Moving Average (SMA): Used for long-term trends, typically with larger periods like 50, 100, or 200.
        ------------------------------
        Parameter(name_of_column): LIST (One OR More) of dataframe column name
        Parameter(ma_number): Number of Moving Average
        ------------------------------
        '''
        
        try:
            new_dataframe = self.data.copy()
        
            for _ in range(len(name_of_column)):
                if (ma_number >=5 and ma_number <= 25):
                    new_dataframe[name_of_column[_]+'_FMA_'+str(ma_number)] = new_dataframe[name_of_column[_]].rolling(window = ma_number).mean()
                elif (ma_number >=40 and ma_number <= 250):
                    new_dataframe[name_of_column[_]+'_SMA_'+str(ma_number)] = new_dataframe[name_of_column[_]].rolling(window = ma_number).mean()
            return new_dataframe
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#    


# In[ ]:





# In[ ]:




