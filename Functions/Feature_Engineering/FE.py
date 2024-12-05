#!/usr/bin/env python
# coding: utf-8

# ---
# # Import Libraries

# In[1]:


import pandas as pd
import pandas_ta as pta
import numpy as np


# ---
# ### Feature Engineering
# * Log Returns
# * Moving Average
# * Rolling Standar Diviation
# * Spread

# In[2]:


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
        Parameter(name_of_column): LIST of dataframe column name
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
        Parameter(name_of_column): LIST of dataframe column name
        Parameter(ma_number): Number of Moving Average (Slow - Fast)
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
    def Rolling_Standar_Diviation(self, name_of_column, rsd_number):
        '''
        This Function of Feature_Engineering return Dataframe with Rolling Standar Diviation Return
        -Short-Term Trends Rolling Window Size: 10–20 periods
        -Long-Term Trends Rolling Window Size: 100–200 periods
        ------------------------------
        Parameter(name_of_column): LIST of dataframe column name
        Parameter(rsd_number): Number of Rolling Standar Diviation
        ------------------------------
        '''
        
        try:
            new_dataframe = self.data.copy()
        
            for _ in range(len(name_of_column)):
                new_dataframe[name_of_column[_]+'_RSD_'+str(rsd_number)] = new_dataframe[name_of_column[_]].rolling(rsd_number).std()
            return new_dataframe
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#    
    def Relative_Strength_Index(self, name_of_column, rsi_number):
        '''
        This Function of Feature_Engineering return Dataframe with Rolling Standar Index Return
        -Buy: RSI < 10 & z = 1
        -Sell: RSI > 60 & z = 0
        
        -Overbought: RSI > 70 indicates the asset may be overbought and could be due for a pullback.
        -Oversold: RSI < 30 suggests the asset might be oversold and could be due for a rebound.
        
        -RSI > 50 suggests bullish momentum.
        -RSI < 50 suggests bearish momentum.
        ------------------------------
        Parameter(name_of_column): LIST of dataframe column name
        Parameter(rsi_number): Number of Rolling Standar Index
        ------------------------------
        '''
        
        try:
            new_dataframe = self.data.copy()
        
            for _ in range(len(name_of_column)):
                new_dataframe[name_of_column[_]+'_RSI_'+str(rsi_number)] = pta.rsi(new_dataframe[name_of_column[_]], length = rsi_number)
            return new_dataframe
        except:
            print('No option!\nError')
#-------------------------------------------------------------------------------------------------------------------------------


# In[ ]:





# In[ ]:




