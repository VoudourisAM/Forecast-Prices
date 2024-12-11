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

# In[3]:


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
    def Log_Return(self, column_name):
        '''
        This Function of Feature_Engineering return Dataframe with Log
        ------------------------------
        Parameter(dataframe): DataFrame
        Parameter(column_name): LIST of dataframe column name
        ------------------------------
        '''
        
        try:
            for _ in range(len(column_name)):
                self.data[column_name[_]+'_Lg'] = np.log(self.data[column_name[_]] / self.data[column_name[_]].shift(1))
            return self.data
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#    
    def Moving_Average(self, column_name, ma_number):
        '''
        This Function of Feature_Engineering return Dataframe with (Slow or Fast) Moving Average
        - For short-term trading: Use shorter MAs (e.g., 5-day and 20-day).
        - For long-term investing: Use longer MAs (e.g., 50-day and 200-day).
        - Fast Moving Average (FMA): Used for short-term trends, typically with smaller periods like 9, 12, or 20.
        - Slow Moving Average (SMA): Used for long-term trends, typically with larger periods like 50, 100, or 200.
        - 1. Buy Signal Occurs when the Fast Moving Average crosses above the Slow Moving Average.
        - 2. Sell Signal Occurs when the Fast Moving Average crosses below the Slow Moving Average.
        ------------------------------
        Parameter(column_name): LIST of dataframe column name
        Parameter(ma_number): Number of Moving Average (Slow - Fast)
        ------------------------------
        '''
        
        try:
            for _ in range(len(column_name)):
                if (ma_number >=5 and ma_number <= 20):
                    self.data[column_name[_]+'_FMA_'+str(ma_number)] = self.data[column_name[_]].rolling(window = ma_number).mean()
                elif (ma_number >=25 and ma_number <= 200):
                    self.data[column_name[_]+'_SMA_'+str(ma_number)] = self.data[column_name[_]].rolling(window = ma_number).mean()
            return self.data
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#
    def Exponential_Moving_Average(self, column_name, ema_number):
        '''
        This Function of Feature_Engineering return Dataframe with Exponential Moving Average
        - Short-term (Fast EMA): Use 5, 7, 10, or 20 periods.
        - Medium-term: Use 50 periods.
        - Long-term (Slow EMA): Use 100 or 200 periods.
        (Use the Same Periods of Slow/Fast Moving Average (or little slow))
        ------------------------------
        Parameter(name_of_column): LIST of dataframe column name
        Parameter(ema_number): Number of EMA
        ------------------------------
        '''
        
        try:
            for _ in range(len(column_name)):
                if (ema_number >=5 and ema_number <= 20):
                    self.data[column_name[_]+'_FEMA_'+str(ema_number)] = self.data[column_name[_]].ewm(span=ema_number, adjust=False).mean()
                elif (ema_number >=25 and ema_number <= 200):
                    self.data[column_name[_]+'_SEMA_'+str(ema_number)] = self.data[column_name[_]].ewm(span=ema_number, adjust=False).mean()
            return self.data
        except:
            print('No option!\nError')
#------------------------------------------------------------------------------------------------------------------------------#    
    def Rolling_Standar_Diviation(self, column_name, rsd_number):
        '''
        This Function of Feature_Engineering return Dataframe with Rolling Standar Diviation
        - Short-Term Trends Rolling Window Size: 10–20 periods
        - Long-Term Trends Rolling Window Size: 100–200 periods
        ------------------------------
        Parameter(column_name): LIST of dataframe column name
        Parameter(rsd_number): Number of Rolling Standar Diviation
        ------------------------------
        '''
        
        try:
            for _ in range(len(column_name)):
                self.data[column_name[_]+'_RSD_'+str(rsd_number)] = self.data[column_name[_]].rolling(rsd_number).std()
            return self.data
        except:
            print('No option!\nError')
#-------------------------------------------------- Momentum-Based Strategies -------------------------------------------------#    
    def Relative_Strength_Index(self, column_name, rsi_number):
        '''
        This Function of Feature_Engineering return Dataframe with Rolling Standar Index
        - Buy: RSI < 10 & z = 1
        - Sell: RSI > 60 & z = 0
        - rsi_number Shorter Periods 7 or 9 or 14
        - rsi_number Longer Periods 21
        
        - Overbought: RSI > 70 indicates the asset may be overbought and could be due for a pullback.
        - Oversold: RSI < 30 suggests the asset might be oversold and could be due for a rebound.
        
        - RSI > 50 suggests bullish momentum.
        - RSI < 50 suggests bearish momentum.
        ------------------------------
        Parameter(column_name): LIST of dataframe column name
        Parameter(rsi_number): Number of Rolling Standar Index
        ------------------------------
        '''
        try:         
            if len(column_name) == 1 and (rsi_number > 0):
                for _ in range(len(column_name)):
                    delta = self.data[column_name[_]].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_number).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_number).mean()
                    rs = gain / loss
                    self.data[column_name[_]+'_RSI_'+str(rsi_number)] = 100 - (100 / (1 + rs))
            elif len(column_name) > 1 and (rsi_number > 0):
                for _ in range(len(column_name)):
                    delta = self.data[column_name[_]].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_number).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_number).mean()
                    rs = gain / loss
                    self.data[column_name[_]+'_RSI_'+str(rsi_number)] = 100 - (100 / (1 + rs))
            else:
                print('No option!\nError')
            return self.data
        except:
            print('No option!\nError')
#-------------------------------------------------------------------------------------------------------------------------------


# In[ ]:



