#!/usr/bin/env python
# coding: utf-8

# ---
# # Binance
# ---
# * 1) Extract All Symbols from Binance
# * 2) Extract Symbols of Binance API
# ---

# In[1]:


from binance.client import Client
client = Client()
    
import pandas as pd
import datetime


# In[2]:


class Binance:
    def Get_All_Coins_Info():
        '''
        This function RETURN list with Cryptocurrency Simbols of Binance
        '''
        
        list_all_symbol_of_crypto = []
        
        #try:
        print('--- Start Extract Data\n.\n.\n.')
        info_all_symbol_of_crypto = client.get_all_tickers()
        info_all_symbol_of_crypto = pd.DataFrame(info_all_symbol_of_crypto) #DataFrame of information coins
    
        list_all_symbol_of_crypto = info_all_symbol_of_crypto.symbol.to_list()
        print('--- End Extract Data')    
        return list_all_symbol_of_crypto
        #except:
        #    print('No option!\nError')
    
    def Get_Historical_Data_1Day(list_of_data):
        '''
        This function RETURN dataframe with data 
        From: 1 Jan, 2020
        Until: 31 Dec, 2023
        ------------------------------
        Parameter(list_of_data): List of Coins
        ------------------------------
        '''
        
        dataframe = pd.date_range(start='01/01/2020' , end='31/12/2023', freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
        
        print('--- Start Extract Data\n.\n.\n.')
        
        for i in range(len(list_of_data)):
            try:
                klines = pd.DataFrame(client.get_historical_klines(list_of_data[i], Client.KLINE_INTERVAL_1DAY, start_str="1 Jan, 2020", end_str="31 Dec, 2023"), columns=columns)
                klines['Open time'] = pd.to_datetime(klines['Open time'], unit='ms')
                #klines['Close time'] = pd.to_datetime(temp['Close time'], unit='ms')
                klines.index = pd.to_datetime(klines['Open time'], unit='ms')
                klines.drop(['Open time', 'Close time', 'Volume', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore'], axis=1, inplace=True) #Delete columns

                for j in range(len(klines.columns)): #Rename the columns
                    klines.rename(columns={klines.columns[j] : list_of_data[i]+'_'+klines.columns[j]}, inplace=True) 
                dataframe = dataframe.join(klines)
            except:
                print('No option!\nError')
        
        print('--- End Extract Data')
        return dataframe


# ---
# # End Binance
# ---

# In[ ]:




