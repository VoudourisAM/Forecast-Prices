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
import datetime


# In[5]:


class Binance:
    '''
    The name class is Binance
    Return all symbols (Get_All_Coins_Info)
    Extract Data (Get_Historical_Data_1Day)
    Re Extract Data (re_Get_Historical_Data_1Day)
    '''
    
    def Get_All_Coins_Info(self, countryCode): #Extract - Filter list Fiat of coin
        '''
        This function RETURN list with Cryptocurrency Simbols of Binance
        ------------------------------
        Parameter(countryCode): Fiat name (BUSD,EUR...)
        ------------------------------
        '''

        try:
            print('--- Start Get All Coins From Binance\n.\n.\n.')

            info_coins = []
            exchange_info = client.get_exchange_info()
            info_df = pd.DataFrame(exchange_info['symbols']) #DataFrame of information coins

            for _ in range(len(info_df)):
                #print(_,': ', info_df['quoteAsset'][_])
                if info_df['quoteAsset'][_] == countryCode:
                    #print(_, ': ', info_df['symbol'][_], ': ', info_df['quoteAsset'][_])
                    info_coins.append(info_df['symbol'][_])
            print('--- End Get_All_Coins_Info From Binance')    
            return info_coins
        except:
            print('No option!\nError')


            
    def Get_Historical_Data_1Day(self, list_of_data):
        '''
        This function RETURN dataframe with data 
        From: 1 Jan, 2020
        Until: 31 Dec, 2023
        ------------------------------
        Parameter(list_of_data): List of Coins
        ------------------------------
        '''
        
        dataframe = pd.date_range(start='2020-01-01' , end='2023-12-31', freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
        
        print('--- Start Extract Data From Binance\n.\n.\n.')
        
        for i in range(len(list_of_data)):
            try:
                klines = pd.DataFrame(client.get_historical_klines(list_of_data[i], Client.KLINE_INTERVAL_1DAY, start_str="2020-01-01", end_str="2023-12-31"), columns=columns)
                klines["Open time"] = pd.to_datetime(klines["Open time"], unit="ms")
                klines.set_index("Open time", inplace=True)
                klines = klines[["Open", "High", "Low", "Close"]]
                #klines['Open time'] = pd.to_datetime(klines['Open time'], unit='ms')
                #klines['Close time'] = pd.to_datetime(temp['Close time'], unit='ms')
                #klines.index = pd.to_datetime(klines['Open time'], unit='ms')
                #klines.drop(['Open time', 'Close time', 'Volume', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore'], axis=1, inplace=True) #Delete columns

                for j in range(len(klines.columns)): #Rename the columns
                    klines.rename(columns={klines.columns[j] : list_of_data[i]+'_'+klines.columns[j]}, inplace=True) 
                dataframe = dataframe.join(klines, how="left")
            except:
                print('No option!\nError')
        
        print('--- End Extract Data From Binance')
        return dataframe
    
    
    
    def re_Get_Historical_Data_1Day(self, old_dataframe, list_of_data):
        '''
        This function RETURNS a DataFrame with data 
        From: Last index of old_dataframe
        Until: Date now
        ------------------------------
        Parameter(old_dataframe): DataFrame
        Parameter(list_of_data): List of Coins
        ------------------------------
        '''
    
        old_dataframe_index = str(old_dataframe.index[-1]) #.strftime('%Y-%m-%d')
        today_date = str(datetime.datetime.now().strftime("%Y-%m-%d"))

        dataframe = pd.date_range(start=old_dataframe_index, end=today_date, freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
        
        print('--- Start Re-Extract Data From Binance\n.\n.\n.')
        
        try:
            for i in range(len(list_of_data)):
                klines = pd.DataFrame(client.get_historical_klines(list_of_data[i], Client.KLINE_INTERVAL_1DAY, start_str=old_dataframe_index, end_str=today_date), columns=columns)
                klines["Open time"] = pd.to_datetime(klines["Open time"], unit="ms")
                klines.set_index("Open time", inplace=True)
                klines = klines[["Open", "High", "Low", "Close"]]
            
                for j in range(len(klines.columns)): #Rename the columns
                    klines.rename(columns={klines.columns[j] : list_of_data[i]+'_'+klines.columns[j]}, inplace=True) 
                dataframe = dataframe.join(klines, how="left")
            dataframe.drop(index=dataframe.index[0].strftime('%Y-%m-%d'), axis=0, inplace=True)
            dataframe = pd.concat([old_dataframe,dataframe], axis=0)
            return dataframe
        except:
            print('No option!\nError')
        
        print('--- End Re-Extract Data From Binance')
        
    
    
    def Get_Historical_Data_1Day_Forex_Symbols(self):
        '''
        This function RETURN dataframe with FOREX data 
        From: 1 Jan, 2020
        Until: 31 Dec, 2023
        '''
        
        dataframe = pd.date_range(start='2020-01-01' , end='2023-12-31', freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        
        list_of_data = []
        list_of_data = ['EURUSDT', 'AUDUSDT', 'GBPUSDT', 'CADUSDT', 'JPYUSDT', 'CHFUSDT', 'NZDUSDT', 'SGDUSDT', 
                        'HKDUSDT', 'NOKUSDT', 'SEKUSDT', 'DKKUSDT', 'PLNUSDT', 'ZARUSDT', 'RUBUSDT', 'TRYUSDT',
                        'MXNUSDT', 'CZKUSDT', 'HUFUSDT', 'ILSUSDT']
        
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
        
        print('--- Start Extract Data From Binance\n.\n.\n.')
        
        for i in range(len(list_of_data)):
            try:
                klines = pd.DataFrame(client.get_historical_klines(list_of_data[i], Client.KLINE_INTERVAL_1DAY, start_str="2020-01-01", end_str="2023-12-31"), columns=columns)
                klines["Open time"] = pd.to_datetime(klines["Open time"], unit="ms")
                klines.set_index("Open time", inplace=True)
                klines = klines[["Open", "High", "Low", "Close"]]
               
                for j in range(len(klines.columns)): #Rename the columns
                    klines.rename(columns={klines.columns[j] : list_of_data[i]+'_'+klines.columns[j]}, inplace=True) 
                dataframe = dataframe.join(klines, how="left")
            except:
                print('No option!\nError')
        
        print('--- End Extract Data From Binance')
        return dataframe
        
    def re_Get_Historical_Data_1Day_Forex_Symbols(self, old_dataframe):
        '''
        This function RETURNS a DataFrame with data 
        From: Last index of old_dataframe
        Until: Date now
        ------------------------------
        Parameter(old_dataframe): DataFrame
        ------------------------------
        '''
    
        old_dataframe_index = str(old_dataframe.index[-1]).strftime('%Y-%m-%d')
        today_date = str(datetime.datetime.now().strftime("%Y-%m-%d"))

        dataframe = pd.date_range(start=old_dataframe_index, end=today_date, freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        
        list_of_data = []
        list_of_data = ['EURUSDT', 'AUDUSDT', 'GBPUSDT', 'CADUSDT', 'JPYUSDT', 'CHFUSDT', 'NZDUSDT', 'SGDUSDT', 
                        'HKDUSDT', 'NOKUSDT', 'SEKUSDT', 'DKKUSDT', 'PLNUSDT', 'ZARUSDT', 'RUBUSDT', 'TRYUSDT',
                        'MXNUSDT', 'CZKUSDT', 'HUFUSDT', 'ILSUSDT']
        
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'TB Base Vol', 'TB Quot Vol', 'Ignore']
        
        print('--- Start Re-Extract Data From Binance\n.\n.\n.')
        
        try:
            for i in range(len(list_of_data)):
                klines = pd.DataFrame(client.get_historical_klines(list_of_data[i], Client.KLINE_INTERVAL_1DAY, start_str=old_dataframe_index, end_str=today_date), columns=columns)
                klines["Open time"] = pd.to_datetime(klines["Open time"], unit="ms")
                klines.set_index("Open time", inplace=True)
                klines = klines[["Open", "High", "Low", "Close"]]
            
                for j in range(len(klines.columns)): #Rename the columns
                    klines.rename(columns={klines.columns[j] : list_of_data[i]+'_'+klines.columns[j]}, inplace=True) 
                dataframe = dataframe.join(klines, how="left")
            dataframe.drop(index=dataframe.index[0].strftime('%Y-%m-%d'), axis=0, inplace=True)
            dataframe = pd.concat([old_dataframe,dataframe], axis=0)
            return dataframe
        except:
            print('No option!\nError')
        
        print('--- End Re-Extract Data From Binance')


# ---
# # End Binance
# ---

# In[ ]:




