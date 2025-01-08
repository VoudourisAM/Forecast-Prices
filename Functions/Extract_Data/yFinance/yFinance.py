#!/usr/bin/env python
# coding: utf-8

# ---
# # Finance
# ---
# * 1) Extract All Symbols from Finance
# * 2) Extract Data of Finance
# ---

# In[4]:


import yfinance as yf
import pandas as pd
import datetime


# In[5]:


class Finance:
    '''
    The name class is Finance
    Return Forex Data with name of Symbols
    '''
    
    def Get_All_Symbols(self):
        '''
        This Function Get_All_Symbols() Return Symbols of Forex.
        '''
        
        print('--- Start Get All FOREX Symbols Finance\n.\n.\n.')

        # Define forex symbols
        forex_symbols = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X', 'USDCAD=X', 'NZDUSD=X',
                         'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'EURAUD=X', 'CHFJPY=X',
                         'USDSEK=X', 'USDNOK=X', 'USDTRY=X', 'EURTRY=X', 'USDZAR=X']
        print('--- End Get All FOREX Symbols Finance\n')    

        return forex_symbols

    def Get_Historical_Data_1Day_Forex(self, forex_symbols):
        '''
        This function Get_Historical_Data_1Day_Forex() have list of symbols forex.
        This Function Get_Historical_Data_1Day_Forex() return a dataframe with forex data (Open High , Low, Close).
        ------------------------------
        Parameter(forex_symbols): List of Forex Symbols
        ------------------------------
        '''
        
        print('--- Start Extract Data From Finance\n.\n.\n.')

        dataframe = pd.date_range(start='2022-01-01', end='2023-12-31', freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)
        

        # Loop through the first 3 forex symbols (adjust as needed)
        for i in range(len(forex_symbols)):
            # Download the data for the current forex symbol
            klines = pd.DataFrame(yf.download(forex_symbols[i], start='2020-01-01', end='2023-12-31', interval="1d"))
        
            # Flatten MultiIndex columns and remove the name
            klines.columns = klines.columns.get_level_values(0)
            klines.columns.name = None
        
            # Add 'Open time' as the index
            klines['Open time'] = klines.index
            klines.set_index("Open time", inplace=True)
        
            # Convert the index to string format (YYYY-MM-DD)
            klines.index = klines.index.strftime('%Y-%m-%d')
        
            # Select relevant columns
            klines = klines[["Open", "High", "Low", "Close"]]
        
            # Rename the columns to include the forex symbol
            for j in range(len(klines.columns)):
                klines.rename(columns={klines.columns[j]: forex_symbols[i][:-2] + '_' + klines.columns[j]}, inplace=True)

            # Merge klines with the main DataFrame using a left join
            if dataframe.empty:
                dataframe = klines  # Initialize with the first klines DataFrame
            else:
                dataframe = dataframe.join(klines, how="left")
        print('--- End Extract Data From Finance\n')

        return dataframe

    def re_Get_Historical_Data_1Day_Forex(self, old_dataframe, forex_symbols):
        '''
        This Function RETURN re-extract Forex data and merge old dataframe with new.
        ------------------------------
        Parameter(old_dataframe): DataFrame
        Parameter(forex_symbols): List of Forex Symbols
        ------------------------------
        '''
        
        print('--- Start re-Extract Data From Finance\n.\n.\n.')

        old_dataframe_index = str(old_dataframe.index[-1]) #.strftime('%Y-%m-%d')
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")

        dataframe = pd.date_range(start=old_dataframe_index, end=today_date, freq="1D").to_frame()
        dataframe.index.name = 'Open time'
        dataframe.drop([0], axis=1, inplace=True)

        # Loop through the first 3 forex symbols (adjust as needed)
        for i in range(len(forex_symbols)):
            # Download the data for the current forex symbol
            klines = pd.DataFrame(yf.download(forex_symbols[i], start=old_dataframe_index, end=today_date, interval="1d"))
        
            # Flatten MultiIndex columns and remove the name
            klines.columns = klines.columns.get_level_values(0)
            klines.columns.name = None
        
            # Add 'Open time' as the index
            klines['Open time'] = klines.index
            klines.set_index("Open time", inplace=True)
        
            # Convert the index to string format (YYYY-MM-DD)
            klines.index = klines.index.strftime('%Y-%m-%d')
        
            # Select relevant columns
            klines = klines[["Open", "High", "Low", "Close"]]
        
            # Rename the columns to include the forex symbol
            for j in range(len(klines.columns)):
                klines.rename(columns={klines.columns[j]: forex_symbols[i][:-2] + '_' + klines.columns[j]}, inplace=True)

            # Merge klines with the main DataFrame using a left join
            if dataframe.empty:
                dataframe = klines  # Initialize with the first klines DataFrame
            else:
                dataframe = dataframe.join(klines, how="left")
        dataframe.drop(index=[dataframe.index[0]], inplace=True)
        dataframe = pd.concat([old_dataframe,dataframe], axis=0)
        print('--- End re-Extract Data From Finance\n')

        return dataframe
    
    def Select_Target_Finance(self, dataframe, column_name):
        '''
        This function RETURN a dataframe with Target
        ------------------------------
        Parameter(dataframe): DataFrame
        Parameter(column_name): Column Name
        ------------------------------
        '''
        try:
            print('--- Start Select_Target (Finance)\n.\nTarget: ',column_name, '\n.')
        
            new_dataframe = dataframe.copy()
            new_dataframe.insert(0, 'Target_'+column_name, new_dataframe[column_name].shift(-1))
    
            print('--- End Select_Target (Finance)\n')
            return new_dataframe
        except:
            print('No option!\nError')


# ---
# # End Finance
# ---

# In[ ]:




