#!/usr/bin/env python
# coding: utf-8

# ---
# # Finance
# ---
# * 1) Extract All Symbols from Finance
# * 2) Extract Data of Finance
# ---

# In[1]:


import yfinance as yf
import pandas as pd


# In[4]:


class Finance:
    '''
    The name class is Finance
    Return Forex Data with name of Symbols
    '''

    def Get_Historical_Data_1Day_Forex(self):
        '''
        This function Get_Historical_Data_1Day_Forex() have list of symbols forex.
        This Function Get_Historical_Data_1Day_Forex() return a dataframe with forex data (Open High , Low, Close).
        '''
        
        # Define forex symbols
        forex_symbols = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X', 'USDCAD=X', 'NZDUSD=X',
                         'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'EURAUD=X', 'CHFJPY=X',
                         'USDSEK=X', 'USDNOK=X', 'USDTRY=X', 'EURTRY=X', 'USDZAR=X']
    
        # Initialize an empty DataFrame for merging the data
        dataframe = pd.DataFrame()

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
    
        return dataframe


# ---
# # End Finance
# ---

# In[ ]:




