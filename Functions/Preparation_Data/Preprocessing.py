#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # PreProcessing or Preparation Data
# 
# ---

# ---
# ### Import Libraries

# In[2]:


import pandas as pd


# ### End Import Libraries
# ---

# In[3]:


#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Correlation(dataframe, column_target, number):
    '''
    This function RETURN a correlation between TARGET Column
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(column_target): Dataframe with TARGET COLUMN (String type)
    Parameter(number): 0.0 - 1.0 Int number
    ------------------------------
    '''
    
    print('--- Start Correlation()\n.\n.')
    
    new_dataframe = dataframe.copy()
    correlation = []
    
    if number >=0.0 or number <=1.0:
        for _ in range(len(new_dataframe.corr()[column_target])):
            if (new_dataframe.corr()[column_target][new_dataframe.corr()[column_target].index[_]] <= -number) or (new_dataframe.corr()[column_target][new_dataframe.corr()[column_target].index[_]] >= number):
                print(_, ' : ', new_dataframe.corr()[column_target].index[_], ' --- ', new_dataframe.corr()[column_target][new_dataframe.corr()[column_target].index[_]])
                correlation.append([new_dataframe.corr()[column_target].index[_]])
                
    print('Correlation between: ', '(', -number, ' - ', number, ')\nLength of Dataframe is: ', len(correlation))
    print('--- End Correlation()')
    return correlation
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Select_Target(dataframe, column_name):
    '''
    This function RETURN a dataframe with Target
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(column_name): Column Name
    ------------------------------
    '''
    try:
        print('--- Start Select_Target()\n.\nTarget: ',column_name, '\n.')
        
        new_dataframe = dataframe.copy()
        new_dataframe['Date'] = new_dataframe.index
        new_dataframe['Date'] = new_dataframe['Date'].shift(-1)
        new_dataframe.drop(['Date'], axis=1, inplace=True)
        new_dataframe.insert(0, 'Target_'+column_name, new_dataframe[column_name].shift(-1))
        new_dataframe.dropna(axis=0, inplace=True)
    
        print('--- End Select_Target()\n')
        return new_dataframe
    except:
        print('No option!\nError')
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
        
        
                
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Drop_Big_NullSum_Columns(dataframe):
    '''
    This Function Return a dataframe, Drop columns with Big-Null Values
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
        
    new_dataframe = dataframe.copy()
    
    null_value = new_dataframe.isnull().sum()
    percent = int((40/100) * len(new_dataframe))

    print('--- Start Drop_Big_NullSum_Columns()\n.\n Drop 40% null columns \n.')
    
    for _ in new_dataframe.columns:
        try:
            #print(_, ' : ', dataframe[_].isnull().sum(), ' --- ', percent, '%')
            
            if null_value[_] > percent:
                new_dataframe.drop(columns=_, axis=1, inplace=True)
        except:
            print('No option!\nError')

    print('--- End  Drop_Big_NullSum_Columns()\n')

    return new_dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Drop_Holidays_Values(dataframe):
    '''
    This function RETURN a dataframe with drop holidays values
    Index of dataframe are time-series (dates)
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    print('--- Start Drop_Holidays_Values()\n.\n.')

    new_dataframe = dataframe.copy()

    try:
        new_dataframe['Days_Of_Weeks'] = new_dataframe.index
        sr = new_dataframe['Days_Of_Weeks']
        sr = pd.to_datetime(sr) 
        result = sr.dt.day_name(locale = 'English') 
        new_dataframe['Days_Of_Weeks'] = result

        drop_holidays_index = []

        for _ in range(len(new_dataframe)):
            #print(_, ' --- ', dataframe.index[_], ' : ', dataframe.iloc[_,-1])
            if new_dataframe.iloc[_,-1] == 'Saturday' or new_dataframe.iloc[_,-1] == 'Sunday':
                #print(_, ' --- ', dataframe.index[_], ' : ', dataframe.iloc[_,-1])
                drop_holidays_index += [new_dataframe.index[_]]

        new_dataframe.drop(index=drop_holidays_index, axis=0, inplace=True)
        new_dataframe.drop(columns=['Days_Of_Weeks'], axis=1, inplace=True)
        print('--- End Drop_Holidays_Values()')
        return new_dataframe

    except:
        print('No option!\nError')
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Drop_Missing_Data(dataframe):
    '''
    This Function Return a dataframe with drop null columns
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    print('--- Start Drop_Missing_Data()\n.\n.')
    new_dataframe = dataframe.copy()

    for _ in new_dataframe.columns:
        if new_dataframe.isnull().sum()[_] > 7:
            #print(_, ': ', dataframe[dataframe.index < '2021-01-01'].isnull().sum()[_])
            new_dataframe.drop(columns=_, axis=1, inplace=True)
    
    print('--- End  Drop_Missing_Data()\n')
    return new_dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Forward_Fill_Data(dataframe):
    '''
    This function RETURN a dataframe with Forward Fill Values by ROW
    ------------------------------
    Parameter(dataframe): DataFrame
    '''
    
    print('--- Start Forward_Fill_Data()\n.\n.')
    new_dataframe = dataframe.copy()
    
    for _ in new_dataframe.columns:
        if new_dataframe.isnull().sum()[_] > 0:
            new_dataframe.ffill(axis ='rows', inplace=True)
    print('--- End Forward_Fill_Data()\n')

    return new_dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Backrward_Fill_Data(dataframe):
    '''
    This function RETURN a dataframe with Backward Fill Values by ROW
    ------------------------------
    Parameter(dataframe): DataFrame
    '''
    
    print('--- Start Backrward_Fill_Data()\n.\n.')
    new_dataframe = dataframe.copy()
    
    for _ in new_dataframe.columns:
        if new_dataframe.isnull().sum()[_] > 0:
            new_dataframe.bfill(axis ='rows', inplace=True)
    print('--- End Backrward_Fill_Data()\n')

    return new_dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Keep_Specific_Columns(dataframe):
    '''
    This Function Return a dataframe and keep specific columns if exist
    (BTCBUSD - BTCBUSD - XRPBUSD - ETHBUSD - LTCBUSD - LINKBUSD - ADABUSD - SOLBUSD - DOGEBUSD - DOTBUSD - AVAXBUSD)
    (Open - High - Low - Close)
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    print('--- Start Keep_Specific_Columns()\n.\n.')
    new_dataframe = dataframe.copy()
        
    for _ in new_dataframe.columns:
        if (_ == 'BTCBUSD_Open') or (_ == 'BTCBUSD_Close') or (_ == 'BTCBUSD_Low') or (_ == 'BTCBUSD_High') or (_ == 'ETHBUSD_Open') or (_ == 'ETHBUSD_Close') or (_ == 'ETHBUSD_Low') or (_ == 'ETHBUSD_High') or (_ == 'XRPBUSD_Open') or (_ == 'XRPBUSD_Close') or (_ == 'XRPBUSD_Low') or (_ == 'XRPBUSD_High') or (_ == 'LTCBUSD_Open') or (_ == 'LTCBUSD_Close') or (_ == 'LTCBUSD_Low') or (_ == 'LTCBUSD_High') or (_ == 'ADABUSD_Open') or (_ == 'ADABUSD_Close') or (_ == 'ADABUSD_Low') or (_ == 'ADABUSD_High') or (_ == 'DOGEBUSD_Open') or (_ == 'DOGEBUSD_Close') or (_ == 'DOGEBUSD_Low') or (_ == 'DOGEBUSD_High') or (_ == 'SOLBUSD_Open') or (_ == 'SOLBUSD_Close') or (_ == 'SOLBUSD_Low') or (_ == 'SOLBUSD_High') or (_ == 'DOTBUSD_Open') or (_ == 'DOTBUSD_Close') or (_ == 'DOTBUSD_Low') or (_ == 'DOTBUSD_High') or (_ == 'LINKBUSD_Open') or (_ == 'LINKBUSD_Close') or (_ == 'LINKBUSD_Low') or (_ == 'LINKBUSD_High') or (_ == 'BNBBUSD_Open') or (_ == 'BNBBUSD_Close') or (_ == 'BNBBUSD_Low') or (_ == 'BNBBUSD_High') or (_ == 'AVAXBUSD_Open') or (_ == 'AVAXBUSD_Close') or (_ == 'AVAXBUSD_Low') or (_ == 'AVAXBUSD_High') or (_ == 'BNBBUSD_Open') or (_ == 'BNBBUSD_Close') or (_ == 'BNBBUSD_Low') or (_ == 'BNBBUSD_High'):
            print(_)
        elif new_dataframe[_].isnull().sum() > 7:
            del new_dataframe[_]
        else:
            continue;
    print('--- End Keep_Specific_Columns()')
    return new_dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#


# ---
# 
# # End PreProcessing or Preparation Data
# 
# ---

# In[ ]:




