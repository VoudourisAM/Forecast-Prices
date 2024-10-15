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
    
    null_value = dataframe.isnull().sum()
    percent = int((40/100) * len(dataframe))

    print('--- Start Drop_Big_NullSum_Columns()\n.\n Drop 40% null columns \n.')

    for _ in dataframe.columns:
        try:
            #print(_, ' : ', dataframe[_].isnull().sum(), ' --- ', percent, '%')
            
            if null_value[_] > percent:
                dataframe.drop(columns=_, axis=1, inplace=True)
        except:
            print('No option!\nError')

    print('--- End  Drop_Big_NullSum_Columns()\n')

    return dataframe
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
    
    try:
        dataframe['Days_Of_Weeks'] = dataframe.index
        sr = dataframe['Days_Of_Weeks']
        sr = pd.to_datetime(sr) 
        result = sr.dt.day_name(locale = 'English') 
        dataframe['Days_Of_Weeks'] = result

        drop_holidays_index = []

        for _ in range(len(dataframe)):
            #print(_, ' --- ', dataframe.index[_], ' : ', dataframe.iloc[_,-1])
            if dataframe.iloc[_,-1] == 'Saturday' or dataframe.iloc[_,-1] == 'Sunday':
                #print(_, ' --- ', dataframe.index[_], ' : ', dataframe.iloc[_,-1])
                drop_holidays_index += [dataframe.index[_]]

        dataframe.drop(index=drop_holidays_index, axis=0, inplace=True)
        dataframe.drop(columns=['Days_Of_Weeks'], axis=1, inplace=True)
        print('--- End Drop_Holidays_Values()')
        return dataframe

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

    for _ in dataframe.columns:
        if dataframe.isnull().sum()[_] > 7:
            #print(_, ': ', dataframe[dataframe.index < '2021-01-01'].isnull().sum()[_])
            dataframe.drop(columns=_, axis=1, inplace=True)
    
    print('--- End  Drop_Missing_Data()\n')
    return dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Forward_Fill_Data(dataframe):
    '''
    This function RETURN a dataframe with Forward Fill Values by ROW
    ------------------------------
    Parameter(dataframe): DataFrame
    '''
    
    print('--- Start Forward_Fill_Data()\n.\n.')

    for _ in dataframe.columns:
        if dataframe.isnull().sum()[_] > 0:
            dataframe.ffill(axis ='rows', inplace=True)
    print('--- End Forward_Fill_Data()\n')

    return dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#
def Backrward_Fill_Data(dataframe):
    '''
    This function RETURN a dataframe with Backward Fill Values by ROW
    ------------------------------
    Parameter(dataframe): DataFrame
    '''
    
    print('--- Start Backrward_Fill_Data()\n.\n.')

    for _ in dataframe.columns:
        if dataframe.isnull().sum()[_] > 0:
            dataframe.bfill(axis ='rows', inplace=True)
    print('--- End Backrward_Fill_Data()\n')

    return dataframe
#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#



#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#

#-------------------------------------------------- PREPERATION ---------------------------------------------------------------#


# ---
# 
# # End PreProcessing or Preparation Data
# 
# ---

# In[ ]:




