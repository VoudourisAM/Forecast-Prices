#!/usr/bin/env python
# coding: utf-8

# ---
# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# ---
# ### Feature Engineering
# * Log Returns
# * Moving Average
# * Rolling Standar Diviation
# * rolling_standar_diviation
# * Spread

# In[ ]:


class Feature_Engineering:
    
#------------------------------------------------------------------------------------------------------------------------------#        
    def Log_Return(dataframe, name_of_column):
        dataframe[name_of_column+'_LG'] = np.log(dataframe[name_of_column] / dataframe[name_of_column].shift(1))
        
        return dataframe
#------------------------------------------------------------------------------------------------------------------------------#    


# In[ ]:





# In[ ]:




