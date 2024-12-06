#!/usr/bin/env python
# coding: utf-8

# ---
# # Import Libraries

# In[5]:


import pandas as pd

import matplotlib.pyplot as plt
import mplcyberpunk as mplcp
import seaborn as sns


# ---

# In[4]:


def Plot_Relative_Strength_Index(dataframe, rsi_column_name):
    '''
    This function visualize a plot Relative Strength Index
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(rsi_column_name): Column Name
    ------------------------------
    '''
    

    try:
        new_dataframe = dataframe.copy()
        list_index = new_dataframe.index
        list_index = pd.to_datetime(list_index)

        plt.style.use("cyberpunk")  # Background color
        pal_red = sns.color_palette("flare")  # Red color palette
        pal_green = sns.color_palette("light:#5A9")  # Green color palette
        fig, ax1 = plt.subplots(figsize=(14, 7), tight_layout=True)  # Figure size

        ax1.plot(list_index, new_dataframe[rsi_column_name].values, linewidth=1, color=pal_red[0])
        mplcp.add_gradient_fill(alpha_gradientglow=0.7) #Glow - Effect lines
    
        ax1.axhline(30, linestyle='--', linewidth=1.5, color=pal_green[5])
        ax1.axhline(70, linestyle='--', linewidth=1.5, color=pal_red[3])
        ax1.text(list_index[0], 32, 'Buy', ha ='right', va ='center', fontsize=15) 
        ax1.text(list_index[0], 68, 'Sell', ha ='right', va ='center', fontsize=15) 
        
        ax1.set_title('Relative Strength Index', fontdict={'fontsize':20})
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(0.3)
        
        plt.show()
    except:
        print('No option!\nError')


# In[ ]:




