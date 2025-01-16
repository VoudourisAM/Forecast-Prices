#!/usr/bin/env python
# coding: utf-8

# ---
# ### Import Libraries

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import math

import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import mplcyberpunk as mplcp
import ipywidgets as widgets

from IPython.display import display,Image, clear_output

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error  


# ### End Import Libraries
# ---

# * Forecast with Linear Regression

# In[1]:


def OHLC_S_F(dataframe, col_open, col_high, col_low, col_close):
    '''
    This Function visualizes a plot of dataframe OPEN - HIGH - LOW - Close values.
    Also if exists Strategies (Moving Average etc...) visualizes a check-box to plot them.
    Visualizes the forecast of model.
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(col_open): Column Name (String)
    Parameter(col_high): Column Name (String)
    Parameter(col_low): Column Name (String)
    Parameter(col_close): Column Name (String)
    ------------------------------
    '''
    
    data_number = 60
    data_pred = 30
    #------------#
    new_dataframe = dataframe.copy()
    new_dataframe.dropna(axis=0, inplace=True)
    list_index = new_dataframe[-data_number:].index
    list_index = pd.to_datetime(list_index)

    list_open = new_dataframe[col_open][-data_number:].values
    list_high = new_dataframe[col_high][-data_number:].values
    list_low = new_dataframe[col_low][-data_number:].values
    list_close = new_dataframe[col_close][-data_number:].values
    #------------#
    
    #------------ Moving Average ------------#
    List_MA = []
    MA = ['_SMA_','_FMA_']
    for __ in range(len(MA)):
        for _ in range(len(new_dataframe.columns)):
            if MA[__] in new_dataframe.columns[_]:
                List_MA.append(new_dataframe.columns[_])

    if len(List_MA) == 2:
        #Create the checkbox widget
        sma_checkbox = widgets.Checkbox(value=False, description='Slow Moving Average', disabled=False, indent=False)
        fma_checkbox = widgets.Checkbox(value=False, description='Fast Moving Average', disabled=False, indent=False)
        mo_av_s = new_dataframe[List_MA[0]][-data_number:].values
        mo_av_f = new_dataframe[List_MA[1]][-data_number:].values
    #------------ Moving Average ------------#    
    
    #------------ Exponential Moving Average ------------#
    List_EMA = []
    EMA = ['_SEMA_','_FEMA_']
    for __ in range(len(EMA)):
        for _ in range(len(new_dataframe.columns)):
            if EMA [__] in new_dataframe.columns[_]:
                List_EMA.append(new_dataframe.columns[_])
    if len(List_EMA) == 2:
        #Create the checkbox widget
        sema_checkbox = widgets.Checkbox(value=False, description='Slow Exponential Moving Average', disabled=False, indent=False)
        fema_checkbox = widgets.Checkbox(value=False, description='Fast Exponential Moving Average', disabled=False, indent=False)
        sema_ = new_dataframe[List_EMA[0]][-data_number:].values
        fema_ = new_dataframe[List_EMA[1]][-data_number:].values
    #------------ Exponential Moving Average ------------#
    
    #------------ Relative_Strength_Index ------------#
    List_RSI = []
    RSI = ['_RSI_']
    for __ in range(len(RSI)):
        for _ in range(len(new_dataframe.columns)):
            if RSI [__] in new_dataframe.columns[_]:
                List_RSI.append(new_dataframe.columns[_])
    if (len(List_RSI) > 0):
        #Create the checkbox widget
        rsi_checkbox = widgets.Checkbox(value=False, description='Relative Strength Index', disabled=False, indent=False)
        rsi = new_dataframe[List_RSI][-data_number:].values
    #------------ Relative_Strength_Index ------------#
    
    #------------ Forecast ------------#
    forecast_checkbox = widgets.Checkbox(value=False, description='Forecast', disabled=False, indent=False)
    #------------ Forecast ------------#
    
    
    
    def plot_ohlc(change=None):
        #Clear the current output
        clear_output(wait=True)

        #Display the checkbox
        if len(List_MA) == 2:
            ma_box = widgets.HBox([sma_checkbox,fma_checkbox])
            display(ma_box)
        if len(List_EMA) == 2:
            ema_box = widgets.HBox([sema_checkbox,fema_checkbox])
            display(ema_box)
        if len(List_RSI) > 0:
            rsi_box = widgets.VBox([rsi_checkbox])
            display(rsi_box)
        forecast_box = widgets.VBox([forecast_checkbox])
        display(forecast_box) 
        
        plt.style.use("cyberpunk")  # Background color
        pal_black = sns.color_palette("Greys")
        pal_red = sns.color_palette("flare")  # Red color palette
        pal_green = sns.color_palette("light:#5A9")  # Green color palette
        pal_blue = sns.color_palette("ch:start=.2,rot=-.3") # Blue color palette
        pal_pink = sns.cubehelix_palette()
        fig, ax1 = plt.subplots(figsize=(15, 7), tight_layout=True)  # Figure size
        
        plt.legend(['Actual','Forecast'], loc="upper right", fontsize=15) #Label - Size of plot
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        for _ in range(len(list_index)):
            if (list_open[_] > list_close[_]):
                ax1.bar(list_index[_], list_open[_] - list_close[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_close[_], edgecolor=pal_red[2], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_red[2])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_red[2])
            elif (list_open[_] < list_close[_]):
                ax1.bar(list_index[_], list_close[_] - list_open[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_open[_], edgecolor=pal_green[4], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_green[4])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_green[4])
            else:
                ax1.bar(list_index[_], list_close[_] - list_open[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_open[_], edgecolor=pal_black[2], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_black[4])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_black[2])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_black[2])
                
        # Update the last bar's color based on the checkbox state
        if len(List_MA) == 2:
            if sma_checkbox.value:
                plt.plot(list_index, mo_av_s, linewidth=1.7, label='Slow Moving Average', linestyle='--', color=pal_blue[1])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fma_checkbox.value:
                plt.plot(list_index, mo_av_f, linewidth=1.7, label='Fast Moving Average', linestyle='--', color=pal_blue[3])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
        if len(List_EMA) == 2:  
            if sema_checkbox.value:
                plt.plot(list_index, sema_, linewidth=1.7, label='Slow Exponential Moving Average', linestyle='--', color=pal_pink[1])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fema_checkbox.value:
                plt.plot(list_index, fema_, linewidth=1.7, label='Fast Exponential Moving Average', linestyle='--', color=pal_pink[3])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
        if len(List_RSI) > 0:
            if rsi_checkbox.value:
                fig, ax2 = plt.subplots(figsize=(17,4), tight_layout=True) #Size of plot dpi=300 for better quality
                outline=mpe.withStroke(linewidth=1.5, alpha=1, foreground='White')
                ax2.plot(list_index, rsi, linewidth=1.5, alpha=0.7, path_effects=[outline], color=pal_red[0])
                mplcp.add_gradient_fill(alpha_gradientglow=0.3) #Glow - Effect lines
    
                ax2.axhline(30, linestyle='--', linewidth=1.5, color=pal_green[5])
                ax2.axhline(70, linestyle='--', linewidth=1.5, color=pal_red[3])
                ax2.text(list_index[0], 32, 'Buy', fontweight='bold', ha ='right', va ='center', fontsize=15) 
                ax2.text(list_index[0], 68, 'Sell', fontweight='bold', ha ='right', va ='center', fontsize=15) 
        
                ax2.set_title('Relative Strength Index', fontdict={'fontsize':20})
                plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
                plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
                plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
                
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color('White')
                ax2.spines['left'].set_linewidth(0.3)
                ax2.spines['bottom'].set_color('White')
                ax2.spines['bottom'].set_linewidth(0.3)
        if forecast_checkbox.value:
            X = new_dataframe.iloc[:, 1:]
            Y = new_dataframe.iloc[:,0]
        
            X_train = X.iloc[0:-data_pred, :]
            X_test = X.iloc[-data_pred:, :]
        
            Y_train = Y.iloc[0:-data_pred]
            Y_test = Y.iloc[-data_pred:]
            index = pd.to_datetime(Y_test.index)
        
            lr = LinearRegression()
            lr.fit(X_train,Y_train)
            predict_model_train = lr.predict(X_train).tolist()
            predict_model = lr.predict(X_test).tolist()

            Train_MSE = mean_squared_error(Y_train, predict_model_train)
            Train_RMSE = round(math.sqrt(Train_MSE),3)
            Test_MSE = mean_squared_error(Y_test, predict_model)
            Test_RMSE = round(math.sqrt(Test_MSE),3)
            
            Residuals_Train = Y_train - predict_model_train
            Residuals_Test = Y_test - predict_model
            Residuals_data_test = pd.DataFrame(data={'Residuals_Train' : Residuals_Train, 'Residuals_Test' : Residuals_Test})
            residuals_index_test = pd.to_datetime(Residuals_data_test.index) #Residuals intex train

            for _ in range(len(predict_model)):
                if (_ > 0) and (Y_test.iloc[_] > Y_test.iloc[_-1]) and (predict_model[_] >= Y_test.iloc[_]):
                    ax1.scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Positive Forecast', edgecolor=pal_green[3], color=pal_green[1])
                elif (_ > 0) and (Y_test.iloc[_] < Y_test.iloc[_-1]) and (predict_model[_] <= Y_test.iloc[_]):
                    ax1.scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Negative Forecast', edgecolor=pal_red[3], color=pal_red[1])
                else:
                    ax1.scatter(index[_], predict_model[_], linewidth=0.3, label='False Forecast', edgecolor=pal_black[5], color=pal_black[1])
            
            fig = gridspec.GridSpec(1, 2)
            pl.figure(figsize=(17, 4), tight_layout=True)
            
            ax3 = pl.subplot(fig[0, 0])
            bar1 = ax3.bar('Train RMSE', Train_RMSE, width=1.3, linewidth=3.5, alpha=0.8, edgecolor=pal_green[4], color='None')
            bar2 = ax3.bar('Test RMSE', Test_RMSE, width=1.3, linewidth=3.5, alpha=0.8, edgecolor=pal_red[2], color='None')
            ax3.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", colors='White') #White
            ax3.tick_params(axis='y', labelsize=0) #White
            ax3.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", left=False, colors='White') #White
            ax3.grid(axis='y', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
            ax3.text(x='Train RMSE', y=Train_RMSE/2, s=Train_RMSE, color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
            ax3.text(x='Test RMSE', y=Test_RMSE/2, s=Test_RMSE, color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_color('White')
            ax3.spines['bottom'].set_linewidth(0.3)
        
            ax4 = pl.subplot(fig[0, 1])
            #ax4.plot(residuals_index_test, Residuals_data_test.Residuals_Train.values, 'o', color=pal_blue[4])
            #ax4.plot(residuals_index_test, Residuals_data_test.Residuals_Test.values, 'o', color=pal_red[3])
            ax4.plot(residuals_index_test, Residuals_data_test.Residuals_Train.values, alpha=0.8, ls='--', label='Residuals Train', color=pal_green[4])
            ax4.plot(residuals_index_test, Residuals_data_test.Residuals_Test.values, alpha=0.8, ls='--', label='Residuals Test', color=pal_red[2])
            ax4.legend()
            #ax4.legend(['Residuals Train - Test'], loc="upper right", fontsize=15) #Label - Size of plot
            ax4.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, direction="in", colors='White') #Rotation label x and y
            ax4.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
            ax4.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
            mplcp.make_lines_glow()
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            ax4.spines['left'].set_color('White')
            ax4.spines['left'].set_linewidth(0.3)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(0.3)
        
        plt.show()
        
        #Attach the observer to the checkbox
        if len(List_MA) == 2:
            sma_checkbox.observe(plot_ohlc, names='value')
            fma_checkbox.observe(plot_ohlc, names='value')
        if len(List_EMA) == 2:
            sema_checkbox.observe(plot_ohlc, names='value')
            fema_checkbox.observe(plot_ohlc, names='value')
        if len(List_RSI) > 0:
            rsi_checkbox.observe(plot_ohlc, names='value')
        forecast_checkbox.observe(plot_ohlc, names='value')
    # Initial plot
    plot_ohlc()


# In[3]:


# Define the OHLC function
def OHLC_S_F_2(dataframe, col_open, col_high, col_low, col_close):
    '''
    This Function visualizes 2 plots of dataframe OPEN - HIGH - LOW - Close values.
    Also if exists Strategies (Moving Average etc...) visualizes a check-box to plot them.
    Visualizes the forecast of model (Show the best price buy and sell with gold).
    First plot visualizes the strategies and forecast (If you want uncomment hlines and see the 
    average of forecast or real Y_test data).
    Second plot visualizes the forecast and the bars start from average of data.
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(col_open): Column Name (String)
    Parameter(col_high): Column Name (String)
    Parameter(col_low): Column Name (String)
    Parameter(col_close): Column Name (String)
    ------------------------------
    '''
    
    data_number = 90
    data_pred = 30
    #------------#
    new_dataframe = dataframe.copy()
    new_dataframe.dropna(axis=0, inplace=True)
    list_index = new_dataframe[-data_number:].index
    list_index = pd.to_datetime(list_index)

    list_open = new_dataframe[col_open][-data_number:].values
    list_high = new_dataframe[col_high][-data_number:].values
    list_low = new_dataframe[col_low][-data_number:].values
    list_close = new_dataframe[col_close][-data_number:].values
    #------------#
    
    #------------ Moving Average ------------#
    List_MA = []
    MA = ['_SMA_','_FMA_']
    for __ in range(len(MA)):
        for _ in range(len(new_dataframe.columns)):
            if MA[__] in new_dataframe.columns[_]:
                List_MA.append(new_dataframe.columns[_])

    if len(List_MA) == 2:
        #Create the checkbox widget
        sma_checkbox = widgets.Checkbox(value=False, description='Slow Moving Average', disabled=False, indent=False)
        fma_checkbox = widgets.Checkbox(value=False, description='Fast Moving Average', disabled=False, indent=False)
        mo_av_s = new_dataframe[List_MA[0]][-data_number:].values
        mo_av_f = new_dataframe[List_MA[1]][-data_number:].values
    #------------ Moving Average ------------#    
    
    #------------ Exponential Moving Average ------------#
    List_EMA = []
    EMA = ['_SEMA_','_FEMA_']
    for __ in range(len(EMA)):
        for _ in range(len(new_dataframe.columns)):
            if EMA [__] in new_dataframe.columns[_]:
                List_EMA.append(new_dataframe.columns[_])
    if len(List_EMA) == 2:
        #Create the checkbox widget
        sema_checkbox = widgets.Checkbox(value=False, description='Slow Exponential Moving Average', disabled=False, indent=False)
        fema_checkbox = widgets.Checkbox(value=False, description='Fast Exponential Moving Average', disabled=False, indent=False)
        sema_ = new_dataframe[List_EMA[0]][-data_number:].values
        fema_ = new_dataframe[List_EMA[1]][-data_number:].values
    #------------ Exponential Moving Average ------------#
    
    #------------ Relative_Strength_Index ------------#
    List_RSI = []
    RSI = ['_RSI_']
    for __ in range(len(RSI)):
        for _ in range(len(new_dataframe.columns)):
            if RSI [__] in new_dataframe.columns[_]:
                List_RSI.append(new_dataframe.columns[_])
    if (len(List_RSI) > 0):
        #Create the checkbox widget
        rsi_checkbox = widgets.Checkbox(value=False, description='Relative Strength Index', disabled=False, indent=False)
        rsi = new_dataframe[List_RSI][-data_number:].values
    #------------ Relative_Strength_Index ------------#
    
    #------------ Forecast ------------#
    forecast_checkbox = widgets.Checkbox(value=False, description='Forecast', disabled=False, indent=False)
    #------------ Forecast ------------#
    
    
    
    def plot_ohlc(change=None):
        #Clear the current output
        clear_output(wait=True)

        #Display the checkbox
        if len(List_MA) == 2:
            ma_box = widgets.HBox([sma_checkbox,fma_checkbox])
            display(ma_box)
        if len(List_EMA) == 2:
            ema_box = widgets.HBox([sema_checkbox,fema_checkbox])
            display(ema_box)
        if len(List_RSI) > 0:
            rsi_box = widgets.VBox([rsi_checkbox])
            display(rsi_box)
        forecast_box = widgets.VBox([forecast_checkbox])
        display(forecast_box) 
        
        plt.style.use("cyberpunk")  # Background color
        pal_black = sns.color_palette("Greys")
        pal_red = sns.color_palette("flare")  # Red color palette
        pal_green = sns.color_palette("light:#5A9")  # Green color palette
        pal_blue = sns.color_palette("ch:start=.2,rot=-.3") # Blue color palette
        pal_pink = sns.cubehelix_palette()
        
        #------- PLOT 1 -------#
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
        
        ax[0].tick_params(axis='x', labelsize=0)
        ax[0].tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax[0].grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        for _ in range(len(list_index)):
            if (list_open[_] - list_close[_]) > 0:
                ax[0].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=0.7, alpha=0.8, bottom=list_close[_], edgecolor=pal_black[0], color=pal_green[4])
                ax[0].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
                ax[0].vlines(x=list_index[_], ymin=list_open[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors="Snow")
                ax[0].vlines(x=list_index[_], ymin=list_close[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors="Snow")
            else:
                ax[0].bar(list_index[_], list_close[_] - list_open[_], width=1, linewidth=0.7, alpha=0.8, bottom=list_open[_], edgecolor=pal_black[0], color=pal_red[2])
                ax[0].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])
                ax[0].vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors="Snow")
                ax[0].vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors="Snow")
                    
        # Update the last bar's color based on the checkbox state
        if len(List_MA) == 2:
            if sma_checkbox.value:
                ax[0].plot(list_index, mo_av_s, linewidth=1.7, label='Slow Moving Average', linestyle='--', color=pal_blue[1])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
                ax[1].plot(list_index, mo_av_s, linewidth=1.7, label='Slow Moving Average', linestyle='--', color=pal_blue[1])
            if fma_checkbox.value:
                ax[0].plot(list_index, mo_av_f, linewidth=1.7, label='Fast Moving Average', linestyle='--', color=pal_blue[3])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
                ax[1].plot(list_index, mo_av_f, linewidth=1.7, label='Fast Moving Average', linestyle='--', color=pal_blue[3])
        if len(List_EMA) == 2:  
            if sema_checkbox.value:
                ax[0].plot(list_index, sema_, linewidth=1.7, label='Slow Exponential Moving Average', linestyle='--', color=pal_pink[1])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
                ax[1].plot(list_index, sema_, linewidth=1.7, label='Slow Exponential Moving Average', linestyle='--', color=pal_pink[1])
            if fema_checkbox.value:
                ax[0].plot(list_index, fema_, linewidth=1.7, label='Fast Exponential Moving Average', linestyle='--', color=pal_pink[3])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
                ax[1].plot(list_index, fema_, linewidth=1.7, label='Fast Exponential Moving Average', linestyle='--', color=pal_pink[3])
        if len(List_RSI) > 0:
            if rsi_checkbox.value:
                fig, ax2 = plt.subplots(figsize=(17,4), tight_layout=True) #Size of plot dpi=300 for better quality
                outline=mpe.withStroke(linewidth=1.5, alpha=1, foreground='White')
                ax2.plot(list_index, rsi, linewidth=1.5, alpha=0.7, path_effects=[outline], color=pal_red[0])
                mplcp.add_gradient_fill(alpha_gradientglow=0.3) #Glow - Effect lines
    
                ax2.axhline(30, linestyle='--', linewidth=1.5, color=pal_green[5])
                ax2.axhline(70, linestyle='--', linewidth=1.5, color=pal_red[3])
                ax2.text(list_index[0], 32, 'Buy', fontweight='bold', ha ='right', va ='center', fontsize=15) 
                ax2.text(list_index[0], 68, 'Sell', fontweight='bold', ha ='right', va ='center', fontsize=15) 
        
                ax2.set_title('Relative Strength Index', fontdict={'fontsize':20})
                plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
                plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
                plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
                
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color('White')
                ax2.spines['left'].set_linewidth(0.3)
                ax2.spines['bottom'].set_color('White')
                ax2.spines['bottom'].set_linewidth(0.3)
        if forecast_checkbox.value:
            X = new_dataframe.iloc[:, 1:]
            Y = new_dataframe.iloc[:,0]
        
            X_train = X.iloc[0:-data_pred, :]
            X_test = X.iloc[-data_pred:, :]
        
            Y_train = Y.iloc[0:-data_pred]
            Y_test = Y.iloc[-data_pred:]
            index = pd.to_datetime(Y_test.index)
        
            lr = LinearRegression()
            lr.fit(X_train,Y_train)
            predict_model = lr.predict(X_test).tolist()
            #ax[0].hlines(y=sum(Y_test)/len(Y_test), xmin=index[0], xmax=index[-1],lw=0.5, linestyles='dashed', color='White')
            #ax[0].hlines(y=sum(predict_model)/len(predict_model), xmin=index[0], xmax=index[-1],lw=2.5, linestyles='dashed', color='White')

            for _ in range(len(predict_model)):
                if (_ > 0) and (Y_test.iloc[_] > Y_test.iloc[_-1]) and (predict_model[_] >= Y_test.iloc[_]):
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Positive Forecast', edgecolor=pal_green[3], color=pal_green[1])
                elif (_ > 0) and (Y_test.iloc[_] < Y_test.iloc[_-1]) and (predict_model[_] <= Y_test.iloc[_]):
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Negative Forecast', edgecolor=pal_red[3], color=pal_red[1])
                else:
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=0.3, label='False Forecast', edgecolor=pal_black[5], color=pal_black[1])
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['left'].set_color('White')
        ax[0].spines['left'].set_linewidth(0.3)
        ax[0].spines['bottom'].set_color('White')
        ax[0].spines['bottom'].set_linewidth(0.1)
        #------- PLOT 1 -------#
        
        #------- PLOT 2 -------#
        ax[1].tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White')
        ax[1].tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax[1].grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
        
        numbers = new_dataframe.iloc[:,0]
        average = sum(numbers)/len(numbers)
        ax[1].axhline(average, linestyle='--', linewidth=0.3, color=pal_black[0])

        for _ in range(len(list_index)):
            if (list_open[_] - list_close[_]) > 0:
                ax[1].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=3, alpha=0.8, bottom=average + (average * 0.003), edgecolor=pal_green[4], color='None')
                ax[1].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
            else:
                ax[1].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=3, alpha=0.8, bottom=average - (average * 0.003), edgecolor=pal_red[2], color='None')
                ax[1].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_color('White')
        ax[1].spines['left'].set_linewidth(0.3)
        ax[1].spines['bottom'].set_color('White')
        ax[1].spines['bottom'].set_linewidth(0.3)
        
        plt.show()
        #------- PLOT 2 -------#
        
        #Attach the observer to the checkbox
        if len(List_MA) == 2:
            sma_checkbox.observe(plot_ohlc, names='value')
            fma_checkbox.observe(plot_ohlc, names='value')
        if len(List_EMA) == 2:
            sema_checkbox.observe(plot_ohlc, names='value')
            fema_checkbox.observe(plot_ohlc, names='value')
        if len(List_RSI) > 0:
            rsi_checkbox.observe(plot_ohlc, names='value')
        forecast_checkbox.observe(plot_ohlc, names='value')
    # Initial plot
    plot_ohlc()


# In[2]:


# Define the OHLC function
def OHLC_S_F__(dataframe, col_open, col_high, col_low, col_close):
    '''
    This Function visualizes 2 plots of dataframe OPEN - HIGH - LOW - Close values.
    Also if exists Strategies (Moving Average etc...) visualizes a check-box to plot them.
    Visualizes the forecast of model (Show the best price buy and sell with gold).
    First plot visualizes the strategies and forecast (If you want uncomment hlines and see the 
    average of forecast or real Y_test data).
    Second plot visualizes the forecast and the bars start from average of data.
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(col_open): Column Name (String)
    Parameter(col_high): Column Name (String)
    Parameter(col_low): Column Name (String)
    Parameter(col_close): Column Name (String)
    ------------------------------
    '''
    
    data_number = 90
    data_pred = 30
    #------------#
    new_dataframe = dataframe.copy()
    new_dataframe.dropna(axis=0, inplace=True)
    list_index = new_dataframe[-data_number:].index
    list_index = pd.to_datetime(list_index)

    list_open = new_dataframe[col_open][-data_number:].values
    list_high = new_dataframe[col_high][-data_number:].values
    list_low = new_dataframe[col_low][-data_number:].values
    list_close = new_dataframe[col_close][-data_number:].values
    #------------#
    
    #------------ Moving Average ------------#
    List_MA = []
    MA = ['_SMA_','_FMA_']
    for __ in range(len(MA)):
        for _ in range(len(new_dataframe.columns)):
            if MA[__] in new_dataframe.columns[_]:
                List_MA.append(new_dataframe.columns[_])

    if len(List_MA) == 2:
        #Create the checkbox widget
        sma_checkbox = widgets.Checkbox(value=False, description='Slow Moving Average', disabled=False, indent=False)
        fma_checkbox = widgets.Checkbox(value=False, description='Fast Moving Average', disabled=False, indent=False)
        mo_av_s = new_dataframe[List_MA[0]][-data_number:].values
        mo_av_f = new_dataframe[List_MA[1]][-data_number:].values
    #------------ Moving Average ------------#    
    
    #------------ Exponential Moving Average ------------#
    List_EMA = []
    EMA = ['_SEMA_','_FEMA_']
    for __ in range(len(EMA)):
        for _ in range(len(new_dataframe.columns)):
            if EMA [__] in new_dataframe.columns[_]:
                List_EMA.append(new_dataframe.columns[_])
    if len(List_EMA) == 2:
        #Create the checkbox widget
        sema_checkbox = widgets.Checkbox(value=False, description='Slow Exponential Moving Average', disabled=False, indent=False)
        fema_checkbox = widgets.Checkbox(value=False, description='Fast Exponential Moving Average', disabled=False, indent=False)
        sema_ = new_dataframe[List_EMA[0]][-data_number:].values
        fema_ = new_dataframe[List_EMA[1]][-data_number:].values
    #------------ Exponential Moving Average ------------#
    
    #------------ Relative_Strength_Index ------------#
    List_RSI = []
    RSI = ['_RSI_']
    for __ in range(len(RSI)):
        for _ in range(len(new_dataframe.columns)):
            if RSI [__] in new_dataframe.columns[_]:
                List_RSI.append(new_dataframe.columns[_])
    if (len(List_RSI) > 0):
        #Create the checkbox widget
        rsi_checkbox = widgets.Checkbox(value=False, description='Relative Strength Index', disabled=False, indent=False)
        rsi = new_dataframe[List_RSI][-data_number:].values
    #------------ Relative_Strength_Index ------------#
    
    #------------ Forecast ------------#
    forecast_checkbox = widgets.Checkbox(value=False, description='Forecast', disabled=False, indent=False)
    #------------ Forecast ------------#
    
    
    
    def plot_ohlc(change=None):
        #Clear the current output
        clear_output(wait=True)

        #Display the checkbox
        if len(List_MA) == 2:
            ma_box = widgets.HBox([sma_checkbox,fma_checkbox])
            display(ma_box)
        if len(List_EMA) == 2:
            ema_box = widgets.HBox([sema_checkbox,fema_checkbox])
            display(ema_box)
        if len(List_RSI) > 0:
            rsi_box = widgets.VBox([rsi_checkbox])
            display(rsi_box)
        forecast_box = widgets.VBox([forecast_checkbox])
        display(forecast_box) 
        
        plt.style.use("cyberpunk")  # Background color
        pal_black = sns.color_palette("Greys")
        pal_red = sns.color_palette("flare")  # Red color palette
        pal_green = sns.color_palette("light:#5A9")  # Green color palette
        pal_blue = sns.color_palette("ch:start=.2,rot=-.3") # Blue color palette
        pal_pink = sns.cubehelix_palette()
        
        #------- PLOT 1 -------#
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
        
        ax[0].tick_params(axis='x', labelsize=0)
        ax[0].tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax[0].grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        for _ in range(len(list_index)):
            if (list_open[_] - list_close[_]) > 0:
                ax[0].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=0.7, alpha=0.8, bottom=list_close[_], edgecolor=pal_black[0], color=pal_green[4])
                ax[0].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
                ax[0].vlines(x=list_index[_], ymin=list_open[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors="Snow")
                ax[0].vlines(x=list_index[_], ymin=list_close[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors="Snow")
            else:
                ax[0].bar(list_index[_], list_close[_] - list_open[_], width=1, linewidth=0.7, alpha=0.8, bottom=list_open[_], edgecolor=pal_black[0], color=pal_red[2])
                ax[0].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])
                ax[0].vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors="Snow")
                ax[0].vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors="Snow")
                    
        # Update the last bar's color based on the checkbox state
        if len(List_MA) == 2:
            if sma_checkbox.value:
                ax[0].plot(list_index, mo_av_s, linewidth=1.7, label='Slow Moving Average', linestyle='--', color=pal_blue[1])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fma_checkbox.value:
                ax[0].plot(list_index, mo_av_f, linewidth=1.7, label='Fast Moving Average', linestyle='--', color=pal_blue[3])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
        if len(List_EMA) == 2:  
            if sema_checkbox.value:
                ax[0].plot(list_index, sema_, linewidth=1.7, label='Slow Exponential Moving Average', linestyle='--', color=pal_pink[1])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fema_checkbox.value:
                ax[0].plot(list_index, fema_, linewidth=1.7, label='Fast Exponential Moving Average', linestyle='--', color=pal_pink[3])
                ax[0].legend(loc="upper right", fontsize=15) #Label - Size of plot
        if forecast_checkbox.value:
            X = new_dataframe.iloc[:, 1:]
            Y = new_dataframe.iloc[:,0]
        
            X_train = X.iloc[0:-data_pred, :]
            X_test = X.iloc[-data_pred:, :]
        
            Y_train = Y.iloc[0:-data_pred]
            Y_test = Y.iloc[-data_pred:]
            index = pd.to_datetime(Y_test.index)
        
            lr = LinearRegression()
            lr.fit(X_train,Y_train)
            predict_model = lr.predict(X_test).tolist()
            
            Test_MSE = mean_squared_error(Y_test, predict_model)
            Test_RMSE = math.sqrt(Test_MSE)
            Test_RMSE = round(Test_RMSE,3)
            
            Residuals_Test = Y_test - predict_model
            Residuals_data_test = pd.DataFrame(data={'Residuals_Test' : Residuals_Test})
            residuals_index_test = pd.to_datetime(Residuals_data_test.index.copy()) #Residuals intex train

            for _ in range(len(predict_model)):
                if (_ > 0) and (Y_test.iloc[_] > Y_test.iloc[_-1]) and (predict_model[_] >= Y_test.iloc[_]):
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Positive Forecast', edgecolor=pal_green[3], color=pal_green[1])
                elif (_ > 0) and (Y_test.iloc[_] < Y_test.iloc[_-1]) and (predict_model[_] <= Y_test.iloc[_]):
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=1.5, label='Negative Forecast', edgecolor=pal_red[3], color=pal_red[1])
                else:
                    ax[0].scatter(index[_], predict_model[_], s=abs(Y_test.iloc[_] - predict_model[_]), linewidth=0.3, label='False Forecast', edgecolor=pal_black[5], color=pal_black[1])
            
            fig = gridspec.GridSpec(1, 2)
            pl.figure(figsize=(17, 4), tight_layout=True)
            
            ax3 = pl.subplot(fig[0, 0])
            bar = ax3.bar('RMSE', Test_RMSE, width=0.3, alpha=0.8, bottom=0, color=pal_red[3])
            ax3.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", colors='White') #White
            ax3.tick_params(axis='y', labelsize=0) #White
            ax3.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", left=False, colors='White') #White
            ax3.grid(axis='y', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
            ax3.text(x='RMSE', y=Test_RMSE/2, s=Test_RMSE, color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
            mplcp.add_bar_gradient(bars=bar)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_color('White')
            ax3.spines['bottom'].set_linewidth(0.3)
        
            ax4 = pl.subplot(fig[0, 1])
            ax4.plot(residuals_index_test, Residuals_data_test.Residuals_Test.values, 'o', color=pal_red[3])
            ax4.legend(['Residuals - Test'], loc="upper right", fontsize=15) #Label - Size of plot
            ax4.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, direction="in", colors='White') #Rotation label x and y
            ax4.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
            ax4.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
            mplcp.make_lines_glow()
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            ax4.spines['left'].set_color('White')
            ax4.spines['left'].set_linewidth(0.3)
            
        if len(List_RSI) > 0:
            if rsi_checkbox.value:
                fig, ax2 = plt.subplots(figsize=(17,4), tight_layout=True) #Size of plot dpi=300 for better quality
                outline=mpe.withStroke(linewidth=1.5, alpha=1, foreground='White')
                ax2.plot(list_index, rsi, linewidth=1.5, alpha=0.7, path_effects=[outline], color=pal_red[0])
                mplcp.add_gradient_fill(alpha_gradientglow=0.3) #Glow - Effect lines
    
                ax2.axhline(30, linestyle='--', linewidth=1.5, color=pal_green[5])
                ax2.axhline(70, linestyle='--', linewidth=1.5, color=pal_red[3])
                ax2.text(list_index[0], 32, 'Buy', fontweight='bold', ha ='right', va ='center', fontsize=15) 
                ax2.text(list_index[0], 68, 'Sell', fontweight='bold', ha ='right', va ='center', fontsize=15) 
        
                ax2.set_title('Relative Strength Index', fontdict={'fontsize':20})
                plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
                plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
                plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
                
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color('White')
                ax2.spines['left'].set_linewidth(0.3)
                ax2.spines['bottom'].set_color('White')
                ax2.spines['bottom'].set_linewidth(0.3)
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['left'].set_color('White')
        ax[0].spines['left'].set_linewidth(0.3)
        ax[0].spines['bottom'].set_color('White')
        ax[0].spines['bottom'].set_linewidth(0.1)
        #------- PLOT 1 -------#
        
        #------- PLOT 2 -------#
        ax[1].tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White')
        ax[1].tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax[1].grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
        
        numbers = new_dataframe.iloc[:,0]
        average = sum(numbers)/len(numbers)
        ax[1].axhline(average, linestyle='--', linewidth=0.3, color=pal_black[0])

        for _ in range(len(list_index)):
            if (list_open[_] - list_close[_]) > 0:
                ax[1].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=3, alpha=0.8, bottom=average + (average * 0.003), edgecolor=pal_green[4], color='None')
                ax[1].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
            else:
                ax[1].bar(list_index[_], list_open[_] - list_close[_], width=1, linewidth=3, alpha=0.8, bottom=average - (average * 0.003), edgecolor=pal_red[2], color='None')
                ax[1].axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_color('White')
        ax[1].spines['left'].set_linewidth(0.3)
        ax[1].spines['bottom'].set_color('White')
        ax[1].spines['bottom'].set_linewidth(0.3)
        
        plt.show()
        #------- PLOT 2 -------#
        
        #Attach the observer to the checkbox
        if len(List_MA) == 2:
            sma_checkbox.observe(plot_ohlc, names='value')
            fma_checkbox.observe(plot_ohlc, names='value')
        if len(List_EMA) == 2:
            sema_checkbox.observe(plot_ohlc, names='value')
            fema_checkbox.observe(plot_ohlc, names='value')
        if len(List_RSI) > 0:
            rsi_checkbox.observe(plot_ohlc, names='value')
        forecast_checkbox.observe(plot_ohlc, names='value')
    # Initial plot
    plot_ohlc()


# In[2]:


def FOREX_OHLC(dataframe, col_open, col_high, col_low, col_close):
    '''
    This Function visualizes a plot of dataframe OPEN - HIGH - LOW - Close values.
    Also if exists Strategies (Moving Average etc...) visualizes a check-box to plot them.
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(col_open): Column Name (String)
    Parameter(col_high): Column Name (String)
    Parameter(col_low): Column Name (String)
    Parameter(col_close): Column Name (String)
    ------------------------------
    '''
    
    data_number = 60
    data_pred = 30
    #------------#
    new_dataframe = dataframe.copy()
    new_dataframe.dropna(axis=0, inplace=True)
    list_index = new_dataframe[-data_number:].index
    list_index = pd.to_datetime(list_index)

    list_open = new_dataframe[col_open][-data_number:].values
    list_high = new_dataframe[col_high][-data_number:].values
    list_low = new_dataframe[col_low][-data_number:].values
    list_close = new_dataframe[col_close][-data_number:].values
    #------------#
    
    #------------ Moving Average ------------#
    List_MA = []
    MA = ['_SMA_','_FMA_', '_EMA_']
    for __ in range(len(MA)):
        for _ in range(len(new_dataframe.columns)):
            if MA[__] in new_dataframe.columns[_]:
                List_MA.append(new_dataframe.columns[_])

    if len(List_MA) == 3:
        #Create the checkbox widget
        sma_checkbox = widgets.Checkbox(value=False, description='Slow Moving Average', disabled=False, indent=False)
        fma_checkbox = widgets.Checkbox(value=False, description='Fast Moving Average', disabled=False, indent=False)
        ema_checkbox = widgets.Checkbox(value=False, description='Extra-F Moving Average', disabled=False, indent=False)
        mo_av_s = new_dataframe[List_MA[0]][-data_number:].values
        mo_av_f = new_dataframe[List_MA[1]][-data_number:].values
        mo_av_ef = new_dataframe[List_MA[2]][-data_number:].values
    #------------ Moving Average ------------#    
    
    #------------ Exponential Moving Average ------------#
    List_EMA = []
    EMA = ['_SEMA_','_FEMA_', '_EFEMA_']
    for __ in range(len(EMA)):
        for _ in range(len(new_dataframe.columns)):
            if EMA [__] in new_dataframe.columns[_]:
                List_EMA.append(new_dataframe.columns[_])
    if len(List_EMA) == 3:
        #Create the checkbox widget
        sema_checkbox = widgets.Checkbox(value=False, description='Slow Exponential Moving Average', disabled=False, indent=False)
        fema_checkbox = widgets.Checkbox(value=False, description='Fast Exponential Moving Average', disabled=False, indent=False)
        efema_checkbox = widgets.Checkbox(value=False, description='Extra-F Exponential Moving Average', disabled=False, indent=False)
        sema_ = new_dataframe[List_EMA[0]][-data_number:].values
        fema_ = new_dataframe[List_EMA[1]][-data_number:].values
        efema_ = new_dataframe[List_EMA[2]][-data_number:].values
    #------------ Exponential Moving Average ------------#
    
    #------------ Relative_Strength_Index ------------#
    List_RSI = []
    RSI = ['_RSI_']
    for __ in range(len(RSI)):
        for _ in range(len(new_dataframe.columns)):
            if RSI [__] in new_dataframe.columns[_]:
                List_RSI.append(new_dataframe.columns[_])
    if (len(List_RSI) > 0):
        #Create the checkbox widget
        rsi_checkbox = widgets.Checkbox(value=False, description='Relative Strength Index', disabled=False, indent=False)
        rsi = new_dataframe[List_RSI][-data_number:].values
    #------------ Relative_Strength_Index ------------#
    
    
    def plot_ohlc(change=None):
        #Clear the current output
        clear_output(wait=True)

        #Display the checkbox
        if len(List_MA) == 3:
            ma_box = widgets.HBox([sma_checkbox,fma_checkbox, ema_checkbox])
            display(ma_box)
        if len(List_EMA) == 3:
            ema_box = widgets.HBox([sema_checkbox,fema_checkbox, efema_checkbox])
            display(ema_box)
        if len(List_RSI) > 0:
            rsi_box = widgets.VBox([rsi_checkbox])
            display(rsi_box)
        
        plt.style.use("cyberpunk")  # Background color
        pal_black = sns.color_palette("Greys")
        pal_red = sns.color_palette("flare")  # Red color palette
        pal_green = sns.color_palette("light:#5A9")  # Green color palette
        pal_blue = sns.color_palette("ch:start=.2,rot=-.3") # Blue color palette
        pal_pink = sns.cubehelix_palette()
        fig, ax1 = plt.subplots(figsize=(15, 7), tight_layout=True)  # Figure size
        
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        for _ in range(len(list_index)):
            if (list_open[_] > list_close[_]):
                ax1.bar(list_index[_], list_open[_] - list_close[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_close[_], edgecolor=pal_red[2], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_red[3])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_red[2])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_red[2])
            elif (list_open[_] < list_close[_]):
                ax1.bar(list_index[_], list_close[_] - list_open[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_open[_], edgecolor=pal_green[4], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_green[5])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_green[4])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_green[4])
            else:
                ax1.bar(list_index[_], list_close[_] - list_open[_], width=0.7, linewidth=1.5, alpha=0.8, bottom=list_open[_], edgecolor=pal_black[2], color='None')
                ax1.axvline(x=list_index[_], alpha=0.1, linewidth=7, color=pal_black[4])
                plt.vlines(x=list_index[_], ymin=list_close[_], ymax=list_high[_], alpha=0.7, linestyles="dashed", colors=pal_black[2])
                plt.vlines(x=list_index[_], ymin=list_open[_], ymax=list_low[_], alpha=0.7, linestyles="dashed", colors=pal_black[2])
                
        # Update the last bar's color based on the checkbox state
        if len(List_MA) == 3:
            if sma_checkbox.value:
                plt.plot(list_index, mo_av_s, linewidth=1.7, label='Slow Moving Average', linestyle='--', color=pal_blue[1])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fma_checkbox.value:
                plt.plot(list_index, mo_av_f, linewidth=1.7, label='Fast Moving Average', linestyle='--', color=pal_blue[3])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if ema_checkbox.value:
                plt.plot(list_index, mo_av_ef, linewidth=1.7, label='Extra-F Moving Average', linestyle='--', color=pal_blue[4])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
                
        if len(List_EMA) == 3:  
            if sema_checkbox.value:
                plt.plot(list_index, sema_, linewidth=1.7, label='Slow Exponential Moving Average', linestyle='--', color=pal_pink[1])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if fema_checkbox.value:
                plt.plot(list_index, fema_, linewidth=1.7, label='Fast Exponential Moving Average', linestyle='--', color=pal_pink[3])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot
            if efema_checkbox.value:
                plt.plot(list_index, efema_, linewidth=1.7, label='Extra-F Exponential Moving Average', linestyle='--', color=pal_blue[4])
                plt.legend(loc="upper right", fontsize=15) #Label - Size of plot    
            
        if len(List_RSI) > 0:
            if rsi_checkbox.value:
                fig, ax2 = plt.subplots(figsize=(17,4), tight_layout=True) #Size of plot dpi=300 for better quality
                outline=mpe.withStroke(linewidth=1.5, alpha=1, foreground='White')
                ax2.plot(list_index, rsi, linewidth=1.5, alpha=0.7, path_effects=[outline], color=pal_red[0])
                mplcp.add_gradient_fill(alpha_gradientglow=0.3) #Glow - Effect lines
    
                ax2.axhline(30, linestyle='--', linewidth=1.5, color=pal_green[5])
                ax2.axhline(50, linestyle='--', linewidth=1.5, color=pal_red[3])
                ax2.text(list_index[0], 32, 'Buy', fontweight='bold', ha ='right', va ='center', fontsize=15) 
                ax2.text(list_index[0], 48, 'Sell', fontweight='bold', ha ='right', va ='center', fontsize=15) 
        
                ax2.set_title('Relative Strength Index', fontdict={'fontsize':20})
                plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
                plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
                plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
                
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_color('White')
                ax2.spines['left'].set_linewidth(0.3)
                ax2.spines['bottom'].set_color('White')
                ax2.spines['bottom'].set_linewidth(0.3)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(0.3)
        
        plt.show()
        
        #Attach the observer to the checkbox
        if len(List_MA) == 3:
            sma_checkbox.observe(plot_ohlc, names='value')
            fma_checkbox.observe(plot_ohlc, names='value')
            ema_checkbox.observe(plot_ohlc, names='value')
        if len(List_EMA) == 3:
            sema_checkbox.observe(plot_ohlc, names='value')
            fema_checkbox.observe(plot_ohlc, names='value')
            efema_checkbox.observe(plot_ohlc, names='value')
        if len(List_RSI) > 0:
            rsi_checkbox.observe(plot_ohlc, names='value')
    # Initial plot
    plot_ohlc()


# ---

# In[ ]:




