#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # DIAGRAMS
# 
# ---

# In[4]:


import pandas as pd 
import numpy as np
import math

import mplcyberpunk as mplcp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as mpe
import matplotlib.pylab as pl
import matplotlib as mpl

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  


# In[1]:


#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Of_Matrix_Correlation(dataframe):
    '''
    This function visualizes the plot of Target Correlation (Target is the first column)
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    dataframe = dataframe.dropna()
    
    plt.style.use("cyberpunk") #Background color
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), tight_layout=True)

    # Full correlation heatmap
    matrix = np.tril(dataframe.corr())
    heatmap1 = sns.heatmap(dataframe.corr(),
                           vmin=-1, vmax=1, annot=True, fmt='.1g', square=False, cbar=False,
                           mask=matrix, ax=ax[0], cmap='coolwarm')
    heatmap1.set_title('Correlation Heatmap', fontdict={'fontsize': 15}, color='Gold', pad=12)
    ax[0].tick_params(axis='x', labelsize=0)
    ax[0].tick_params(axis='y', labelrotation=7, labelsize=12, width=3, length=7, direction='in', colors='White')
    ax[0].grid(zorder=3, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey')

    # Correlation with target column heatmap
    target_corr = dataframe.corr()[[dataframe.columns[0]]].sort_values(by=dataframe.columns[0], ascending=False)
    heatmap2 = sns.heatmap(target_corr,
                           vmin=-1, vmax=1, annot=True, fmt='.1g', square=False, cbar=False, ax=ax[1], cmap='coolwarm')
    heatmap2.set_title('Correlation with Target', fontdict={'fontsize': 15}, color='Gold', pad=12)
    ax[1].tick_params(axis='x', labelsize=12, colors='White')
    ax[1].tick_params(axis='y', labelsize=0, colors='White')
    plt.show()
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Of_Line(dataframe, column_name):
    '''
    This function visualize a plot price of features
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(column_name): Column Name
    ------------------------------
    '''
    
    dataframe = dataframe.dropna()

    pal_dark = sns.color_palette("Greys")
    data = dataframe.copy()
    data.index = pd.to_datetime(data.index)
    
    plt.style.use("cyberpunk") #Background color
    
    fig = plt.figure(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality
    
    outline=mpe.withStroke(linewidth=0.3, alpha=0.7, foreground='Gold')
    plt.plot(data.index, data[column_name].values,  path_effects=[outline], alpha=1, color=pal_dark[3]) #Line of plot
    
    plt.legend([column_name], loc="upper right", fontsize=15) #Label - Size of plot
    plt.xlabel('Period', fontsize=20, color='Gold') #Left title
    plt.ylabel('Price ($)', fontsize=20, color='Gold') #Bottom title
    
    plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, bottom=True, direction="in", colors='White') #Rotation label x and y
    plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y

    
    plt.grid(zorder=3, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
    
    plt.axis('on') #Background label
    
    #plt.text(x=data.index[-1], y=data[column_name][-1], s=column_name, fontsize=12) #Possitio of text
    
    mplcp.add_gradient_fill(alpha_gradientglow=0.7) #Glow - Effect lines
    #mplcp.make_lines_glow(alpha_line=0.5, n_glow_lines=10,  diff_linewidth=1.05) #Glow - Effect lines
    
    plt.show() #Figure show   def Plot_Of_Line(dataframe, column_name):
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Train_Test_Split(dataframe):
    '''
    This function visualize 2-graphs Train and Test of Dataframe (animated).
    1) Horizon bar.
    2) The price of Target.
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    def length_table_data(dataframe):
        '''
        ------------------------------
        Parameter(dataframe): DataFrame
        ------------------------------
        '''
        
        dataframe = dataframe.dropna()
        X = dataframe.iloc[:, 1:]
        y = dataframe.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 80% X_train - 20% X_test
    
        list_length_y_train = [len(y_train)]
        list_length_y_test = [len(y_test)]
    
        while len(y_test) > 0:
            if len(y_test) >= 30:
                length_y_train = list_length_y_train[-1] + 30
                list_length_y_train.append(length_y_train)

                y_test = y_test.iloc[30:]
                list_length_y_test.append(len(y_test))
            else:
                break  # Exit loop if less than 30
    
        length_dict = {
            "length_y_train": list_length_y_train,
            "length_y_test": list_length_y_test
        }
        table_data = pd.DataFrame(length_dict)
        return table_data, y

    # Example usage:
    table_data, y = length_table_data(dataframe)
    
    def percent_plot(table):
        '''
        Percent the X-axis
        ------------------------------
        Parameter(table): Data of table
        ------------------------------
        '''
            
        x = []
        y = []
        total_length = table.length_y_train[0] + table.length_y_test[0]
        for i in range(1, 6, 1):
            x += [int(i/5 * total_length)]
            percent = str(int(i/5*100)) + '%'
            y += [percent]
        plt.xticks(x, y, fontsize=15)

    plt.style.use("cyberpunk")
    pal_blue = sns.color_palette("Blues", len(table_data))
    pal_red = sns.color_palette("flare", len(table_data))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)

    def animation_bar_line(i):
        ax1.cla()
        ax2.cla()
    
        for _ in range(i + 1):
            if _ == 0:
                ax1.hlines(y=3, xmin=0, xmax=table_data.length_y_train[_], colors=pal_blue[_], linewidth=25)
                ax1.hlines(y=3, xmin=table_data.length_y_train[_] + 3, xmax=table_data.length_y_train[_] + table_data.length_y_test[_], linewidth=25, colors=pal_red[_])
            else:
                ax1.hlines(y=3, xmin=table_data.length_y_train[_ - 1], xmax=table_data.length_y_train[_], colors=pal_blue[_], linewidth=25)
                ax1.hlines(y=3, xmin=table_data.length_y_train[_], xmax=table_data.length_y_train[_] + table_data.length_y_test[_], colors=pal_red[_], linewidth=25)

            ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 2, y=3, s=table_data.length_y_train[i], color='Black', weight='bold', va='center', fontsize=15)

            if len(str(table_data.length_y_test[i])) == 3:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.06, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 4:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.08, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 5:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.10, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 6:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.12, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) >= 7:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.14, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
            else:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.04, y=3, s=table_data.length_y_test[i], color='Black', weight='bold', va='center', fontsize=15)
    
        percent_plot(table=table_data)
        ax1.tick_params(axis='x', colors='White', direction="in", width=7, length=12, labelrotation=30, labelsize=15)
        ax1.tick_params(axis='y', width=0, length=0, labelsize=0)
        ax1.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey')
        ax1.set_ylabel('Recursive Forecast\nSplit', fontsize=20, color='Gold')
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(1.5) 
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)   
  
        for _ in range(i + 1):
            #ax2.plot(y.index[0:table_data.length_y_train[_]], y.values[0:table_data.length_y_train[_]], linewidth=1.5, color=pal_blue[4])
            ax2.plot(y.index[0:table_data.length_y_train[_]], y.values[0:table_data.length_y_train[_]], linewidth=1.5, color=pal_blue[len(table_data)-2])
            #ax2.plot(y.index[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], y.values[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], linewidth=1.2, color=pal_red[len(table_data)-3])
            #ax2.plot(y.index[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], y.values[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], linewidth=1.2, alpha=0, color=pal_red[len(table_data)-3])

        ax2.tick_params(axis='x', colors='White', direction="in", width=7, length=12, labelrotation=30, labelsize=15)
        ax2.tick_params(axis='y', width=0, length=0, labelsize=0)
        ax2.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey')
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
    
    ani = FuncAnimation(fig=fig, func=animation_bar_line, frames=range(len(table_data)), interval=1000, repeat_delay=1500)
    ani.save(filename="../Forecast_Prices/Animation/plot_bar_line.gif", writer="pillow")
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Train_Test_Splits(dataframe):
    '''
    This function visualize 2-graphs Train and Test of Dataframe (animated).
    1) Horizon bar.
    2) The price of Target.
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''
    
    def length_table_data(dataframe):
        '''
        ------------------------------
        Parameter(dataframe): DataFrame
        ------------------------------
        '''
        
        dataframe = dataframe.dropna()
        X = dataframe.iloc[:, 1:]
        y = dataframe.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 80% X_train - 20% X_test
    
        list_length_y_train = [len(y_train)]
        list_length_y_test = [len(y_test)]
    
        while len(y_test) > 0:
            if len(y_test) >= 30:
                length_y_train = list_length_y_train[-1] + 30
                list_length_y_train.append(length_y_train)

                y_test = y_test.iloc[30:]
                list_length_y_test.append(len(y_test))
            else:
                break  # Exit loop if less than 30
    
        length_dict = {
            "length_y_train": list_length_y_train,
            "length_y_test": list_length_y_test
        }
        table_data = pd.DataFrame(length_dict)
        return table_data, y

    # Example usage:
    table_data, y = length_table_data(dataframe)
    
    def div_numbers_of_data(length_table_data, table_data_column1, table_data_column2, index_of_table):
        '''
        Div the numbers of Dataframe
        ------------------------------
        Parameter(length_table_data): Number of countdown (length of table)
        Parameter(table_data_column1): Column of table
        Parameter(table_data_column2): Column of table
        Parameter(index_of_table): Number of table
        ------------------------------
        '''
        
        div_train = []
        div_test = []
        
        for div in range(length_table_data, 0 , -1):
            div_train.append(int(table_data_column1[index_of_table]/div))
            div_test.append(int(table_data_column2[index_of_table]/div))

        return div_train, div_test
    
    def percent_plot(table):
        '''
        Percent the X-axis
        ------------------------------
        Parameter(table): Data of table
        ------------------------------
        '''
        
        x = []
        y = []
        total_length = table.length_y_train[0] + table.length_y_test[0]
        for i in range(1, 6, 1):
            x += [int(i/5 * total_length)]
            percent = str(int(i/5*100)) + '%'
            y += [percent]
        plt.xticks(x, y, fontsize=15)

    plt.style.use("cyberpunk")
    pal_blue = sns.color_palette("Blues", len(table_data))
    pal_red = sns.color_palette("flare", len(table_data))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)

    def animation_bar_line(i):
        ax1.cla()
        ax2.cla()
    
        for _ in range(i + 1):
            
            for c in range(len(table_data)):
                
                div_train_data, div_test_data = div_numbers_of_data(length_table_data=len(table_data), table_data_column1=table_data.length_y_train, table_data_column2=table_data.length_y_test, index_of_table=_)

                if c == 0:
                    ax1.hlines(y=3, xmin=0, xmax=div_train_data[c+1], colors=pal_blue[c], linewidth=25)
                    ax1.hlines(y=3, xmin=div_train_data[-1], xmax=div_train_data[-1] + div_test_data[c+1], colors=pal_red[c], linewidth=25)          

                elif c < len(table_data)-1:
                    ax1.hlines(y=3, xmin=div_train_data[c], xmax=div_train_data[c+1], colors=pal_blue[c], linewidth=25)
                    ax1.hlines(y=3, xmin=div_train_data[len(table_data)-1] + div_test_data[c], xmax=div_train_data[len(table_data)-1] + div_test_data[c+1], colors=pal_red[c], linewidth=25)          
                
                    
            ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 2, y=3, s=table_data.length_y_train[i], color='White', weight='bold', va='center', fontsize=15)
            
            if len(str(table_data.length_y_test[i])) == 3:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.06, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 4:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.08, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 5:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.10, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) == 6:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.12, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
            elif len(str(table_data.length_y_test[i])) >= 7:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.14, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
            else:
                ax1.text(x=(table_data.length_y_train[_] + table_data.length_y_test[_]) / 1.04, y=3, s=table_data.length_y_test[i], color='White', weight='bold', va='center', fontsize=15)
                
        percent_plot(table=table_data)
        ax1.tick_params(axis='x', colors='White', direction="in", width=7, length=12, labelrotation=30, labelsize=15)
        ax1.tick_params(axis='y', width=0, length=0, labelsize=0)
        ax1.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey')
        ax1.set_ylabel('Recursive Forecast\nSplit', fontsize=20, color='Gold')
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(1.5) 
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)   
  
        for _ in range(i + 1):
            ax2.plot(y.index[0:table_data.length_y_train[_]], y.values[0:table_data.length_y_train[_]], alpha=0, linewidth=1.3, color='White')
            ax2.plot(y.index[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], y.values[table_data.length_y_train[_]:table_data.length_y_train[_] + table_data.length_y_test[_]], alpha=0, linewidth=1.3, color='White')
            
            for c in range(len(table_data)):
                div_train_data, div_test_data = div_numbers_of_data(length_table_data=len(table_data), table_data_column1=table_data.length_y_train, table_data_column2=table_data.length_y_test, index_of_table=_)
            
                if c == 0:
                    ax2.plot(y.index[0:div_train_data[c+1]], y.values[0:div_train_data[c+1]], linewidth=1.3, color=pal_blue[4])
                    ax2.plot(y.index[div_train_data[-1]:div_train_data[-1] + div_test_data[c]], y.values[div_train_data[-1]:div_train_data[-1] + div_test_data[c]], linewidth=0.3, color=pal_red[3])
                    #ax2.plot(y.index[0:div_train_data[c+1]], y.values[0:div_train_data[c+1]], linewidth=1.3, color=pal_blue[c])
                    #ax2.plot(y.index[div_train_data[-1]:div_train_data[-1] + div_test_data[c]], y.values[div_train_data[-1]:div_train_data[-1] + div_test_data[c]], linewidth=1.3, color=pal_red[c])
                elif c < len(table_data)-1:
                    ax2.plot(y.index[div_train_data[c]:div_train_data[c+1]], y.values[div_train_data[c]:div_train_data[c+1]], linewidth=1.3, color=pal_blue[4])
                    ax2.plot(y.index[div_train_data[len(table_data)-1] + div_test_data[c]:div_train_data[len(table_data)-1] + div_test_data[c+1]], y.values[div_train_data[-1] + div_test_data[c]:div_train_data[-1] + div_test_data[c+1]], linewidth=0.3, color=pal_red[3])
                    #ax2.plot(y.index[div_train_data[c]:div_train_data[c+1]], y.values[div_train_data[c]:div_train_data[c+1]], linewidth=1.3, color=pal_blue[c])
                    #ax2.plot(y.index[div_train_data[len(table_data)-1] + div_test_data[c]:div_train_data[len(table_data)-1] + div_test_data[c+1]], y.values[div_train_data[-1] + div_test_data[c]:div_train_data[-1] + div_test_data[c+1]], linewidth=1.3, color=pal_red[c])
            
        ax2.tick_params(axis='x', colors='White', direction="in", width=7, length=12, labelrotation=30, labelsize=15)
        ax2.tick_params(axis='y', width=0, length=0, labelsize=0)
        ax2.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey')
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
    
    ani = FuncAnimation(fig=fig, func=animation_bar_line, frames=range(len(table_data)), interval=1500, repeat_delay=2500)
    ani.save(filename="../Forecast_Prices/Animation/plot_bars_lines.gif", writer="pillow")
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Recursive_Forecast_Plot_Split(dataframe, model):
    '''
    This function visualize 3 plots with Recursive Forecast method
    1) Actual and Forecast Price
    2) The Metric of Forecast (RMSE)
    3) The Residuals of Forecast
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(model): A Model (With parameters or not)
    ------------------------------
    '''
    
    try:
        prediction = [] #Create a list value
    
        dataframe = dataframe.dropna()
        X = dataframe.iloc[:, 1:]
        y = dataframe.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #80% X_train - 20% X_test
    
        Xtrain = X_train.copy()
        Xtest = X_test.copy()
        ytrain = y_train.copy()
        ytest = y_test.copy()
    
        #------------ Plot 1,4 ------------#
        forecast_index = y_test.index.copy()
        forecast_index = pd.to_datetime(forecast_index) #y_test index
        #------------ Plot 1,4 ------------#
    
        #------------ Drop the Xtest rows and Fit again the Xtrain ------------#
        while len(Xtest) > 0:    
            if len(Xtest) >= 30:
            
                model.fit(Xtrain, ytrain) #Fit the model
                forecast = model.predict(Xtest.iloc[0:30]).tolist() #Predict the days
                prediction = prediction + forecast #Insert the forecast values in prediction
        
                Xtrain = pd.concat([Xtrain, Xtest.iloc[0:30]]) #Insert values from second dataframe to first
                ytrain = pd.concat([ytrain, ytest.iloc[0:30]]) #Insert values from second dataframe to first
        
                Xtest.drop(Xtest.index[range(30)], inplace=True) #Drop Rows
                ytest.drop(ytest.index[range(30)], inplace=True) #Drop Rows

            else:
                forecast = model.predict(Xtest).tolist() #Predict the days
                prediction = prediction + forecast #Insert the forecast values in prediction
                break;
        #------------ Drop the Xtest rows and Fit again the Xtrain ------------#
    
        #print('X_train: ', X_train.shape, '\nXtrain: ', Xtrain.shape, '\n\n',
        #      'y_train: ', y_train.shape, '\nytrain: ', ytrain.shape)
        #print()
        #print('X_test: ', X_test.shape, '\nXtest: ', Xtest.shape, '\n\n',
        #      'y_test: ', y_test.shape, '\nytest: ', ytest.shape)
        #print()
        #print('Prediction: ', len(prediction))
    
        plt.style.use("cyberpunk") #Background color
        pal_red = sns.color_palette("flare") #Color
        pal_blue = sns.color_palette("Blues") #Color
    
#------------------------------------------------------------------------------------------------------------------------------#
    
        #------------ Plot 1 ------------#
        fig, ax1 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax1.plot(forecast_index, y_test, ls='--', color=pal_red[3])
        ax1.plot(forecast_index, prediction, color=pal_blue[4])

        plt.ylabel('Forecast', fontsize=20, color='Gold') #Bottom title
        plt.legend(['Actual','Forecast'], loc="upper right", fontsize=15) #Label - Size of plot
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        mplcp.make_lines_glow()
        #mplcp.make_lines_glow(alpha_line=0.5, n_glow_lines=10,  diff_linewidth=1.05) #Glow - Effect lines

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 1 ------------#
    
#------------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------------#
    
        #------------ Plot 2,3 ------------#
        metric_train = model.predict(X=Xtrain).tolist()
    
        Train_MSE = mean_squared_error(ytrain, metric_train)
        Train_RMSE = math.sqrt(Train_MSE)
        Train_RMSE = round(Train_RMSE,3)
        
        Test_MSE = mean_squared_error(y_test, prediction)
        Test_RMSE = math.sqrt(Test_MSE)
        Test_RMSE = round(Test_RMSE,3)

        Metric_data = pd.DataFrame(data={'Train' : [Train_RMSE],
                                         'Test' : [Test_RMSE]})
        #Metric_data.astype(float)
    
        fig = gridspec.GridSpec(3, 2)
        pl.figure(figsize=(15, 15), tight_layout=True)

        ax2 = pl.subplot(fig[0, 0])
        plt.ylabel('Root Mean Square Error', fontsize=20, color='Gold') #Bottom title
    
        bar1 = ax2.bar(Metric_data.columns[0], Metric_data['Train'], width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_blue[3], color=pal_blue[4])
        bar2 = ax2.bar(Metric_data.columns[1], Metric_data['Test'], width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_red[2], color=pal_red[3])
    
        #ax2.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", colors='White') #White
        ax2.tick_params(axis='y', labelsize=0) #White
        #ax2.tick_params(axis='y', labelsize=0) #Rotation label x and y
        #ax2.tick_params(axis='y', labelrotation=30, labelsize=15, left=False, colors='White') #White
        ax2.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", left=False, colors='White') #White
        ax2.grid(axis='y', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        ax2.text(x=Metric_data.Train.name, y=Metric_data.Train.sum()/2, s=Metric_data.Train[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
        ax2.text(x=Metric_data.Test.name, y=Metric_data.Test.sum()/2, s=Metric_data.Test[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels

        mplcp.add_bar_gradient(bars=bar1)
        mplcp.add_bar_gradient(bars=bar2)
    
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(0.3)



        ax3 = pl.subplot(fig[0, 1])
        
        ax3.barh(Metric_data.columns[0], Metric_data['Train']+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_blue[4])
        ax3.barh(Metric_data.columns[1], Metric_data['Test']+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_red[3])
    
        ax3.barh(Metric_data.columns[0], Metric_data['Train'], height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_blue[3], color=pal_blue[4])
        ax3.barh(Metric_data.columns[1], Metric_data['Test'], height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_red[2], color=pal_red[3])
    
        ax3.legend(['RMSE - Train','RMSE - Test'], loc="upper right", fontsize=15) #Label - Size of plot
        ax3.tick_params(axis='y', width=7, length=12, labelrotation=30, labelsize=15, left=True, bottom=False, direction="in", colors='White')
        ax3.tick_params(axis='x', labelsize=0) #Rotation label x and y
        ax3.grid(axis='x', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
        #ax3.tick_params(axis='x', labelrotation=30, labelsize=15, bottom=False, colors='White')

        ax3.text(x=Metric_data.Train.sum()/2, y=Metric_data.Train.name, s=Metric_data.Train[0], color='White', weight='extra bold', va='center', fontsize=15) #Text of labels
        ax3.text(x=Metric_data.Test.sum()/2, y=Metric_data.Test.name, s=Metric_data.Test[0], color='White', weight='extra bold', va='center', fontsize=15) #Text of labels
    
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_color('White')
        ax3.spines['left'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 2,3 ------------#
    
#------------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------------#
    
        length_of_dataframe = int(len(dataframe) * 0.79)

        #------------ Plot 4 ------------#
        Residuals_Train = ytrain - metric_train
        Residuals_Test = y_test - prediction
    
        Residuals_data_train = pd.DataFrame(data={'Residuals_Train' : Residuals_Train})
        Residuals_data_test = pd.DataFrame(data={'Residuals_Test' : Residuals_Test})

        residuals_index_train = Residuals_data_train.index.copy()
        residuals_index_train = pd.to_datetime(residuals_index_train) #Residuals intex train
    
        fig, ax4 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax4.plot(residuals_index_train[0:length_of_dataframe], Residuals_data_train.Residuals_Train[0:length_of_dataframe], color=pal_blue[4])
        ax4.plot(forecast_index, Residuals_data_test.Residuals_Test.values, color=pal_red[3])
        ax4.legend(['Residuals - Train','Residuals - Test'], loc="upper right", fontsize=15) #Label - Size of plot

        ax4.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, direction="in", colors='White') #Rotation label x and y
        #ax4.tick_params(axis='y', labelsize=0) #White
        ax4.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        #ax4.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax4.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
    
        mplcp.make_lines_glow()

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color('White')
        ax4.spines['left'].set_linewidth(0.3)
        ax4.spines['bottom'].set_color('White')
        ax4.spines['bottom'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 4 ------------#
    except:
        print('No option!\nError')
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Percentage(dataframe, column_name_1, column_name_2):
    '''
    This function visualize the trend of 3 plots
    1 and 2 (top-right) visualize the trend of Volume value
    3 visualize the target price of value
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(column_name_1): Column Name (string)
    Parameter(column_name_2): Column Name (string)
    ------------------------------
    '''
    
    try:
        dataframe = dataframe.dropna()

        pal_red = sns.color_palette("flare") #Color
        pal_blue = sns.color_palette("Blues") #Color

        plt.style.use("cyberpunk") #Background color

        fig = plt.figure(figsize=(16, 8), tight_layout=True)

        gs = fig.add_gridspec(6,6)

        ax1 = fig.add_subplot(gs[:2,0:5])
        ax1 = sns.histplot(data=dataframe[column_name_1], shrink=.95, kde=True, kde_kws=dict(cut=0.1), alpha=0.3, linewidth = 0.7, element = "bars", #step,poly
                           line_kws = {'linewidth':'0.7'},
                           ec=pal_blue[3], color=pal_red[3])
        ax1.lines[0].set_color(pal_blue[4])

        plt.tick_params(axis='both', labelsize=0) #Rotation label x and y
        plt.xlabel('0', fontsize=0)
        plt.ylabel('0', fontsize=0)
        plt.grid(False) #Grid of plot

        ax1 = plt.gca()
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['top'].set_color('White')
        ax1.spines['top'].set_linewidth(0.3)
        ax1.invert_yaxis()  # labels read top-to-bottom
    


        ax2 = fig.add_subplot(gs[1:,4:6])
        ax2 = sns.histplot(data=dataframe[column_name_1], y=dataframe[column_name_1], shrink=.95, kde=True, kde_kws=dict(cut=0.1), alpha=0.3, linewidth = 0.7, element = "bars", #step,poly
                           line_kws = {'linewidth':'0.7'},
                           ec=pal_blue[3], color=pal_red[3])
        ax2.lines[0].set_color(pal_blue[3])

        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, bottom=True, direction="in", colors='White')
        plt.tick_params(axis='y', labelsize=0) #Rotation label x and y
        plt.xlabel('0', fontsize=0)
        plt.ylabel('Volume', fontsize=20, color='Gold')
        ax2.yaxis.set_label_position("right")
        plt.grid(False) #Grid of plot
        #plt.xlabel('Volume', fontsize=20, color='Gold') #Left title

        ax2 = plt.gca()
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(0.3)
        ax2.spines['right'].set_color('White')
        ax2.spines['right'].set_linewidth(0.3)
        ax2.invert_xaxis()  # labels read top-to-bottom



        ax3 = fig.add_subplot(gs[2:,:4])
        ax3 = sns.histplot(data=dataframe[column_name_2], x=dataframe[column_name_2], shrink=.95, kde=True, kde_kws=dict(cut=0.1), alpha=0.3, linewidth = 0.7, element = "bars", #step,poly
                           line_kws = {'linewidth':'0.7'}, 
                           ec=pal_red[3], color=pal_blue[3])
        ax3.lines[0].set_color(pal_red[4])

        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, bottom=True, direction="in", colors='White')
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(False) #Grid of plot
        #plt.grid(zorder=3, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        plt.xlabel('Price', fontsize=20, color='Gold') #Left title
        plt.ylabel('Count', fontsize=20, color='Gold', loc='top') #Bottom title


        ax3 = plt.gca()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color('White')
        ax3.spines['left'].set_linewidth(0.3)
        ax3.spines['bottom'].set_color('White')
        ax3.spines['bottom'].set_linewidth(0.3)

        plt.show()
    except:
        print('No option!\nError')
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Plot_Of_Missing_Data(dataframe):
    '''
    This function visualize a plot with missing data
    ------------------------------
    Parameter(dataframe): DataFrame
    ------------------------------
    '''

    plt.style.use("cyberpunk") #Background color
    pal_green = sns.color_palette("Dark2", 7)
    colours = ['#34495E', pal_green[0]] 
    
    fig, (x1) = plt.subplots(figsize=(20,8), tight_layout=True)
    sns.heatmap(dataframe.isnull(), cmap=colours, cbar=False, yticklabels=False, ax=x1)

    plt.tick_params(axis='y', labelrotation=7, labelsize=7, colors='White') #Rotation label x and y
    plt.tick_params(axis='x', labelsize=12, colors='White') #Rotation label x and y
    plt.xlabel('Missing Data of Columns', fontsize=20, color='Gold') #Bottom title
    plt.ylabel('', fontsize=0) #Bottom title
    
    plt.show()
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#
def Recursive_Forecast_Train_Test_Plot_Split(dataframe, model):
    '''
    This function visualize 3 plots with Recursive Forecast method
    1) Actual and Forecast Price
    2) The Metric of Forecast (RMSE)
    3) The Residuals of Forecast
    ------------------------------
    Parameter(dataframe): DataFrame
    Parameter(model): A Model (With parameters or not)
    ------------------------------
    '''
    
    try:
        prediction = [] #Create a list value
        
        dataframe = dataframe.dropna()
        X = dataframe.iloc[:, 1:]
        y = dataframe.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #80% X_train - 20% X_test
    
        Xtrain = X_train.copy()
        Xtest = X_test.copy()
        ytrain = y_train.copy()
        ytest = y_test.copy()

        model.fit(Xtrain, ytrain) #Fit the model
        model_train_predict = model.predict(Xtrain)
        
        #------------ Plot 1,4 ------------#
        index = y_train.index.copy()
        index = pd.to_datetime(index) #y_test index
        
        forecast_index = y_test.index.copy()
        forecast_index = pd.to_datetime(forecast_index) #y_test index
        #------------ Plot 1,4 ------------#

        #------------ Drop the Xtest rows and Fit again the Xtrain ------------#
        while len(Xtest) > 0:    
            if len(Xtest) >= 30:
            
                model.fit(Xtrain, ytrain) #Fit the model
                forecast = model.predict(Xtest.iloc[0:30]).tolist() #Predict the days
                prediction = prediction + forecast #Insert the forecast values in prediction
        
                Xtrain = pd.concat([Xtrain, Xtest.iloc[0:30]]) #Insert values from second dataframe to first
                ytrain = pd.concat([ytrain, ytest.iloc[0:30]]) #Insert values from second dataframe to first
        
                Xtest.drop(Xtest.index[range(30)], inplace=True) #Drop Rows
                ytest.drop(ytest.index[range(30)], inplace=True) #Drop Rows

            else:
                forecast = model.predict(Xtest).tolist() #Predict the days
                prediction = prediction + forecast #Insert the forecast values in prediction
                break;
    #------------ Drop the Xtest rows and Fit again the Xtrain ------------#
    
    #print('X_train: ', X_train.shape, '\nXtrain: ', Xtrain.shape, '\n\n',
    #      'y_train: ', y_train.shape, '\nytrain: ', ytrain.shape)
    #print()
    #print('X_test: ', X_test.shape, '\nXtest: ', Xtest.shape, '\n\n',
    #      'y_test: ', y_test.shape, '\nytest: ', ytest.shape)
    #print()
    #print('Prediction: ', len(prediction))
    
        plt.style.use("cyberpunk") #Background color
        pal_red = sns.color_palette("flare") #Color
        pal_blue = sns.color_palette("Blues") #Color
    
#------------------------------------------------------------------------------------------------------------------------------#
        #------------ Plot 1 ------------#
        fig, ax1 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax1.plot(index, y_train, ls='--', linewidth=1.5, label='Actual Price', color=pal_red[0])
        ax1.plot(index, model_train_predict, linewidth=1, label='Model Train', color=pal_blue[1])

        ax1.plot(forecast_index, y_test, ls='--', color=pal_red[0])
        #ax1.plot(forecast_index, prediction, label='asdasdfdg', color=pal_blue[4])
        
        #plt.legend() #Label - Size of plot
        plt.legend(['Actual Price','Model Train'], loc="upper right", fontsize=15) #Label - Size of plot
        plt.ylabel('Model Train', fontsize=20, color='Gold') #Bottom title
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        mplcp.make_lines_glow()

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('White')
        ax1.spines['left'].set_linewidth(0.3)
        ax1.spines['bottom'].set_color('White')
        ax1.spines['bottom'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 1 ------------#
#------------------------------------------------------------------------------------------------------------------------------#
  
#------------------------------------------------------------------------------------------------------------------------------#
        #------------ Plot 2 ------------#
        fig, ax2 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax2.plot(forecast_index, y_test, ls='--', label='Actual', color=pal_red[3])
        ax2.plot(forecast_index, prediction, label='Forecast', color=pal_blue[4])

        plt.ylabel('Forecast', fontsize=20, color='Gold') #Bottom title
        #plt.legend() #Label - Size of plot
        plt.legend(['Actual','Forecast'], loc="upper right", fontsize=15) #Label - Size of plot
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        mplcp.make_lines_glow()
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('White')
        ax2.spines['left'].set_linewidth(0.3)
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 2 ------------#
#------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------------------#
        #------------ Plot 2,3 ------------#
        metric_train = model.predict(X=Xtrain).tolist()
    
        Train_MSE = mean_squared_error(ytrain, metric_train)
        Train_RMSE = math.sqrt(Train_MSE)
        Train_RMSE = round(Train_RMSE,3)
        
        Test_MSE = mean_squared_error(y_test, prediction)
        Test_RMSE = math.sqrt(Test_MSE)
        Test_RMSE = round(Test_RMSE,3)

        Metric_data = pd.DataFrame(data={'Train' : [Train_RMSE],
                                         'Test' : [Test_RMSE]})
        #Metric_data.astype(float)
    
        fig = gridspec.GridSpec(3, 2)
        pl.figure(figsize=(15, 15), tight_layout=True)

        ax2 = pl.subplot(fig[0, 0])
        plt.ylabel('Root Mean Square Error', fontsize=20, color='Gold') #Bottom title
    
        bar1 = ax2.bar(Metric_data.columns[0], Metric_data['Train'], width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_blue[3], color=pal_blue[4])
        bar2 = ax2.bar(Metric_data.columns[1], Metric_data['Test'], width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_red[2], color=pal_red[3])
    
        ax2.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", colors='White') #White
        ax2.tick_params(axis='y', labelsize=0) #White
        #ax2.tick_params(axis='y', labelsize=0) #Rotation label x and y
        #ax2.tick_params(axis='y', labelrotation=30, labelsize=15, left=False, colors='White') #White
        ax2.tick_params(axis='x', width=7, length=12, labelrotation=30, labelsize=15, bottom=True, direction="in", left=False, colors='White') #White
        ax2.grid(axis='y', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        ax2.text(x=Metric_data.Train.name, y=Metric_data.Train.sum()/2, s=Metric_data.Train[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
        ax2.text(x=Metric_data.Test.name, y=Metric_data.Test.sum()/2, s=Metric_data.Test[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels

        mplcp.add_bar_gradient(bars=bar1)
        mplcp.add_bar_gradient(bars=bar2)
    
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color('White')
        ax2.spines['bottom'].set_linewidth(0.3)



        ax3 = pl.subplot(fig[0, 1])
        
        ax3.barh(Metric_data.columns[0], Metric_data['Train']+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_blue[4])
        ax3.barh(Metric_data.columns[1], Metric_data['Test']+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_red[3])
    
        ax3.barh(Metric_data.columns[0], Metric_data['Train'], height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_blue[3], color=pal_blue[4])
        ax3.barh(Metric_data.columns[1], Metric_data['Test'], height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_red[2], color=pal_red[3])
    
        ax3.legend(['RMSE - Train','RMSE - Test'], loc="upper right", fontsize=15) #Label - Size of plot
        ax3.tick_params(axis='y', width=7, length=12, labelrotation=30, labelsize=15, left=True, bottom=False, direction="in", colors='White')
        ax3.tick_params(axis='x', labelsize=0) #Rotation label x and y
        ax3.grid(axis='x', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
        #ax3.tick_params(axis='x', labelrotation=30, labelsize=15, bottom=False, colors='White')

        ax3.text(x=Metric_data.Train.sum()/2, y=Metric_data.Train.name, s=Metric_data.Train[0], color='White', weight='extra bold', va='center', fontsize=15) #Text of labels
        ax3.text(x=Metric_data.Test.sum()/2, y=Metric_data.Test.name, s=Metric_data.Test[0], color='White', weight='extra bold', va='center', fontsize=15) #Text of labels
    
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_color('White')
        ax3.spines['left'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 2,3 ------------#
#------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------------------#
    
        length_of_dataframe = int(len(dataframe) * 0.79)

        #------------ Plot 4 ------------#
        Residuals_Train = ytrain - metric_train
        Residuals_Test = y_test - prediction
    
        Residuals_data_train = pd.DataFrame(data={'Residuals_Train' : Residuals_Train})
        Residuals_data_test = pd.DataFrame(data={'Residuals_Test' : Residuals_Test})

        residuals_index_train = Residuals_data_train.index.copy()
        residuals_index_train = pd.to_datetime(residuals_index_train) #Residuals intex train
    
        fig, ax4 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax4.plot(residuals_index_train[0:length_of_dataframe], Residuals_data_train.Residuals_Train[0:length_of_dataframe], color=pal_blue[4])
        ax4.plot(forecast_index, Residuals_data_test.Residuals_Test.values, color=pal_red[3])
        
        ax4.legend(['Residuals - Train','Residuals - Test'], loc="upper right", fontsize=15) #Label - Size of plot
        ax4.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3, direction="in", colors='White') #Rotation label x and y
        #ax4.tick_params(axis='y', labelsize=0) #White
        ax4.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        ax4.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot
    
        mplcp.make_lines_glow()

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color('White')
        ax4.spines['left'].set_linewidth(0.3)
        ax4.spines['bottom'].set_color('White')
        ax4.spines['bottom'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 4 ------------#
#------------------------------------------------------------------------------------------------------------------------------#

    except:
        print('No option!\nError')
#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#



#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#

#-------------------------------------------------- PLOTS ---------------------------------------------------------------------#


# ---
# 
# # END DIAGRAMS
# 
# ---

# In[ ]:




