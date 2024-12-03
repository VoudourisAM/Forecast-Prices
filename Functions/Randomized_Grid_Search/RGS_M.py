#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # Randomized Grid SearchCV
# 
# ---

# ---
# ### Import Libraries

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import datetime
import math
import time

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as mpe
get_ipython().run_line_magic('matplotlib', 'inline')
import mplcyberpunk as mplcp

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error  
from sklearn.tree import DecisionTreeRegressor


# ### End Import Libraries
# ---

# ## Regression Models
# ---

# * LinearRegression

# In[ ]:


#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#
def Characteristics_of_LinearRegression():
    '''
    This function visualize a importance model columns and average of this.
    ------------------------------
    Parameter(model): ML - Model
    ------------------------------
    '''
    lr = LinearRegression()
    #--------------------------------------------------#
    poss_coef = lr.coef_[model.coef_ > 0]    
    neg_coef = lr.coef_[model.coef_ < 0]   
    
    poss_avg = np.average(poss_coef)
    neg_avg = np.average(neg_coef)
    #--------------------------------------------------#
    
    plt.style.use("cyberpunk") #Background color
    pal_red = sns.color_palette("flare") #Color
    pal_blue = sns.color_palette("Blues") #Color
    
    fig, ax = plt.subplots(figsize=(18,8), tight_layout=True) #Size of plot dpi=300 for better quality
    
    for _ in range(len(lr.coef_)):
        if (lr.coef_[_] >= poss_avg) or (lr.coef_[_] <= neg_avg):
            if model.lr[_] > 0:
                bar1 = ax.bar(lr.feature_names_in_[_], lr.coef_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_blue[3], color=pal_blue[4])
            elif lr.coef_[_] < 0:
                bar1 = ax.bar(lr.feature_names_in_[_], lr.coef_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_red[2], color=pal_red[3])
            else:
                bar1 = ax.bar(lr.feature_names_in_[_], lr.coef_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, color='White')
        else:
            bar1 = ax.bar(model.feature_names_in_[_], model.coef_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor='Silver', color='Snow')

    ax.hlines(y = poss_avg, xmin = len(lr.coef_) * 0.12, xmax = model.feature_names_in_[-4], linestyles = 'dashed', color = 'White')
    ax.text(lr.feature_names_in_[0], poss_avg, 'Avg Importance Columns', ha ='left', va ='center') 
    ax.text(lr.feature_names_in_[-1], poss_avg, round(poss_avg,2), ha ='right', va ='center') 

    ax.hlines(y = neg_avg, xmin = lr.feature_names_in_[3], xmax = len(model.coef_) * 0.88, linestyles = 'dashed', color = 'White')
    ax.text(lr.feature_names_in_[-1], neg_avg, 'Avg Importance Columns', ha ='right', va ='center') 
    ax.text(lr.feature_names_in_[0], neg_avg, round(neg_avg,2), ha ='left', va ='center') 

    plt.legend([lr], loc="upper right", fontsize=15, labelcolor='Gold')
    plt.xlabel('Columns Name', fontsize=20, color='Gold') #Left title
    plt.ylabel('Importance Model Values', fontsize=20, color='Gold') #Bottom title
    plt.tick_params(axis='y', labelrotation=30, labelsize=12, colors='White')
    plt.tick_params(axis='x', width=3, length=7, labelrotation=30, labelsize=7, bottom=True, direction="in", left=False, colors='White') #White
    
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(0.7)
    ax.spines['top'].set_color('Gold')
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(0.7)
    ax.spines['right'].set_color('Gold')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)     
    plt.show()
#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#


# * DecisionTreeRegressor

# In[1]:


#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#
def DecisionTreeRegressorModel(dataframe):
    '''
    This function print RMSE (Train - Test) of Random Grid Search.
    Visualize Forecast of last month and RMSE of Drid Search
    ------------------------------
    Parameter(dataframe): Dataframe
    ------------------------------
    '''
    #hyperparamet_names = []
    #hyperparamet_values = []

    #for key, value in random_search.best_params_.items():
        #print(f"{key}: {value}")
        #hyperparamet_names.append(key)
        #hyperparamet_values.append(value)
    
    #dataframe_of_model = pd.DataFrame(data=hyperparamet_values, index=hyperparamet_names, columns=['Model_Importance'])
    
    #------------#
    dtr = DecisionTreeRegressor()
    #------------#
    
    def Plot_Of_Last_Months(model):
        #------------ Plot ------------#
        forecast_index = y_test.index.copy()
        forecast_index = pd.to_datetime(forecast_index) #y_test index

        plt.style.use("cyberpunk") #Background color
        pal_red = sns.color_palette("flare") #Color
        pal_blue = sns.color_palette("Blues") #Color
        
        fig, ax1 = plt.subplots(figsize=(15,5), tight_layout=True) #Size of plot dpi=300 for better quality

        ax1.plot(forecast_index, y_test, ls='--', color=pal_red[3])
        ax1.plot(forecast_index, forecast_grid_search, color=pal_blue[4])

        plt.title('Forecast of DecisionTreeRegressor (Grid - Search)', fontsize=20, color='Gold') #Bottom title
        plt.legend(['Actual','Forecast'], loc="upper right", fontsize=15) #Label - Size of plot
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
        #------------ Plot ------------#
    
        #------------ Plot 2,3 ------------#
        Metric_data = pd.DataFrame(data={'Train' : [Train_RMSE],
                                         'Test' : [Test_RMSE]})
        
        fig = gridspec.GridSpec(3, 2)
        pl.figure(figsize=(15, 12), tight_layout=True)

        ax2 = pl.subplot(fig[0, 0])
        plt.ylabel('Root Mean Square Error\nGrid-Search', fontsize=20, color='Gold') #Bottom title
    
        bar1 = ax2.bar('Train', Train_RMSE, width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_blue[3], color=pal_blue[4])
        bar2 = ax2.bar('Test', Test_RMSE, width=0.3, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_red[2], color=pal_red[3])
    
        ax2.tick_params(axis='y', labelsize=0) #White
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
        
        ax3.barh('Train', Train_RMSE+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_blue[4])
        ax3.barh('Test', Test_RMSE+0.1, height=0.4, linewidth=3, alpha=0.3, left=0, color=pal_red[3])
    
        ax3.barh('Train', Train_RMSE, height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_blue[3], color=pal_blue[4])
        ax3.barh('Test', Test_RMSE, height=0.3, linewidth=3, alpha=0.8, left=0, edgecolor=pal_red[2], color=pal_red[3])
    
        ax3.legend(['RMSE - Train','RMSE - Test'], loc="upper right", fontsize=15) #Label - Size of plot
        ax3.tick_params(axis='y', width=7, length=12, labelrotation=30, labelsize=15, left=True, bottom=False, direction="in", colors='White')
        ax3.tick_params(axis='x', labelsize=0) #Rotation label x and y
        ax3.grid(axis='x', zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        ax3.text(x=Metric_data.Train.sum()/2, y=Metric_data.Train.name, s=Metric_data.Train[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels
        ax3.text(x=Metric_data.Test.sum()/2, y=Metric_data.Test.name, s=Metric_data.Test[0], color='White', weight='extra bold', ha='center', fontsize=15) #Text of labels

        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_color('White')
        ax3.spines['left'].set_linewidth(0.3)
        plt.show()
        #------------ Plot 2,3 ------------#

        pl.figure(figsize=(15, 10), tight_layout=True)
        ax4 = pl.subplot(fig[1, 0:])
        
        #Residual for test
        residuals_train = y_train - forecast_train_grid_search
        residuals_index_train = y_train.index.copy()
        residuals_index_train = pd.to_datetime(residuals_index_train)
        
        #Residual for test
        residuals_test = y_test - forecast_grid_search
        residuals_index_test = y_test.index.copy()
        residuals_index_test = pd.to_datetime(residuals_index_test)

        ax4.plot(residuals_index_train, residuals_train, color=pal_blue[4])
        ax4.plot(residuals_index_test, residuals_test, color=pal_red[3])
        
        ax4.legend(['Residuals - Train','Residuals - Test'], loc="upper right", fontsize=15) #Label - Size of plot
        plt.tick_params(axis='x', labelrotation=30, labelsize=15, width=10, length=3,  direction="in", colors='White') #Rotation label x and y
        plt.tick_params(axis='y', labelrotation=30, labelsize=15, colors='White') #Rotation label x and y
        plt.grid(zorder=1, alpha=0.2, linestyle='--', linewidth=0.5, color='darkgrey') #Grid of plot

        mplcp.make_lines_glow()

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color('White')
        ax4.spines['left'].set_linewidth(0.3)
        ax4.spines['bottom'].set_color('White')
        ax4.spines['bottom'].set_linewidth(0.3)
        plt.show()

        
        
    def Train_Test_Split(dataframe):
        X = dataframe.iloc[:, 1:]
        y = dataframe.iloc[:, 0]
        
        X_train = X[0:len(X)-30]
        y_train = y[0:len(y)-30]
        X_test = X[-30:]
        y_test = y[-30:]

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False) #80% X_train - 20% X_test
    
        print('-------------------------')
        print('Full Length is: ', len(X))
        print('Length of Train is: ', len(X_train))
        print('Length of Test is: ', len(X_test))
        print('-------------------------\n\n\n')
    
        return X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = Train_Test_Split(dataframe=dataframe)

    def RMSE(model, Xtrain, Xtest, ytrain, ytest):
        global predict_train, predict_test, Train_RMSE, Test_RMSE
        
        model.fit(Xtrain,ytrain)
        #------------#
        predict_train = model.predict(Xtrain)
        predict_test = model.predict(Xtest)
        #------------#

        #print('The Random Hyparameters Tuning is: ', random_search.best_estimator_)
        print('------------------------------------------------------------------')

        Train_MSE = mean_squared_error(ytrain, predict_train)
        Train_RMSE = math.sqrt(Train_MSE)
        Train_RMSE = round(Train_RMSE,3)
        print('Train-Set Root Mean Square Error: ', Train_RMSE)
        
        Test_MSE = mean_squared_error(ytest, predict_test)
        Test_RMSE = math.sqrt(Test_MSE)
        Test_RMSE = round(Test_RMSE,3)
        print('Test-Set Root Mean Square Error: ', Test_RMSE)
        print('------------------------------------------------------------------')


        
    def Random_Grid_Search():   
        #Define the parameter grid
        param_distributions = {
            "max_depth": [None, 5, 10, 20, 50],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": [None, "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 50],
            "min_impurity_decrease": [0.0, 0.01, 0.1],
            "criterion": ["squared_error", "absolute_error", "friedman_mse"]}

        #Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=dtr,
            param_distributions=param_distributions,
            n_iter=100,  # Number of random samples
            cv=None,        # Number of cross-validation folds
            scoring="neg_mean_squared_error",  # Metric to optimize
            random_state=0,
            n_jobs=-1)    # Use all available cores

        print('!!!------------... Start Randomized Search CV ...------------!!!')
        for _ in range(0,3,1):
            random_search.fit(X_train,y_train)
            print('The Random Hyparameters Tuning is: ', random_search.best_estimator_)
            RMSE(model=random_search, Xtrain=X_train, Xtest=X_test, ytrain=y_train, ytest=y_test)
            if _ < 2:
                print('\n')
        print('!!!------------... End Randomized Search CV ...------------!!!\n\n\n')
    Random_Grid_Search()

    
    
    def Grid_Search():   
        global forecast_train_grid_search, forecast_grid_search
        #Define the parameter grid
        param_distributions = {
            "max_depth": [19,20,21],
            "min_samples_leaf": [1, 2, 3],
            "min_impurity_decrease": [0.01, 0.02, 0.03],
            "max_leaf_nodes" : [49, 50, 51]}

        #Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=dtr,
            param_grid=param_distributions,
            #verbose=10,
            cv=None,
            refit=True,
            n_jobs=-1)    # Use all available cores
    
        grid_search.fit(X_train , y_train)
        forecast_train_grid_search = grid_search.predict(X_train)
        forecast_grid_search = grid_search.predict(X_test)

        print('!!!------------... Start Grid Search CV ...------------!!!')
        print('The Grid Hyparameters Tuning is: ', grid_search.best_estimator_)
        RMSE(model=grid_search, Xtrain=X_train, Xtest=X_test, ytrain=y_train, ytest=y_test)
        print('!!!------------... End Grid Search CV ...------------!!!\n\n\n')
    
        Plot_Of_Last_Months(model=grid_search)
    Grid_Search()
#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#



#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#
def Characteristics_of_DecisionTreeRegressor(model):
    '''
    This function visualize a importance model columns and average of this.
    ------------------------------
    Parameter(model): ML - Model
    ------------------------------
    '''

    #--------------------------------------------------#
    poss_coef = model.feature_importances_[model.feature_importances_ > 0]    
    neg_coef = model.feature_importances_[model.feature_importances_ < 0]   
    
    poss_avg = np.average(poss_coef)
    neg_avg = np.average(neg_coef)
    #--------------------------------------------------#
    
    plt.style.use("cyberpunk") #Background color
    pal_red = sns.color_palette("flare") #Color
    pal_blue = sns.color_palette("Blues") #Color
    
    fig, ax = plt.subplots(figsize=(18,8), tight_layout=True) #Size of plot dpi=300 for better quality
    
    for _ in range(len(model.feature_importances_)):
        if (model.feature_importances_[_] >= poss_avg) or (model.feature_importances_[_] <= neg_avg):
            if model.feature_importances_[_] > 0:
                bar1 = ax.bar(model.feature_names_in_[_], model.feature_importances_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_blue[3], color=pal_blue[4])
            elif model.feature_importances_[_] < 0:
                bar1 = ax.bar(model.feature_names_in_[_], model.feature_importances_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor=pal_red[2], color=pal_red[3])
            else:
                bar1 = ax.bar(model.feature_names_in_[_], model.feature_importances_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, color='White')
        else:
            bar1 = ax.bar(model.feature_names_in_[_], model.feature_importances_[_], width=0.7, linewidth=3, alpha=0.8, bottom=0, edgecolor='Silver', color='Snow')

    ax.hlines(y = poss_avg, xmin = len(model.feature_importances_) * 0.12, xmax = model.feature_names_in_[-4], linestyles = 'dashed', color = 'White')
    ax.text(model.feature_names_in_[0], poss_avg, 'Avg Importance Columns', ha ='left', va ='center') 
    ax.text(model.feature_names_in_[-1], poss_avg, round(poss_avg,2), ha ='right', va ='center') 

    ax.hlines(y = neg_avg, xmin = model.feature_names_in_[3], xmax = len(model.feature_importances_) * 0.88, linestyles = 'dashed', color = 'White')
    ax.text(model.feature_names_in_[-1], neg_avg, 'Avg Importance Columns', ha ='right', va ='center') 
    ax.text(model.feature_names_in_[0], neg_avg, round(neg_avg,2), ha ='left', va ='center') 

    plt.legend([model], loc="upper right", fontsize=15, labelcolor='Gold')
    plt.xlabel('Columns Name', fontsize=20, color='Gold') #Left title
    plt.ylabel('Importance Model Values', fontsize=20, color='Gold') #Bottom title
    plt.tick_params(axis='y', labelrotation=30, labelsize=12, colors='White')
    plt.tick_params(axis='x', width=3, length=7, labelrotation=30, labelsize=7, bottom=True, direction="in", left=False, colors='White') #White
    
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(0.7)
    ax.spines['top'].set_color('Gold')
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(0.7)
    ax.spines['right'].set_color('Gold')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)     
    plt.show()
#------------------------------------------- Randomized Grid SearchCV --------------------------------------------------------#


# ---
# 
# # Randomized Grid SearchCV
# 
# ---

# In[ ]:




