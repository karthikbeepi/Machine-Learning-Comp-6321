#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from math import sqrt

f = open("results.txt", "a")
f.write("*******************************************************************************")
f.write("\nResults printed below are for Regression Data set 7 Bike Sharing\n" )
f.write("*******************************************************************************\n")
f.write("\n")


# In[2]:


bikes_hour_df_raws = pd.read_csv('hour.csv')
bikes_hour_df_raws.head()


# In[3]:


# removing useless features  'casual' and 'registered' as they will not help us model demand from a single user behaviour
# bikes_hour_df = bikes_hour_df_raws.drop(['casual' , 'registered'], axis=1)
bikes_hour_df = bikes_hour_df_raws


# In[4]:


plt.scatter(bikes_hour_df['temp'], bikes_hour_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s temp')
plt.xlabel('temp')
plt.ylabel('Number of bikes rented')
f.write("\nAs observed there is a relation between temp and count of bikes")


# In[5]:


plt.scatter(bikes_hour_df['atemp'], bikes_hour_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s temp')
plt.xlabel('temp')
plt.ylabel('Number of bikes rented')
f.write("\nAs observed there is a relation between temp and count of bikes")


# In[6]:


# removing useless feature 'atemp' as both 'temp' and 'atemp' have the same relation with cnt and can may present redundancy 
bikes_hour_df = bikes_hour_df_raws.drop(['atemp'], axis=1)


# In[7]:


# lets copy for editing without effecting original
bikes_df_model_data = bikes_hour_df.copy()

outcome = 'cnt'

#making feature list for each modeling - experiment by adding feature to the exclusion list
feature = [feat for feat in list(bikes_df_model_data) if feat not in [outcome, 'instant', 'dteday']]

#spliting data into train and test portion
X_trian, X_test, y_train, y_test = train_test_split(bikes_df_model_data[feature],
                                                   bikes_df_model_data[outcome],
                                                   test_size=0.3, random_state=0)

from sklearn import linear_model
lr_model = linear_model.LinearRegression()

#training model in training set
lr_model.fit(X_trian, y_train)

# making predection using the test set
y_pred = lr_model.predict(X_test)

#root mean squared error
f.write('\nRMAE: %.2f' % sqrt(mean_absolute_error(y_test, y_pred)))


# In[8]:


# lets copy for editing without effecting original
bikes_df_model_data = bikes_hour_df.copy()

outcome = 'cnt'

#making feature list for each modeling - experiment by adding feature to the exclusion list
feature = [feat for feat in list(bikes_df_model_data) if feat not in [outcome, 'instant', 'dteday']]

X_train, X_test, y_train, y_test = train_test_split(bikes_df_model_data[feature],
                                                   bikes_df_model_data[outcome],
                                                   test_size=0.3, random_state=0)

lr_model = LinearRegression()

#training model in training set
lr_model.fit(X_train, y_train)

# making predection using the test set
y_pred = lr_model.predict(X_test)

#root mean squared error
f.write('\nRMAE: %.2f' % sqrt(mean_absolute_error(y_test, y_pred)))


# In[9]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nLinearRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nLinearRegressor With Parameters:'+str(grid_search.best_params_))  
    f.write('\nLinear Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for LinearRegressor on test set: '+str (mean_absolute_error(y_test, y_pred)))


# In[10]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
#     Kernels = ['linear', 'poly', 'rbf']
    Kernels = ['poly']
#     Cs = [0.001, 0.01, 0.1, 1]
    Cs = [1]
#     Gammas = [0.001, 0.01, 0.1]
    Gammas = [ 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nSVR MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nSVR With Parameters: '+str(grid_search.best_params_))    
    f.write('\nSVR coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for SVR on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[11]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nRandomForestRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nRandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    f.write('\nRandom Forest coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for Random Forest Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[12]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nDecisionTreeRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nDecisionTreeRegressor With Parameters: '+str(grid_search.best_params_)) 
    f.write('\nDecision Tree coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for Decision Tree Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[13]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Learning_rates = [0.01,0.05,0.1,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nAdaBoostRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nAdaBoostRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nAdaBoost Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for AdaBoost Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[14]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq, kernel_expsine]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nGaussianRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nGaussianRegressor With Parameters:'+str(grid_search.best_params_)) 
    f.write('\nGaussian Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for Gaussian Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[15]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Hidden_Layer_Sizes = [1, 5, (5,5), (10,5)]
    Activations = ['logistic', 'relu']
    param_grid = {'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations}
    grid_search = GridSearchCV(MLPRegressor(max_iter=900,random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nNeuralNetworkRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nNeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nNeural Network Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for NeuralNetwork Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[151]:



#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
f.write("\nnow ="+str(datetime.now()))
# svr_best_param           = svr_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
random_forest_best_param = random_forest_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
decision_tree_best_param = decision_tree_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
ada_boost_best_param     = ada_boost_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
linear_best_param         = linear_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
neural_network_best_param = neural_network_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
#gaussian_best_param       = gaussian_regressor_param_selection(x_train_scaled, y_train, 3)
#f.write("\nnow ="+str(datetime.now()))


# In[17]:


f.write("\nnow ="+str(datetime.now()))
svr_best_param           = svr_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))

