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
from sklearn.metrics import mean_squared_error
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


# In[2]:


#Dataset for Concrete strength
df_concrete = pd.read_excel('Concrete_Data.xls')
df_concrete.head()


# In[3]:


X_train,X_test,y_train,y_test = train_test_split(
    df_concrete[df_concrete.columns[:-1]],
    df_concrete[df_concrete.columns[-1]],
    random_state=0)


# In[4]:


#Preparing the training and testing dataset.
# X_train, X_test, y_train, y_test = train_test_split(df_crime.iloc[:, 0:100].values, df_crime.iloc[:, 100].values, test_size=0.33, random_state=0)
#Standardising the data set
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
#     Kernels = ['linear','poly', 'rbf']
#     Cs = [0.001, 0.01]
#     Gammas = [0.001, 0.1]
#     param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
#     param_grid = {'kernel':Kernels}
#     grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search = SVR()
    grid_search.fit(X, y)
    print('SVR MSE Score for training data: '+str(grid_search.score(X_train, y_train)))
    print('SVR With Parameters: '+str(grid_search))    
    print('SVR coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for SVR on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[6]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    Estimators = np.arange(1,100,25)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    print('RandomForestRegressor MSE Score for training data: '+str(grid_search.best_score_))
    print('RandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    print('Random Forest coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    print('MSE for Random Forest Regressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[7]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    print('DecisionTreeRegressor MSE Score for training data: '+str(grid_search.best_score_))
    print('DecisionTreeRegressor With Parameters: '+str(grid_search.best_params_)) 
    print('Decision Tree coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    print('MSE for Decision Tree Regressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[8]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    Estimators = np.arange(1,100,25)
    Learning_rates = [0.01,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    print('AdaBoostRegressor MSE Score for training data: '+str(grid_search.best_score_))
    print('AdaBoostRegressor With Parameters:'+str(grid_search.best_params_))
    print('AdaBoost Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    print('MSE for AdaBoost Regressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[9]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    print('Skipped due to poor accuracy')
#     grid_search = GaussianProcessRegressor(random_state=0)
#     grid_search.fit(X, y)
#     print('GaussianProcessRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
#     print('GaussianProcessRegressor With Parameters: '+str(grid_search))    
#     print('GaussianProcessRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
#     y_pred = grid_search.predict(X_test)
#     print('MSE for GaussianProcessRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[10]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    print('LinearRegressor MSE Score for training data: '+str(grid_search.best_score_))
    print('LinearRegressor With Parameters:'+str(grid_search.best_params_))  
    print('Linear Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    print('MSE for LinearRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[11]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    Hidden_Layer_Sizes = [5, (10,5)]
    Activations = ['logistic', 'relu']
    param_grid = {'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations}
    grid_search = GridSearchCV(MLPRegressor(max_iter=900,random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    print('NeuralNetworkRegressor MSE Score for training data: '+str(grid_search.best_score_))
    print('NeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))
    print('Neural Network Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    print('MSE for NeuralNetwork Regressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[12]:


#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
print("now ="+str(datetime.now()))
linear_best_param         = linear_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))
random_forest_best_param = random_forest_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))
decision_tree_best_param = decision_tree_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))
ada_boost_best_param     = ada_boost_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))
neural_network_best_param = neural_network_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))
# gaussian_best_param       = gaussian_regressor_param_selection(x_train, y_train, X_test, y_test, 3)
# print("now ="+str(datetime.now()))
svr_best_param           = svr_param_selection(X_train, y_train, X_test, y_test, 3)
print()
print("now ="+str(datetime.now()))


# In[ ]:




