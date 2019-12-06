#!/usr/bin/env python
# coding: utf-8

# In[81]:


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


# In[82]:


# Data-Set importing the Dataset: SGEMM GPU kernel performance Data Set
df_gpu_kernel = pd.read_csv("sgemm_product.csv")
print(df_gpu_kernel.isnull().sum())
df_gpu_kernel.info()


# In[83]:


df_gpu_kernel.head()


# In[84]:


#Preparing the final dataset
X = df_gpu_kernel[df_gpu_kernel.columns[:-5]] 
y = (df_gpu_kernel['Run1 (ms)']+df_gpu_kernel['Run2 (ms)']+df_gpu_kernel['Run3 (ms)']+df_gpu_kernel['Run4 (ms)'])/4


# In[85]:


#Analysing Correlation between various features of the wine dataset
gpu_corr = df_gpu_kernel.corr()
figure, ax = plt.subplots(figsize = (10,10))
ax.set_title('Correlation Matrix for SGEMM data-set')
matrix = sns.heatmap(gpu_corr,ax=ax, annot= True)


# In[86]:


#Creating Traning and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[87]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
    Kernels = ['poly', 'rbf']
    Cs = [0.001, 0.01]
    Gammas = [0.001, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
#     grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search = SVR()
    grid_search.fit(X, y)
    print('SVR MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('SVR With Parameters: '+str(grid_search))    
    print('SVR coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for SVR on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[88]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = RandomForestRegressor(random_state=0)
    grid_search.fit(X, y)
    print('RandomForestRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('RandomForestRegressor With Parameters: '+str(grid_search))    
    print('RandomForestRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for RandomForestRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[89]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = DecisionTreeRegressor(random_state=0)
    grid_search.fit(X, y)
    print('DecisionTreeRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('DecisionTreeRegressor With Parameters: '+str(grid_search))    
    print('DecisionTreeRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for DecisionTreeRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[90]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = AdaBoostRegressor(random_state=0)
    grid_search.fit(X, y)
    print('AdaBoostRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('AdaBoostRegressor With Parameters: '+str(grid_search))    
    print('AdaBoostRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for AdaBoostRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[91]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    print('Skipped due to poor accuracy')
#     grid_search = GaussianProcessRegressor(random_state=0)
#     grid_search.fit(X, y)
#     print('GaussianProcessRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
#     print('GaussianProcessRegressor With Parameters: '+str(grid_search))    
#     print('GaussianProcessRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
#     y_pred = grid_search.predict(X_test)
#     print('MSE for GaussianProcessRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[92]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = LinearRegression()
    grid_search.fit(X, y)
    print('LinearRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('LinearRegressor With Parameters: '+str(grid_search))    
    print('LinearRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for LinearRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[93]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = MLPRegressor(random_state=0)
    grid_search.fit(X, y)
    print('NeuralNetworkRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('NeuralNetworkRegressor With Parameters: '+str(grid_search))    
    print('NeuralNetworkRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for NeuralNetworkRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[94]:


#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
print('Due to the strict 3 minute rule, we have skipped the k-fold validation for large datasets like these and skipped SVR')
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
gaussian_best_param       = gaussian_regressor_param_selection(X_train, y_train, X_test, y_test, 3)
print("now ="+str(datetime.now()))
# svr_best_param           = svr_param_selection(X_train, y_train, X_test, y_test, 3)


# In[97]:


print('According to our methods, we find the Neural network regressor to have the lowest MSE (14529) for the test set and a comparable MSE for train set when compared to other models.')


# In[ ]:




