#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from datetime import datetime

f = open("results.txt", "a")
f.write("*******************************************************************************")
f.write("\nResults printed below are for Regression Data set 3 Parkinson speech\n" )
f.write("*******************************************************************************\n")
f.write("\n")

# In[2]:


#importing the Dataset
df_parkinson = pd.read_csv('train_data.txt',sep=',', header=None)


# In[3]:


df_parkinson.head()


# In[4]:


#Based on the Attribute Information given we first remove the Non-Predictive features.
non_predictive_features = [0,28]
df_parkinson = df_parkinson.drop(columns=non_predictive_features, axis=1)
df_parkinson.head()


# In[5]:


df_parkinson.info()


# In[6]:


#Analysing Correlation between various features of the wine dataset
parkinson_corr = df_parkinson.corr()
figure, ax = plt.subplots(figsize = (10,10))
ax.set_title('Correlation Matrix for parkinson data-set')
matrix = sns.heatmap(parkinson_corr,ax=ax, annot= True)


# In[7]:


#Feature Selection by dropping features having correlation less than 0.05 with target variable quality
abs_val_parkinson_corr = parkinson_corr[27].drop(27).abs()
imp_feature_idx = abs_val_parkinson_corr[abs_val_parkinson_corr > 0.05].index.values.tolist()


# In[8]:


#Preparing the final dataset
x = df_parkinson[imp_feature_idx] 
y = df_parkinson[27]
#Creating Traning and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,stratify=y)


# In[9]:


#Standardizing the prepared training and test data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[10]:


def svr_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Kernels = ['linear', 'poly', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1]
    Gammas = [0.001, 0.01, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nSVR Lowest MAE Score: '+str(grid_search.best_score_))
    f.write('\nSVR With Parameters: '+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[11]:


def random_forest_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nRandomForestRegressor Lowest MAE Score: '+str(grid_search.best_score_))
    f.write('\nRandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[12]:


def decision_tree_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nDecisionTreeRegressor Lowest MAE Score: '+str(grid_search.best_score_))
    f.write('\nDecisionTreeRegressor With Parameters: '+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[13]:


def ada_boost_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Learning_rates = [0.01,0.05,0.1,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nAdaBoostRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write('\nAdaBoostRegressor With Parameters:'+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[14]:


def gaussian_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    # kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nGaussianRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write('\nGaussianRegressor With Parameters:'+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[15]:


def linear_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nLinearRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write('\nLinearRegressor With Parameters:'+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[16]:


def neural_network_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Learning_rates = ['constant','adaptive']
    Learning_rates_init = [0.001, 0.01, 0.1, 0.3]
    Hidden_Layer_Sizes = [1, 5, 10, (5,5), (10,5)]
    Activations = ['logistic', 'tanh', 'relu']
    Alphas = [0.0001,0.002]
    param_grid = {'learning_rate': Learning_rates, 'learning_rate_init': Learning_rates_init, 'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations, 'alpha': Alphas}
    grid_search = GridSearchCV(MLPRegressor(max_iter=900), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nNeuralNetworkRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write('\nNeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))    
    return grid_search.best_params_


# In[17]:


#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
f.write("\nnow ="+str(datetime.now()))
svr_best_param           = svr_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
random_forest_best_param = random_forest_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
decision_tree_best_param = decision_tree_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
ada_boost_best_param     = ada_boost_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
linear_best_param         = linear_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
neural_network_best_param = neural_network_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))
gaussian_best_param       = gaussian_regressor_param_selection(x_train_scaled, y_train, 3)
f.write("\nnow ="+str(datetime.now()))


# In[18]:


best_svr = SVR(C=1, gamma=0.001, kernel='linear')
best_svr.fit(x_train_scaled, y_train)
y_pred = best_svr.predict(x_test_scaled)
f.write('\nMAE for SVR: '+str(mean_absolute_error(y_test, y_pred)))

best_decision_tree_regressor = DecisionTreeRegressor(max_features='auto', min_samples_leaf=0.05, random_state=0)
best_decision_tree_regressor.fit(x_train_scaled, y_train)
y_pred = best_decision_tree_regressor.predict(x_test_scaled)
f.write('\nMAE for Decision Tree Regressor: '+str(mean_absolute_error(y_test, y_pred)))

best_random_forest_regressor = RandomForestRegressor(max_features='sqrt', min_samples_leaf=0.01, n_estimators=91, random_state=0)
best_random_forest_regressor.fit(x_train_scaled, y_train)
y_pred = best_random_forest_regressor.predict(x_test_scaled)
f.write('\nMAE for Random Forest Regressor: '+str(mean_absolute_error(y_test, y_pred)))

best_ada_boost_regressor = AdaBoostRegressor(learning_rate=0.1, loss='linear', n_estimators=61, random_state=0)
best_ada_boost_regressor.fit(x_train_scaled, y_train)
y_pred = best_ada_boost_regressor.predict(x_test_scaled)
f.write('\nMAE for AdaBoost Regressor: '+str(mean_absolute_error(y_test, y_pred)))

best_linear_regressor = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
best_linear_regressor.fit(x_train_scaled, y_train)
y_pred = best_linear_regressor.predict(x_test_scaled)
f.write('\nMAE for Linear Regressor: '+str(mean_absolute_error(y_test, y_pred)))

best_neural_network_regressor = MLPRegressor(activation='logistic', alpha=0.002, hidden_layer_sizes=10, learning_rate='constant', learning_rate_init=0.01, random_state=0)
best_neural_network_regressor.fit(x_train_scaled, y_train)
y_pred = best_neural_network_regressor.predict(x_test_scaled)
f.write('\nMAE for Neural Network Regressor: '+str(mean_absolute_error(y_test, y_pred)))

best_gaussian_regressor = GaussianProcessRegressor(kernel=1**2 * RationalQuadratic(alpha=0.1, length_scale=1))
best_gaussian_regressor.fit(x_train_scaled, y_train)
y_pred = best_gaussian_regressor.predict(x_test_scaled)
f.write('\nMAE for Gaussian Regressor: '+str(mean_absolute_error(y_test, y_pred)))


# In[ ]:




