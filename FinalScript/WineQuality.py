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


# In[2]:

f = open("results.txt", "a")
f.write("*******************************************************************************\n")
f.write("\nResults printed below are for Regression Data set 1 Wine Quality\n" )
f.write("*******************************************************************************\n")
f.write("\n")


# Data-Set has two different wine types Red-White
#importing the Dataset
df_red_wine = pd.read_csv('winequality-red.csv',sep=';')
df_white_wine = pd.read_csv('winequality-white.csv',sep=';')

#adding a new column Type - 1(Red Wine), 2(White Wine), to datasets
df_red_wine['type'] = 1.0
df_white_wine['type'] = 2.0

#Combining datasets to one 
df_wine = pd.concat([df_red_wine, df_white_wine], axis=0)


# In[3]:


# A clean dataset with no NULL values and all numeric values
f.write(str(df_wine.isnull().sum()))
f.write("\n")
df_wine.info()


# In[4]:


df_wine.head()


# In[5]:


df_wine.tail()


# In[6]:


#Analysing Correlation between various features of the wine dataset
wine_corr = df_wine.corr()
figure, ax = plt.subplots(figsize = (10,10))
ax.set_title('Correlation Matrix for wine data-set')
matrix = sns.heatmap(wine_corr,ax=ax, annot= True)


# In[7]:


#Feature Selection by dropping features having correlation less than 0.05 with target variable quality
abs_val_wine_corr = wine_corr['quality'].drop('quality').abs()
imp_feature_idx = abs_val_wine_corr[abs_val_wine_corr > 0.05].index.values.tolist()


# In[8]:


#Preparing the final dataset
x = df_wine[imp_feature_idx] 
y = df_wine['quality']


# In[9]:


#Creating Traning and test set
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0,stratify=y)


# In[10]:


#Standardizing the prepared training and test data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


def svr_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Kernels = ['linear', 'poly', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1]
    Gammas = [0.001, 0.01, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('SVR Lowest MAE Score: '+str(grid_search.best_score_))
    f.write("\n")
    f.write('SVR With Parameters: '+str(grid_search.best_params_))    
    f.write("\n")
    return grid_search.best_params_


# In[12]:


def random_forest_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('RandomForestRegressor Lowest MAE Score: '+str(grid_search.best_score_))
    f.write("\n")
    f.write('RandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    f.write("\n")
    return grid_search.best_params_


# In[13]:


def decision_tree_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('DecisionTreeRegressor Lowest MAE Score: '+str(grid_search.best_score_))
    f.write("\n")
    f.write('DecisionTreeRegressor With Parameters: '+str(grid_search.best_params_))    
    f.write("\n")
    return grid_search.best_params_


# In[14]:


def ada_boost_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Learning_rates = [0.01,0.05,0.1,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('AdaBoostRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write("\n")
    f.write('AdaBoostRegressor With Parameters:'+str(grid_search.best_params_))   
    f.write("\n") 
    return grid_search.best_params_


# In[15]:


def gaussian_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq, kernel_expsine]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('GaussianRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write("\n")
    f.write('GaussianRegressor With Parameters:'+str(grid_search.best_params_))    
    f.write("\n")
    return grid_search.best_params_


# In[16]:


def linear_regressor_param_selection(X, y, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('LinearRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write("\n")
    f.write('LinearRegressor With Parameters:'+str(grid_search.best_params_))   
    f.write("\n") 
    return grid_search.best_params_


# In[17]:


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
    f.write('NeuralNetworkRegressor Lowest MAE Score:'+str(grid_search.best_score_))
    f.write("\n")
    f.write('NeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))   
    f.write("\n") 
    return grid_search.best_params_


# In[20]:


#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
f.write("\n")
f.write("now ="+str(datetime.now()))
f.write("\n")
svr_best_param           = svr_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
random_forest_best_param = random_forest_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
decision_tree_best_param = decision_tree_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
ada_boost_best_param     = ada_boost_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
linear_best_param         = linear_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
neural_network_best_param = neural_network_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")
gaussian_best_param       = gaussian_regressor_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))


# In[ ]:


#Checking MAE of each of the best regressors on test data


# In[18]:


f.write("now ="+str(datetime.now()))
f.write("\n")
svr_best_param           = svr_param_selection(X_train_scaled, y_train, 3)
f.write("now ="+str(datetime.now()))
f.write("\n")


# In[33]:


best_svr = SVR(C=1, gamma=0.1, kernel='rbf')
best_svr.fit(X_train_scaled, y_train)
y_pred = best_svr.predict(X_test_scaled)
f.write('MAE for SVR: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")
f.write(best_svr.score(X_test_scaled, y_test))
f.write("\n")


# In[34]:


best_decision_tree_regressor = DecisionTreeRegressor(max_features='auto', min_samples_leaf=0.05, random_state=0)
best_decision_tree_regressor.fit(X_train_scaled, y_train)
y_pred = best_decision_tree_regressor.predict(X_test_scaled)
f.write('MAE for Decision Tree Regressor: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")


# In[35]:


best_random_forest_regressor = RandomForestRegressor(max_features='auto', min_samples_leaf=0.01, n_estimators=91, random_state=0)
best_random_forest_regressor.fit(X_train_scaled, y_train)
y_pred = best_random_forest_regressor.predict(X_test_scaled)
f.write('MAE for Random Forest Regressor: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")


# In[36]:


best_ada_boost_regressor = AdaBoostRegressor(learning_rate=0.01, loss='linear', n_estimators=91, random_state=0)
best_ada_boost_regressor.fit(X_train_scaled, y_train)
y_pred = best_ada_boost_regressor.predict(X_test_scaled)
f.write('MAE for AdaBoost Regressor: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")


# In[37]:


best_linear_regressor = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
best_linear_regressor.fit(X_train_scaled, y_train)
y_pred = best_linear_regressor.predict(X_test_scaled)
f.write('MAE for Linear Regressor: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")


# In[38]:


best_neural_network_regressor = MLPRegressor(activation='logistic', alpha=0.002, hidden_layer_sizes=(10, 5), learning_rate='constant', learning_rate_init=0.01, random_state=0)
best_neural_network_regressor.fit(X_train_scaled, y_train)
y_pred = best_neural_network_regressor.predict(X_test_scaled)
f.write('MAE for Neural Network Regressor: '+str(mean_absolute_error(y_test, y_pred)))
f.write("\n")


# In[ ]:




