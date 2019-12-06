#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier


# In[29]:


#Dataset for Concrete strength
df_merck1 = pd.read_csv('ACT2_competition_training.csv')
df_merck1.head()


# In[30]:


df_merck2 = pd.read_csv('ACT4_competition_training.csv')
df_merck2.head()


# In[31]:


df_merck = pd.concat([df_merck1,df_merck2], axis=0, ignore_index=True)
df_merck.head()


# In[32]:


print(df_merck.info())
df_merck.replace(r'^\s*$', 0, regex=True, inplace = True)
df_merck.replace('?', 0, inplace = True)
df_merck.replace('[A-Z]+[0-9]*',0, inplace = True)
print(df_merck.isnull().sum())
df_merck.head()


# In[33]:


X = df_merck.loc[:, df_merck.columns != 'Act']
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
y = df_merck['Act']
X.head()


# In[34]:


X.shape


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=0)


# In[36]:


#Standardizing the prepared training and test data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[37]:


selector = VarianceThreshold(threshold=.50 * (1 - .5))
selector.fit(X_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)


# In[38]:


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


# In[39]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = RandomForestRegressor(random_state=0)
    grid_search.fit(X, y)
    print('RandomForestRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('RandomForestRegressor With Parameters: '+str(grid_search))    
    print('RandomForestRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for RandomForestRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[40]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = DecisionTreeRegressor(random_state=0)
    grid_search.fit(X, y)
    print('DecisionTreeRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('DecisionTreeRegressor With Parameters: '+str(grid_search))    
    print('DecisionTreeRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for DecisionTreeRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[41]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = AdaBoostRegressor(random_state=0)
    grid_search.fit(X, y)
    print('AdaBoostRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('AdaBoostRegressor With Parameters: '+str(grid_search))    
    print('AdaBoostRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for AdaBoostRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[42]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    print('Skipped due to poor accuracy')
#     grid_search = GaussianProcessRegressor(random_state=0)
#     grid_search.fit(X, y)
#     print('GaussianProcessRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
#     print('GaussianProcessRegressor With Parameters: '+str(grid_search))    
#     print('GaussianProcessRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
#     y_pred = grid_search.predict(X_test)
#     print('MSE for GaussianProcessRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[43]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = LinearRegression()
    grid_search.fit(X, y)
    print('LinearRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('LinearRegressor With Parameters: '+str(grid_search))    
    print('LinearRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for LinearRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[44]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    grid_search = MLPRegressor(random_state=0)
    grid_search.fit(X, y)
    print('NeuralNetworkRegressor MSE Score for training data: '+str(grid_search.score(X_test, y_test)))
    print('NeuralNetworkRegressor With Parameters: '+str(grid_search))    
    print('NeuralNetworkRegressor coefficient of determination R^2 on test data: '+str(grid_search.score(X_test, y_test)))
    y_pred = grid_search.predict(X_test)
    print('MSE for NeuralNetworkRegressor on test set: '+str(mean_squared_error(y_test, y_pred)))


# In[45]:


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
# print()
print("now ="+str(datetime.now()))


# In[ ]:


print('According to our methods, Decision tree regressor gave us the lowest MSE on test data but as the train error is quite low it is unlikely that this is the best model. We suggest either Neural network Regressor or the Adaboost regressor which have comparable train and test accuracy(Neural Network = 51% and Adaboost = 58%)')

