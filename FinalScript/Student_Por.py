#!/usr/bin/env python
# coding: utf-8

# In[16]:


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


f = open("results.txt", "a")
f.write("*******************************************************************************")
f.write("\nResults printed below are for Regression Data set 6 Student Portuguese\n" )
f.write("*******************************************************************************\n")
f.write("\n")


# In[17]:


student_marks = pd.read_csv('student-por.csv',sep=';',  skipinitialspace=True)
student_marks.head()


# In[18]:



#Data Cleaning

# Select only categorical variables
category_df = student_marks.select_dtypes(include=['object'])

# One hot encode the variables
enocoded_df = pd.get_dummies(category_df)

# Put the grade back in the dataframe
enocoded_df['G3'] = student_marks['G3']

# Find correlations with grade
enocoded_df.corr()['G3'].sort_values()

# selecting the most correlated values and dropping the others
labels = student_marks['G3']

# drop the school and grade columns
student_marks = student_marks.drop(['school', 'G1', 'G2'], axis='columns')
    
# One-Hot Encoding of Categorical Variables
student_marks = pd.get_dummies(student_marks)

# Find correlations with the Grade
correlated_data = student_marks.corr().abs()['G3'].sort_values(ascending=False)

# Maintain the top 8 most correlation features with Grade
correlated_data = correlated_data[:9]


# In[19]:


student_marks = student_marks.loc[:, correlated_data.index]
student_marks.head()


# In[20]:


#Dropping higher_no as it is a complement of higher_yes
student_marks = student_marks.drop('higher_no', axis='columns')
student_marks.head()
X_train, X_test, y_train, y_test = train_test_split(student_marks, labels, test_size = 0.25, random_state=0)


# In[21]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Kernels = ['linear', 'poly', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1]
    Gammas = [0.001, 0.01, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nSVR MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nSVR With Parameters: '+str(grid_search.best_params_))    
    f.write('\nSVR coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for SVR on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[22]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nRandomForestRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nRandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    f.write('\nRandom Forest coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Random Forest Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[23]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nDecisionTreeRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nDecisionTreeRegressor With Parameters: '+str(grid_search.best_params_)) 
    f.write('\nDecision Tree coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Decision Tree Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[24]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Learning_rates = [0.01,0.05,0.1,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nAdaBoostRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nAdaBoostRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nAdaBoost Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for AdaBoost Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[25]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq, kernel_expsine]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nGaussianRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nGaussianRegressor With Parameters:'+str(grid_search.best_params_)) 
    f.write('\nGaussian Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Gaussian Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[26]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nLinearRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nLinearRegressor With Parameters:'+str(grid_search.best_params_))  
    f.write('\nLinear Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for LinearRegressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[27]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Hidden_Layer_Sizes = [1, 5, (5,5), (10,5)]
    Activations = ['logistic', 'relu']
    param_grid = {'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations}
    grid_search = GridSearchCV(MLPRegressor(max_iter=900,random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nNeuralNetworkRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nNeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nNeural Network Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for NeuralNetwork Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[28]:


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


# In[29]:


f.write("\nnow ="+str(datetime.now()))
svr_best_param           = svr_param_selection(X_train, y_train, X_test, y_test, 3)
f.write("\nnow ="+str(datetime.now()))

