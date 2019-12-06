#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from datetime import datetime


f = open("results.txt", "a")
f.write("*******************************************************************************")
f.write("\nResults printed below are for Regression Data set 5 Facebook Metrics\n" )
f.write("*******************************************************************************\n")
f.write("\n")

# In[42]:


df_fb_metrics = pd.read_csv('dataset_Facebook.csv',sep=';')


# In[43]:


print(df_fb_metrics.info())
df_fb_metrics.replace(r'^\s*$', np.nan, regex=True, inplace = True)
df_fb_metrics.replace('?', np.nan, inplace = True)
print(df_fb_metrics.isnull().sum())


# In[44]:


#Since we are going to consider Total Interactions as the output variable (being the sum of Comment , Like, Share) 
#we can drop Comment , Like, Share and the null values associated with them.
individual_interaction_components = ['comment','like','share']
df_fb_metrics = df_fb_metrics.drop(columns=individual_interaction_components, axis=1)
df_fb_metrics.head()


# In[45]:


#We now have a categorical input variable Type.
df_fb_metrics["Type"].value_counts()


# In[46]:


#With just 4 different types of values we can plan to use OneHotEncoding which would add 4 new columns to our dataframe 
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(df_fb_metrics["Type"])
encoded_df = pd.DataFrame(lb_results, columns=lb_style.classes_)
df_fb_metrics = pd.concat([df_fb_metrics,encoded_df], axis=1) 
cols = list(df_fb_metrics)
cols.insert(1, cols.pop(cols.index('Link')))
cols.insert(2, cols.pop(cols.index('Photo')))
cols.insert(3, cols.pop(cols.index('Status')))
cols.insert(4, cols.pop(cols.index('Video')))
df_fb_metrics = df_fb_metrics.loc[:, cols]
df_fb_metrics = df_fb_metrics.drop(columns=['Type'],axis=1)
df_fb_metrics.head()


# In[47]:


#We now have a categorical input variable Type.
df_fb_metrics["Paid"].value_counts()


# In[48]:


df_fb_metrics[df_fb_metrics.isna().any(axis=1)]


# In[49]:


#Fill missing value with most common value
df_fb_metrics = df_fb_metrics.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_fb_metrics.shape


# In[50]:


#Removing Outliers remove records that are above the 90th percentile
outlier_cut_off_value = np.percentile(df_fb_metrics['Total Interactions'],90)
df_fb_metrics = df_fb_metrics[df_fb_metrics['Total Interactions']<outlier_cut_off_value]
df_fb_metrics.shape


# In[51]:


#Preparing the training and testing dataset.
X_train, X_test, y_train, y_test = train_test_split(df_fb_metrics.iloc[:, 0:18].values, df_fb_metrics.iloc[:, 18].values, test_size=0.33, random_state=0)
#Standardising the data set
scaler = StandardScaler()
scaler.fit(X_train)
x_train_scaled = scaler.transform(X_train)
x_test_scaled = scaler.transform(X_test)


# In[52]:


plt.figure(figsize=(12,10))
df_fb_metrics_corr = df_fb_metrics.corr()
mask = np.zeros_like(df_fb_metrics_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df_fb_metrics_corr,annot=True,cbar=False, mask=mask)


# In[53]:


#Running the dataset across various regressors


# In[62]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Kernels = ['linear', 'poly', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1]
    Gammas = [0.001, 0.01, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nSVR MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nSVR With Parameters: '+str(grid_search.best_params_))    
    f.write('\nSVR coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for SVR on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[63]:


def random_forest_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'n_estimators': Estimators, 'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nRandomForestRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nRandomForestRegressor With Parameters: '+str(grid_search.best_params_))    
    f.write('\nRandom Forest coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Random Forest Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[64]:


def decision_tree_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Max_features = ['auto', 'sqrt']
    Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
    param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nDecisionTreeRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nDecisionTreeRegressor With Parameters: '+str(grid_search.best_params_)) 
    f.write('\nDecision Tree coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Decision Tree Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[65]:


def ada_boost_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Estimators = np.arange(1,100,15)
    Learning_rates = [0.01,0.05,0.1,0.3]
    Losses = ['linear', 'square', 'exponential']
    param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'loss': Losses}
    grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=0),random_state=0), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nAdaBoostRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nAdaBoostRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nAdaBoost Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for AdaBoost Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[66]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    # kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nGaussianRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nGaussianRegressor With Parameters:'+str(grid_search.best_params_)) 
    f.write('\nGaussian Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for Gaussian Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[67]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nLinearRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nLinearRegressor With Parameters:'+str(grid_search.best_params_))  
    f.write('\nLinear Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for LinearRegressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[68]:


def neural_network_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Hidden_Layer_Sizes = [1, 5, (5,5), 10, (10,5)]
    Activations = ['logistic', 'tanh', 'relu']
    param_grid = {'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations}
    grid_search = GridSearchCV(MLPRegressor(max_iter=1000,learning_rate='adaptive',solver='lbfgs',random_state=0), param_grid, cv=nfolds, n_jobs=-1, iid=False)
    grid_search.fit(X, y)
    f.write('\nNeuralNetworkRegressor MSE Score for training data: '+str(grid_search.best_score_))
    f.write('\nNeuralNetworkRegressor With Parameters:'+str(grid_search.best_params_))
    f.write('\nNeural Network Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMSE for NeuralNetwork Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[69]:
f = open("results.txt", "a")

#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
f.write("\nnow ="+str(datetime.now()))
svr_best_param           = svr_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
random_forest_best_param = random_forest_regressor_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
decision_tree_best_param = decision_tree_regressor_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
ada_boost_best_param     = ada_boost_regressor_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
linear_best_param         = linear_regressor_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
neural_network_best_param = neural_network_regressor_param_selection(x_train_scaled, y_train, x_test_scaled, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
#gaussian_best_param       = gaussian_regressor_param_selection(X_train_scaled, y_train, X_test_scaled, 3)
#f.write("\nnow ="+str(datetime.now()))


# In[ ]:




