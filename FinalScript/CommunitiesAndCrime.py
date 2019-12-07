#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
f.write("\nResults printed below are for Regression Data set 2 Communities\n" )
f.write("*******************************************************************************\n")
f.write("\n")

# In[11]:


attribute_names = pd.read_csv('attributes.txt', delim_whitespace = True, header=None, skipinitialspace=True)
df_crime = pd.read_csv('communities.data',sep=',',names=attribute_names[0], header=None, skipinitialspace=True)


# In[12]:


df_crime.head()


# In[13]:


f.write(str(df_crime.info()))
df_crime.replace(r'^\s*$', np.nan, regex=True, inplace = True)
df_crime.replace('?', np.nan, inplace = True)
f.write(str(df_crime.isnull().sum()))


# In[14]:


#As shown above dataset has lot of features with high number of missing values and needs some preprocessing.


# In[15]:


#Based on the Attribute Information given in supporting file communities.names we first remove the Non-Predictive features.
non_predictive_features = ['state','county','community','communityname','fold']
df_crime = df_crime.drop(columns=non_predictive_features, axis=1)
df_crime.head()


# In[16]:


#Analysing the misssing values
cols_with_missing_vals = df_crime.columns[df_crime.isnull().any()]
f.write('\nTotal Number of records:'+str(df_crime.shape[0]))
f.write('\nNumber of cols with missing values:'+str(cols_with_missing_vals.shape[0]))


# In[17]:


df_crime[cols_with_missing_vals].describe()


# In[18]:


# Each of the 22 out of the 23 cols have approx 84% values missings i.e. (1675/1994). So it makes sense to drop these columns.
# Col 'OtherPerCap' has only one missing value. So we replace it with the mean.


# In[19]:


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df_crime[['OtherPerCap']])
df_crime[['OtherPerCap']] = imp.transform(df_crime[['OtherPerCap']])
df_crime = df_crime.dropna(axis=1)
df_crime.head()


# In[20]:


#Preparing the training and testing dataset.
X_train, X_test, y_train, y_test = train_test_split(df_crime.iloc[:, 0:100].values, df_crime.iloc[:, 100].values, test_size=0.33, random_state=0)
#Standardising the data set
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


#Goal attribute to be predicted - ViolentCrimesPerPop: Total number of violent crimes per 100K popuation (numeric - decimal)


# In[22]:


crime_corr = df_crime.corr()
fig = plt.figure(figsize = (15,15))
mask = np.zeros_like(crime_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(crime_corr,mask=mask)
plt.title('Correlation Matrix for crime data-set')
plt.show()


# In[23]:


#With the number of features too high and a high degree of co-relation between some of the features 
#it makes sense to reduce this MultiCollinearity by doing dimensionality reduction (applying PCA).


# In[24]:


pca = PCA()
X_train = pca.fit_transform(X_train)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('CommunityCrime Dataset Explained Variance')
plt.show()


# In[25]:


#So anywhere close 40 features we retain more than 90% of the variance in our data set. 
#Setting number of Principal Componenets = 40 (Features = 40)


# In[26]:


num_pc = 40
pca = PCA(n_components = num_pc)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
f.write('\nDataset variance percentage retained: '+str(np.sum(pca.explained_variance_ratio_) * 100))


# In[27]:


#Running the dataset across various regressors


# In[28]:


def svr_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    Kernels = ['linear', 'poly', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1]
    Gammas = [0.001, 0.01, 0.1]
    param_grid = {'kernel':Kernels, 'C': Cs, 'gamma' : Gammas}
    grid_search = GridSearchCV(SVR(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nSVR MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nSVR With Parameters: '+str(grid_search.best_params_))    
    f.write('\nSVR coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for SVR on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[29]:


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


# In[30]:


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


# In[36]:


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


# In[37]:


def gaussian_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    kernel_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    # kernel_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    Kernels = [kernel_rbf, kernel_rq]
    param_grid = {'kernel': Kernels}
    grid_search = GridSearchCV(GaussianProcessRegressor(random_state=0), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nGaussianRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nGaussianRegressor With Parameters:'+str(grid_search.best_params_)) 
    f.write('\nGaussian Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for Gaussian Regressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[38]:


def linear_regressor_param_selection(X, y, X_test, y_test, nfolds):
    f = open("results.txt", "a")
    param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    f.write('\nLinearRegressor MAE Score for training data: '+str(grid_search.best_score_))
    f.write('\nLinearRegressor With Parameters:'+str(grid_search.best_params_))  
    f.write('\nLinear Regressor coefficient of determination R^2 on test data: '+str(grid_search.best_estimator_.score(X_test, y_test)))
    y_pred = grid_search.best_estimator_.predict(X_test)
    f.write('\nMAE for LinearRegressor on test set: '+str(mean_absolute_error(y_test, y_pred)))


# In[39]:


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


# In[40]:


#Using the 3-Fold HyperParam Search to evaluate the best hyperparams for each model
f.write("\nnow ="+str(datetime.now()))
svr_best_param           = svr_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
random_forest_best_param = random_forest_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
decision_tree_best_param = decision_tree_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
ada_boost_best_param     = ada_boost_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
linear_best_param         = linear_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
neural_network_best_param = neural_network_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))
gaussian_best_param       = gaussian_regressor_param_selection(X_train_pca, y_train, X_test_pca, y_test, 3)
f.write("\nnow ="+str(datetime.now()))


# In[ ]:




