#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Basic imports for all datasets
import numpy as np 
import pandas as pd   # for data reading 
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics           # For accuracy_score
import sklearn.model_selection   # For GridSearchCV and RandomizedSearchCV
import scipy
import scipy.stats               # For reciprocal distribution
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, ExpSineSquared


# In[9]:


#Dataset 2 : Default of credit card clients

#Loading dataset
df = pd.read_excel('default of credit card clients.xls', skiprows=1)


# In[10]:


#Data cleaning part

#Features listed for X values
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

#Renamed Pay 0 to Pay 1 as column was poorly labeled
df = df.rename(columns={'PAY_0': 'PAY_1'})

#As 0,5,6 are not documented properly, we label them as others
fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
df.loc[fil, 'EDUCATION'] = 4

#As 0 is not documented properly, we label them as others
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

df.sample(5)


# In[11]:


#Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df[features], 
    df['default payment next month'] , 
    test_size=0.2, 
    random_state=0)


# In[17]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(45,55, 2))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[18]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
             "max_iter":np.arange(200,400, 75)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[14]:


#Decision tree classification
DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,4)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[15]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[19]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,25)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[20]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[21]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
# batch_size = [10, 20, 40, 60, 80, 100]
batch_size = [100]
# epochs = [10, 50, 100]
epochs = [10]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30] 
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum, 
              'activation' : activation, }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[22]:


#SVM classifier
svm_model = sklearn.svm.SVC()
# Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# Epsilons = [0.1,0.2,0.5,0.3]
# Cs = [0.001, 0.01, 0.1, 1, 10]
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : Gammas}

# mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
svm_model.fit(X_train, y_train)
print(svm_model)

y_pred = svm_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[1]:


print('According to our methods, we find that the most of the models (Adaboost, Logisitic regression, Decision trees and Random forest) have the same test accuaracy (roughly 82%). Hence, we suggest going for Logisitic regression because it provides the same level of accuracy with a much simpler model which so there\'s little chance to overfit')


# In[ ]:




