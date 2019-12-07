#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.svm
from sklearn.tree import DecisionTreeClassifier 
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.neural_network



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings


# In[28]:


#Dataset 1 : Diabetic Retinopathy
data = np.loadtxt('messidor_features.arff', delimiter=',', comments = '@', max_rows = 1000)
X_train, X_test, y_train, y_test = train_test_split(
    data[:,:19],data[:,19], test_size=0.2, random_state=0)


# In[29]:


f = open("results.txt", "a")
f.write("\nResults printed below are for Clasification Data set 1 Diabetic Retinopathy\n" )


# In[30]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(5,20, 2))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[31]:


#SVM
svm_model = sklearn.svm.SVC(kernel = 'linear')
svm_model.fit(X_train, y_train);
y_pred = svm_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[32]:


#Decision tree classification
DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[33]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[34]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[35]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30] 
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum, 
              'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[36]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
             "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[37]:


# #SVM classifier
# svm_model = sklearn.svm.SVC()
# Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# Epsilons = [0.1,0.2,0.5,0.3]
# Cs = [0.001, 0.01, 0.1, 1, 10]
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : Gammas, 'kernel' : Kernels}

# mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
# print(mdls.best_estimator_)

# y_pred = mdls.best_estimator_.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


# In[38]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[39]:


f.write("Best classifier is Neural Network with 74.5% accuracy\n")




#Dataset 2 : Default of credit card clients

#Loading dataset
df = pd.read_excel('default of credit card clients.xls', skiprows=1)


# In[3]:


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


# In[4]:


#Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(
                                                    df[features],
                                                    df['default payment next month'] ,
                                                    test_size=0.2,
                                                    random_state=0)


# In[5]:


f.write("\nResults printed below are for Clasification Data set 2 Default of Credit Card\n" )


# In[6]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,52, 5))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[7]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[8]:


#Decision tree classification
DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)

y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[9]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[ ]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[ ]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[ ]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30]
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum,
    'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[ ]:


#SVM classifier
svm_model = sklearn.svm.SVC()
Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Epsilons = [0.1,0.2,0.5,0.3]
Cs = [0.001, 0.01, 0.1, 1, 10]
Gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : Gammas}

mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = svm_model.predict(X_test)

y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[ ]:


f.write("Best classifier is Random Forest Classifier with 92.84% accuracy \n")




#Dataset 3


data = np.loadtxt('breast-cancer-wisconsin.data',dtype = 'str', delimiter=',')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data[data == '?'] = np.nan
data = imp.fit_transform(data)
data = data.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(
                                                    data[:,:10], data[:,10], test_size=0.2, random_state=0)


# In[15]:


f.write("\nResults printed below are for Clasification Data set 3 Breast Cancer\n" )


# In[16]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,52, 5))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[17]:


#SVM
svm_model = sklearn.svm.SVC(kernel = 'linear')
svm_model.fit(X_train, y_train);
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[18]:


#Decision tree classification
DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[19]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[20]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[21]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[22]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 2].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[23]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30]
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum,
    'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[24]:


# #SVM classifier
# svm_model = sklearn.svm.SVC()
# Kernels = ['linear']
# Epsilons = [0.1,0.2,0.5,0.3]
# Cs = [0.001, 0.01, 0.1, 1, 10]
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : Gammas, 'kernel': Kernels}

# mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
# print(mdls.best_estimator_)

# y_pred = mdls.best_estimator_.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


# In[25]:


f.write("Best classifier is Random Forest Classifier with 97.85% accuracy\n")



#Dataset 5


data = np.loadtxt('german.data-numeric',dtype = 'str')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data[data == '?'] = np.nan
data = imp.fit_transform(data)
data = data.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(
                                                    data[:,:24], data[:,24], test_size=0.2, random_state=0)


# In[45]:


f.write("\nResults printed below are for Clasification Data set 5 Statlog German\n" )


# In[46]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(5,20, 2))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[47]:


#Decision tree classification
DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")



# In[48]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[49]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[50]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30]
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum,
    'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=0,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[51]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[52]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 1].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[53]:


#SVM classifier
svm_model = sklearn.svm.SVC()
Kernels = ['linear']
Epsilons = [0.1,0.2,0.5,0.3]
Cs = [0.001, 0.01, 0.1, 1, 10]
Gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : Gammas, 'kernel': Kernels}

mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = svm_model.predict(X_test)

y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[54]:


f.write("Best classifier is Neural Network Classifier with 77.5% accuracy\n")




#Dataset 7






col_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
             'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']


# In[3]:


income_train_df = pd.read_csv('adult.data', sep=",\s", names=col_names, engine = 'python')
income_test_df = pd.read_csv('adult.test', sep=",\s", names=col_names, engine = 'python', skiprows=1)
income_df = pd.concat([income_train_df,income_test_df])
income_df.head()


# In[4]:


#First lets properly encode the target variable
income_df['income'] = income_df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})


# In[5]:


#Checking any null values
income_df.replace(r'^\s*$', np.nan, regex=True, inplace = True)
income_df.replace('?', np.nan, inplace = True)
print(income_df.isnull().sum())


# In[6]:



#We have three columns with high number of missing values - (workclass, occupation, native-country)
#We will simply impute them using a simple imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
missing_val_cols = ['workclass', 'occupation', 'native-country']
imputer.fit(income_df[["workclass"]])
income_df["workclass"]=imputer.transform(income_df[["workclass"]]).ravel()
imputer.fit(income_df[["occupation"]])
income_df["occupation"]=imputer.transform(income_df[["occupation"]]).ravel()
imputer.fit(income_df[["native-country"]])
income_df["native-country"]=imputer.transform(income_df[["native-country"]]).ravel()


# In[7]:


#Checking type of available features
print(income_df.info())
#As we can see there is a lot of categorical data preset in the data set which we need to encode to numeric values.
#So we define a custom multi col encoder class to do that.


# In[8]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        '''
            Transforms columns of X specified in self.columns using
            LabelEncoder(). If no columns specified, transforms all
            columns in X.
            '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[9]:


income_df = MultiColumnLabelEncoder(columns = list(set(income_df.columns) - set(income_df.describe().columns))).fit_transform(income_df)


# In[10]:


#Preparing training and testing datasets
income_data = income_df.values
income_data = income_data.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(income_data[:,:14],income_data[:,14], test_size=0.33, random_state=0)


# In[11]:


f.write("\nResults printed below are for Clasification Data set 7 Adult Data \n" )


# In[12]:


#k-Nearest neighbours classification
print("now ="+str(datetime.now()))
knn_model = KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,52,5))}
mdls = model_selection.GridSearchCV(knn_model, param_grid, verbose=1, cv=3, n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[13]:


#Logistic regression
print("now ="+str(datetime.now()))
logistic_model = linear_model.LogisticRegression(n_jobs=-1,random_state=0)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'saga'],
    "max_iter":np.arange(100,400, 100)}
mdls = model_selection.GridSearchCV(logistic_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[14]:


#Decision tree classification
print("now ="+str(datetime.now()))
DTC_model = DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
mdls = model_selection.GridSearchCV(DTC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[15]:


#Random forest classification
print("now ="+str(datetime.now()))
RFC_model = ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,5)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
mdls = model_selection.GridSearchCV(RFC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[16]:


#AdaBoost classification
print("now ="+str(datetime.now()))
ABC_model = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0),random_state=0)
Estimators = np.arange(50,110,10)
Learning_rates = [0.05,0.1,0.3,1]
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates}
mdls = model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[17]:


#Gaussian naive Bayes classification
print("now ="+str(datetime.now()))
zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5,, n_jobs=-1,iid=False).fit(X_train, y_train)
# print(mdls.best_estimator_)
y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[18]:


#Neural network classification
print("now ="+str(datetime.now()))
NNC_model = MLPClassifier()
Hidden_Layer_Sizes = [1, 5, (5,5), (10,5)]
Learning_rates_init = [0.001, 0.01, 0.1]
Activations = ['logistic', 'tanh', 'relu']
param_grid = {'learning_rate_init': Learning_rates_init, 'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations}
mdls = model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[19]:


#Standardizing the prepared training and test data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


# #SVM classifier
# print("now ="+str(datetime.now()))
# svm_model = svm.SVC()
# Kernels = ['poly', 'rbf']
# param_grid = {'kernel':Kernels}
# mdls = model_selection.GridSearchCV(svm_model, param_grid, verbose=1, cv=3, n_jobs=-1,iid=False).fit(X_train, y_train)
# print(mdls.best_estimator_)
# y_pred = svm_model.predict(X_test)
# y_train_pred = mdls.best_estimator_.predict(X_train)
# f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
# f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[21]:


f.write("Best classifier is Random forest classification is 84.07% accuracy\n")



#Dataset 8

f.write("\nResults printed below are for Clasification Data set 8 Yeast\n" )


# In[18]:


data = np.loadtxt('yeast.data',dtype = 'str')
columns = np.arange(10)
df = pd.DataFrame(data,columns=columns)
print(df.head())
df = MultiColumnLabelEncoder(columns = [0,9]).fit_transform(df)
data = df.values
data = data.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(
                                                    data[:,:9],data[:,9], test_size=0.2, random_state=0)


# In[19]:



# knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=35)
# knn_model.fit(X_train, y_train)
# y_pred = knn_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)

knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,10, 1))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")



# In[20]:


#Logistic regression
logistic_model = sklearn.linear_model.LogisticRegression(C = 35,fit_intercept=False, penalty='l2', solver='lbfgs',max_iter = 1000)
logistic_model.fit(X_train, y_train);
y_pred = logistic_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)

param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[21]:


#SVM classifier
svm_model = sklearn.svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[22]:


#Decision tree classification
# DTC_model = sklearn.tree.DecisionTreeClassifier()
# DTC_model.fit(X_train, y_train)
# y_pred = DTC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")




# In[23]:


#Random forest classification
# RFC_model = sklearn.ensemble.RandomForestClassifier()
# RFC_model.fit(X_train, y_train)
# y_pred = RFC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)



RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(95,102,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[24]:


#AdaBoost classification
# ABC_model = sklearn.ensemble.AdaBoostClassifier()
# ABC_model.fit(X_train, y_train)
# y_pred = ABC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)



ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[25]:


#Gaussian naive Bayes classification

class0_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
class1_prob = y_train[y_train == 1].shape[0]/y_train.shape[0]
class2_prob = y_train[y_train == 2].shape[0]/y_train.shape[0]
class3_prob = y_train[y_train == 3].shape[0]/y_train.shape[0]
class4_prob = y_train[y_train == 4].shape[0]/y_train.shape[0]
class5_prob = y_train[y_train == 5].shape[0]/y_train.shape[0]
class6_prob = y_train[y_train == 6].shape[0]/y_train.shape[0]
class7_prob = y_train[y_train == 7].shape[0]/y_train.shape[0]
class8_prob = y_train[y_train == 8].shape[0]/y_train.shape[0]
class9_prob = y_train[y_train == 9].shape[0]/y_train.shape[0]

prob = np.array([class0_prob,class1_prob, class2_prob, class3_prob, class4_prob, class5_prob, class6_prob,class7_prob,class8_prob,class9_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)
y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[26]:


#Neural network classification
# NNC_model = sklearn.neural_network.MLPClassifier()
# NNC_model.fit(X_train, y_train)
# y_pred = NNC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


NNC_model = sklearn.neural_network.MLPClassifier(early_stopping = False)
batch_size = [50, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [ 0.4, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30]
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum,
    'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[27]:


# #SVM classifier
# svm_model = sklearn.svm.SVC()
# Kernels = ['linear']
# Epsilons = [0.1,0.2,0.5,0.3]
# Cs = [0.001, 0.01, 0.1, 1, 10]
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : Gammas, 'kernel': Kernels}

# mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
# print(mdls.best_estimator_)

# y_pred = mdls.best_estimator_.predict(X_test)
# f.write("Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[28]:


f.write("Best classifier is Random Forest Classifier with 61.61% accuracy and accuracy for this particulaer could not be improved beyond this. (and the authors have also stated that 55% was the best accuracy they could achieve)\n")



#Dataset 9


thoracic_data = np.loadtxt('ThoraricSurgery.arff',dtype = 'str',delimiter=',',comments='@')
thoracic_df = pd.DataFrame(data=thoracic_data[:,:])
thoracic_df.head()


# In[37]:


col_names = ['Diagnosis','FVC','FEV1','Performance','Pain','Haemoptysis','Dyspnoea','Cough','Weakness','Tumor_Size',
             'Diabetes_Mellitus','MI_6mo','PAD','Smoking','Asthma','Age','Risk1YrDeath']
#col_names = ['DGN','PRE4','PRE5','PRE6','PRE7','PRE8','PRE9','PRE10','PRE11','PRE14','PRE17','PRE19','PRE25','PRE30','PRE32','Age','Risk1Yr']
thoracic_df.columns = col_names
thoracic_df.head()


# In[38]:



thoracic_df.info()


# In[39]:


#Checking any null values
thoracic_df.replace(r'^\s*$', np.nan, regex=True, inplace = True)
thoracic_df.replace('?', np.nan, inplace = True)
print(thoracic_df.isnull().sum())


# In[40]:



#Handling True/False data values by converting them to 1/0.
t_f_cols = ['Pain', 'Haemoptysis', 'Dyspnoea', 'Cough', 'Weakness', 'Diabetes_Mellitus', 'MI_6mo', 'PAD', 'Smoking', 'Asthma', 'Risk1YrDeath']
thoracic_df[t_f_cols] = (thoracic_df[t_f_cols] == 'T').astype(float)
thoracic_df.head()


# In[41]:


#Diagnosis,Performance,Tumor_Size has alphanumerical categorical data with consistent numeric part we can only extract the numeric part.
thoracic_df['Diagnosis'] = thoracic_df.Diagnosis.str[-1:].astype(float)
thoracic_df['Performance'] = thoracic_df.Performance.str[-1:].astype(float)
thoracic_df['Tumor_Size'] = thoracic_df.Tumor_Size.str[-1:].astype(float)

#Convertig other numeric types to float
thoracic_df['FVC'] = thoracic_df.FVC.str[-1:].astype(float)
thoracic_df['FEV1'] = thoracic_df.FEV1.str[-1:].astype(float)
thoracic_df['Age'] = thoracic_df.Age.str[-1:].astype(float)

thoracic_df.head()


# In[42]:



#Prepare Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(thoracic_df.iloc[:,:16],thoracic_df.iloc[:,16], test_size=0.33, random_state=0)


# In[43]:


f.write("\nResults printed below are for Clasification Data set 9 Thoracic Surgery\n" )


# In[44]:


print("now ="+str(datetime.now()))
knn_model = KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,52,5))}
mdls = model_selection.GridSearchCV(knn_model, param_grid, verbose=1, cv=3, n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[45]:


# #SVM classifier
# print("now ="+str(datetime.now()))
# svm_model = svm.SVC()
# Kernels = ['linear', 'poly', 'rbf']
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'kernel':Kernels, 'gamma' : Gammas}
# mdls = model_selection.GridSearchCV(svm_model, param_grid, verbose=1, cv=3, n_jobs=-1,iid=False).fit(X_train, y_train)
# print(mdls.best_estimator_)
# y_pred = svm_model.predict(X_test)
# y_train_pred = mdls.best_estimator_.predict(X_train)
# f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
# f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[46]:



#Decision tree classification
print("now ="+str(datetime.now()))
DTC_model = DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
mdls = model_selection.GridSearchCV(DTC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[47]:


#Random forest classification
print("now ="+str(datetime.now()))
RFC_model = ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,5)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, 'min_samples_leaf': Min_samples_leafs}
mdls = model_selection.GridSearchCV(RFC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[48]:


#AdaBoost classification
print("now ="+str(datetime.now()))
ABC_model = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0),random_state=0)
Estimators = np.arange(50,110,10)
Learning_rates = [0.05,0.1,0.3,1]
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates}
mdls = model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[49]:


#Logistic regression
print("now ="+str(datetime.now()))
logistic_model = linear_model.LogisticRegression(n_jobs=-1,random_state=0)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'saga'],
    "max_iter":np.arange(100,400, 100)}
mdls = model_selection.GridSearchCV(logistic_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[50]:


#Gaussian naive Bayes classification
print("now ="+str(datetime.now()))
zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5,, n_jobs=-1,iid=False).fit(X_train, y_train)
# print(mdls.best_estimator_)
y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[51]:


#Neural network classification
print("now ="+str(datetime.now()))
NNC_model = MLPClassifier(max_iter=500)
Hidden_Layer_Sizes = [1, 5, 10, (5,5), (10,5)]
Learning_rates = ['constant','adaptive']
Learning_rates_init = [0.001, 0.01, 0.1]
Activations = ['logistic', 'tanh', 'relu']
Alphas = [0.0001,0.002]
param_grid = {'learning_rate': Learning_rates, 'learning_rate_init': Learning_rates_init, 'hidden_layer_sizes': Hidden_Layer_Sizes, 'activation': Activations, 'alpha': Alphas}
mdls = model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=3,n_jobs=-1,iid=False).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[52]:


# #SVM classifier
# print("now ="+str(datetime.now()))
# svm_model = svm.SVC()
# Kernels = ['linear', 'poly', 'rbf']
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'kernel':Kernels, 'gamma' : Gammas}
# mdls = model_selection.GridSearchCV(svm_model, param_grid, verbose=1, cv=3, n_jobs=-1,iid=False).fit(X_train, y_train)
# print(mdls.best_estimator_)
# y_pred = svm_model.predict(X_test)
# y_train_pred = mdls.best_estimator_.predict(X_train)
# f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
# f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[53]:


f.write("Best classifier is Random forest classification with 84.61%\n")



#Dataset 10


f.write("\nResults printed below are for Clasification Data set 10 Seismic Bumps\n" )


# In[19]:


data = np.loadtxt('seismic-bumps.arff',dtype = 'str',delimiter=',',comments=('@','%'))
columns = np.arange(19)
df = pd.DataFrame(data,columns=columns)
df = MultiColumnLabelEncoder(columns = [0,1,2,7]).fit_transform(df)
data = df.values
data = data.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(
                                                    data[:,:18],data[:,18], test_size=0.2, random_state=42)


# In[20]:


#k-Nearest neighbours classification
# knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=35)
# knn_model.fit(X_train, y_train)
# y_pred = knn_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)



knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,30, 3))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for KNN is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[21]:


#Logistic regression
logistic_model = sklearn.linear_model.LogisticRegression(C = 35,fit_intercept=False, penalty='l2', solver='lbfgs',max_iter = 1000)
logistic_model.fit(X_train, y_train);
y_pred = logistic_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Logistic regression is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[22]:


#SVM classifier
svm_model = sklearn.svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
y_pred = svm_model.predict(X_test)

y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[23]:


#Decision tree classification
# DTC_model = sklearn.tree.DecisionTreeClassifier()
# DTC_model.fit(X_train, y_train)
# y_pred = DTC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


DTC_model = sklearn.tree.DecisionTreeClassifier(random_state=0)
Max_features = ['auto', 'sqrt', 'log2']
Max_depths = np.arange(1,34,2)
Min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
Min_samples_leafs = np.linspace(0.01, 0.05, 5, endpoint=True)
param_grid = {'max_features': Max_features, 'max_depth': Max_depths,  'min_samples_split': Min_samples_splits, 'min_samples_leaf': Min_samples_leafs}

mdls = sklearn.model_selection.GridSearchCV(DTC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Decision Tree Classifier is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[24]:


#Random forest classification
# RFC_model = sklearn.ensemble.RandomForestClassifier()
# RFC_model.fit(X_train, y_train)
# y_pred = RFC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)



RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=0,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Random forest classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[25]:


#AdaBoost classification
# ABC_model = sklearn.ensemble.AdaBoostClassifier()
# ABC_model.fit(X_train, y_train)
# y_pred = ABC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)



ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for AdaBoost classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[26]:


#Gaussian naive Bayes classification

zero_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
one_prob = 1 - zero_prob
prob = np.array([zero_prob,one_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Gaussian naive Bayes classification is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[27]:


#Neural network classification
# NNC_model = sklearn.neural_network.MLPClassifier()
# NNC_model.fit(X_train, y_train)
# y_pred = NNC_model.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)


NNC_model = sklearn.neural_network.MLPClassifier()
batch_size = [50, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1]
momentum = [ 0.4, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30]
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum,
    'activation' : activation, 'alpha':alpha }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
y_train_pred = mdls.best_estimator_.predict(X_train)
f.write("Train Accuracy for Neural network classification Classifier is " + str( sklearn.metrics.accuracy_score(y_train, y_train_pred ) * 100 ) + "%\n")
f.write("Test Accuracy for Neural network classification is "+ str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[28]:


# #SVM classifier
# svm_model = sklearn.svm.SVC()
# Kernels = ['linear']
# Epsilons = [0.1,0.2,0.5,0.3]
# Cs = [0.001, 0.01, 0.1, 1, 10]
# Gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : Gammas, 'kernel': Kernels}

# mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
# print(mdls.best_estimator_)

# y_pred = mdls.best_estimator_.predict(X_test)
# f.write("Accuracy for SVM is " + str( sklearn.metrics.accuracy_score(y_test, y_pred ) * 100 ) + "%\n")


# In[29]:


f.write("Best classifier is Random Forest Classifier with 92.84% accuracy \n")
f.close()









