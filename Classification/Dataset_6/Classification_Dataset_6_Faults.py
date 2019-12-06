#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Dataset 6 : Faults 

#Loading dataset
np.random.seed(0)
data = np.loadtxt('Faults.NNA')
df = pd.DataFrame(data)
df.head()
display(df.describe(include="all"))


# In[3]:


#Splitting training and testing data
y_dataframe =data[:,27:34]
print(y_dataframe.shape)
y = []
for i in range(y_dataframe.shape[0]):
    if y_dataframe[i][0] == 1:
        y.append(0)
#         print("Heree")
    elif y_dataframe[i][1] == 1:
        y.append(1)
    elif y_dataframe[i][2] == 1:
        y.append(2)
    elif y_dataframe[i][3] == 1:
        y.append(3)
    elif y_dataframe[i][4] == 1:
        y.append(4)
    elif y_dataframe[i][5] == 1:
        y.append(5)
    elif y_dataframe[i][6] == 1:
        y.append(6)
#     print("Her"+str(i))
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    data[:,0:27], 
    y, 
    test_size=0.2, 
    random_state=0)
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)


# In[4]:


#k-Nearest neighbours classification
knn_model = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
param_grid = {'n_neighbors':(np.arange(2,52, 5))}

mdls = sklearn.model_selection.GridSearchCV(knn_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)
y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[5]:


#Logistic regression (for classification)
#Fit_intercept is set to True because we don't have bias
# logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
logistic_model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
param_grid = { "fit_intercept":[True], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
             "max_iter":np.arange(100,400, 100)}


mdls = sklearn.model_selection.GridSearchCV(logistic_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[6]:


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
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[7]:


#Random forest classification
RFC_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
Estimators = np.arange(100,105,1)
Max_features = ['auto', 'sqrt', 'log2']
param_grid = {'n_estimators': Estimators,'max_features': Max_features, }

mdls = sklearn.model_selection.GridSearchCV(RFC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[8]:


#AdaBoost classification
ABC_model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
Estimators = np.arange(50,100,10)
Learning_rates = [0.01,0.05,0.1,0.3,1]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'n_estimators': Estimators, 'learning_rate': Learning_rates, 'algorithm': algorithm}

mdls = sklearn.model_selection.GridSearchCV(ABC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[9]:


#Gaussian naive Bayes classification

class1_prob = y_train[y_train == 0].shape[0]/y_train.shape[0]
class2_prob = y_train[y_train == 1].shape[0]/y_train.shape[0]
class3_prob = y_train[y_train == 2].shape[0]/y_train.shape[0]
class4_prob = y_train[y_train == 3].shape[0]/y_train.shape[0]
class5_prob = y_train[y_train == 4].shape[0]/y_train.shape[0]
class6_prob = y_train[y_train == 5].shape[0]/y_train.shape[0]
class7_prob = y_train[y_train == 6].shape[0]/y_train.shape[0]

prob = np.array([class1_prob, class2_prob, class3_prob, class4_prob, class5_prob, class6_prob,class7_prob])
GNB_model = sklearn.naive_bayes.GaussianNB(priors = prob)
GNB_model.fit(X_train, y_train)
# mdls = sklearn.model_selection.GridSearchCV(GNB_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
# print(mdls.best_estimator_)

y_pred = GNB_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[10]:


#Neural network classification
NNC_model = sklearn.neural_network.MLPClassifier()
# batch_size = [10, 20, 40, 60, 80, 100]
batch_size = [100]
# epochs = [10, 50, 100]
epochs = [10]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [1, 5, 10, 15, 20, 25, 30] 
activation = ['identity', 'logistic', 'tanh', 'relu']
alpha = [0.0001,0.002]
param_grid = {'batch_size':batch_size,  'momentum':momentum, 
              'activation' : activation, }

mdls = sklearn.model_selection.GridSearchCV(NNC_model, param_grid, verbose=1,cv=5).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[11]:


#SVM classifier
svm_model = sklearn.svm.SVC()
Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Epsilons = [0.1,0.2,0.5,0.3]
Cs = [0.001, 0.01, 0.1, 1, 10]
Gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : Gammas}

mdls = sklearn.model_selection.GridSearchCV(svm_model, param_grid, verbose=1,cv=3).fit(X_train, y_train)
print(mdls.best_estimator_)

y_pred = mdls.best_estimator_.predict(X_test)
sklearn.metrics.accuracy_score(y_test, y_pred)


# In[1]:


print('According to our methods, we find that Random forest is the only classifier with a respectable accuracy i.e. 75% and hence our choice for this dataset')


# In[ ]:




