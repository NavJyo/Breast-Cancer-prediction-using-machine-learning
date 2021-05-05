#!/usr/bin/env python
# coding: utf-8

# ### Importing Libaries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import sklearn
import scipy


sns.set()


# ---

# ### Data

# In[2]:


data = pd.read_csv('Breast_cancer_data (1).csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# ---

# ### EDA

# In[6]:


data.describe().T


# **Histogram Plot**

# In[7]:


data_feature = data.columns

for feature in data_feature:
    p = sns.distplot(a = data[feature])
    plt.show()


# **Removal of Zeros**\
# \
# Since there are many zeros in data and values of 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' cannot be zero,
# Therefore, Converriting Zeros into NaN value

# In[8]:


#data_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# In[9]:


#data[data_zeros] = np.where((data[data_zeros] == 0), np.nan, data[data_zeros])


# In[10]:


data.isnull().sum()


# In[11]:


# for feature in data_feature:
#     plt.hist(data[feature])
#     plt.show()
p = data.hist(figsize = (20,20))


# **Handling Missing Values**\
# \
# Filling NaN values with suitable mean and median values

# In[12]:


data.describe().T


# In[13]:


#data['Glucose'] = data['Glucose'].fillna(data['Glucose'].mean())
# data.isnull().sum()


# In[14]:


#data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].mean())
# data.isnull().sum()


# In[15]:


#sns.boxplot(y = 'SkinThickness', data = data)


# In[16]:


#data['SkinThickness'].mean(), data['SkinThickness'].median() 


# In[17]:


#data['SkinThickness'] = data['SkinThickness'].fillna(data['SkinThickness'].median())
# data.isnull().sum()


# In[18]:


#data['Insulin'].mean(), data['Insulin'].median() 


# In[19]:


#data['Insulin'] = data['Insulin'].fillna(data['Insulin'].median())
# data.isnull().sum()


# In[20]:


#data['BMI'].mean(), data['BMI'].median() 


# In[21]:


#data['BMI'] = data['BMI'].fillna(data['BMI'].median())
# data.isnull().sum()


# In[23]:


for i in range(6):
    print(data.columns[i])


# In[24]:


# for feature in data.columns:
#     plt.hist(data[feature])
#     plt.title(feature)
#     plt.show()
p = data.hist(figsize = (20,20))


# **Pair Plot to see Distribution of all data at a time and dependencies**

# In[25]:


sns.pairplot(data =data, hue = 'diagnosis')
plt.show()


# **Heat Map**\
# \
# Gives Relation of different attribute with each other

# In[26]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu")
plt.show()


# In[27]:


from scipy import stats
for feature in data.columns:
    stats.probplot(data[feature], plot = plt)
    plt.title(feature)
    plt.show()


# ---

# ## Standardizing Data

# In[28]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


# In[29]:


data.head()


# In[30]:


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[31]:


X.head()


# In[32]:


y.head()


# In[33]:


# X[:] = scale.fit_transform(X[:])


# In[34]:


X.head()


# ---

# ## Splitting data into train and test set

# In[35]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[36]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler() 


# In[37]:


from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
scaler.fit(data) 
scaled_data=scaler.transform(data) 
scaled_data  


# In[43]:


from sklearn.decomposition import PCA 

pca=PCA(n_components=5) 

pca.fit(scaled_data) 
x_pca=pca.transform(scaled_data)


# In[44]:


scaled_data.shape 


# In[45]:


x_pca.shape  


# In[46]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 0) 


# ---

# ## Fitting data in various models

# In[47]:


def svm_classifier(X_train, X_test, y_train, y_test):
    
    classifier_svm = SVC(kernel = 'rbf', random_state = 0)
    classifier_svm.fit(X_train, y_train)

    y_pred = classifier_svm.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_svm.score(X_train, y_train)}\nTest score : {classifier_svm.score(X_test, y_test)}")
#     print("-"*100)
#     print(cm)


# In[48]:



def knn_classifier(X_train, X_test, y_train, y_test):
    
    classifier_knn = KNeighborsClassifier(metric = 'minkowski', p = 2)
    classifier_knn.fit(X_train, y_train)

    y_pred = classifier_knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_knn.score(X_train, y_train)}\nTest score : {classifier_knn.score(X_test, y_test)}")
#     print("-"*100)
#     print(cm)


# In[49]:



def naive_classifier(X_train, X_test, y_train, y_test):
    
    classifier_naive = GaussianNB()
    classifier_naive.fit(X_train, y_train)

    y_pred = classifier_naive.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_naive.score(X_train, y_train)}\nTest score : {classifier_naive.score(X_test, y_test)}")
#     print("-"*100)
#     print(cm)


# In[50]:



def tree_classifier(X_train, X_test, y_train, y_test):
    
    classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_tree.fit(X_train, y_train)

    y_pred = classifier_tree.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_tree.score(X_train, y_train)}\nTest score : {classifier_tree.score(X_test, y_test)}")
#     print("-"*100)
#     print(cm)


# In[51]:



def forest_classifier(X_train, X_test, y_train, y_test):
    classifier_forest = RandomForestClassifier(criterion = 'entropy', random_state = 0)
    classifier_forest.fit(X_train, y_train)

    y_pred = classifier_forest.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_forest.score(X_train, y_train)}\nTest score : {classifier_forest.score(X_test, y_test)}")
#     print("-"*100)
#     print(cm)


# In[52]:



def print_score(X_train, X_test, y_train, y_test):
    print("SVM:\n")
    svm_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("KNN:\n")
    knn_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Naive:\n")
    naive_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Decision Tree:\n")
    tree_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Random Forest:\n")
    forest_classifier(X_train, X_test, y_train, y_test)


# In[53]:


print_score(X_train, X_test, y_train, y_test)


# ---

# ## Performance Metrics

# In[54]:


classifier_forest = RandomForestClassifier(criterion = 'entropy')
classifier_forest.fit(X_train, y_train)
y_pred = classifier_forest.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm


# In[55]:


# classifier_svm = SVC(kernel = 'rbf', random_state = 0, probability=True)
# classifier_svm.fit(X_train, y_train)
# y_pred = classifier_svm.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# cm


# In[56]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# **Checking data is balanced or not**

# In[58]:


data['diagnosis'].value_counts()


# **Classification Report (Accuracy, Precision, Recall, F1-score)**

# In[59]:


from sklearn.metrics import roc_auc_score, roc_curve, classification_report


# In[60]:


print(classification_report(y_test, y_pred))


# **Getting probability instead of A/B test**

# In[61]:


y_pred_prob = classifier_forest.predict_proba(X_test)[:,1]
y_pred_prob


# **Evaluating FPR, TPR, Threshold**

# In[62]:


fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
print("FPR:\n\n", fpr)


print("-"*100)

print("TPR:\n\n", tpr)


# **Plotting ROC Curve**

# In[63]:


plt.plot([0, 1], [0, 1], "k--", label = '50% AUC')
plt.plot(fpr, tpr, label = "Random Forest")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve - Random Forest")
plt.show()


# **ROC Score**

# In[64]:


roc_auc_score(y_test,y_pred_prob)


# ---

# ## Hyperparameter Tunning

# In[65]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_forest, X = X_train, y = y_train, cv = 10)
print(accuracies.mean(), accuracies.std())


# In[66]:


from sklearn.model_selection import GridSearchCV


# In[67]:


parameters = {
    'n_estimators': [25, 50, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [14, 20, 25, 30]
}


# In[68]:


grid_search = GridSearchCV(estimator = classifier_forest,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10, 
                          n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
print('best_accuracy = ',grid_search.best_score_)
print('best_parameters = ', grid_search.best_params_)


# In[69]:


classifier_forest = RandomForestClassifier(criterion = 'gini', max_depth = 25, n_estimators = 200, random_state = 0)
classifier_forest.fit(X_train, y_train)
y_pred = classifier_forest.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm


# In[70]:


print(classification_report(y_test, y_pred))


# ---

# ## Saving model using pickle

# In[71]:


filename = 'breastcancer.pkl'
pickle.dump(classifier_forest, open(filename, 'wb'))


# In[76]:


#model = open('breastcancer.pkl','rb')
#forest = pickle.load(model)


# In[73]:


#y_pred = forest.predict(X_test)


# In[75]:


confusion_matrix(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




