#!/usr/bin/env python
# coding: utf-8

# # Importing basic libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score


# In[2]:


# Read the data

df = pd.read_csv('https://raw.githubusercontent.com/subhashdixit/Support_Vector_Machines/main/SVC/Red_Wine_Quality/winequality-red.csv')


# In[3]:


df.head()


# In[4]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[5]:


# Shape of the data

df.shape


# In[6]:


# Uniques values in target

df.quality.unique()


# In[7]:


# Number of values in each category

df['quality'].value_counts()

Observation:- Highly Imbalanced data.
# In[8]:


df.columns


# In[9]:


# Information about the data

df.info()


# In[10]:


# Statistical Analysis

df.describe().T


# In[11]:


# Check Null values in the dataset

df.isnull().sum()


# In[12]:


# Checking for Outliers

plt.figure(figsize=(20,8))
sns.boxplot(data=df)
plt.title("Boxplot of all the features")


# In[13]:


# Handling outliers 

# Creating a function which will return upper and lower limit for outs
def find_boundaries(df, var, distance):
    iqr = df[var].quantile(0.75) - df[var].quantile(0.25)
    low_bound = df[var].quantile(0.25) - (iqr*distance)
    upp_bound = df[var].quantile(0.75) + (iqr*distance)
    return upp_bound, low_bound


# In[14]:


df.columns


# In[15]:


outliers_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

# Removing Outliers of the dataset
for i in outliers_columns:
    upper_boundary, lower_boundary = find_boundaries(df,i,1.5)
    outliers = np.where(df[i]>upper_boundary, True, np.where(df[i]<lower_boundary,True, False))
    outliers_df = df.loc[outliers,i]
    df_trimed = df.loc[~outliers,i]
    df[i] = df_trimed


# In[16]:


# Boxplot after removing the outlier

plt.figure(figsize=(20,8))
sns.boxplot(data=df)
plt.title("Boxplot of all the features")


# In[17]:


for x in outliers_columns:
    df[x] = df[x].astype('float32')


# # Graphical Analysis

# In[18]:


df.columns


# In[19]:


df['quality'] = df['quality'].astype('int')


# In[20]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(data=df.corr(),annot=True)


# # Segregate independent and dependent variable

# In[21]:


X = df.drop('quality',axis=1)
y = df['quality']


# In[22]:


# Train Test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[23]:


print(X_train.shape)
print(X_test.shape)


# In[24]:


print(y_train.shape)
print(y_test.shape)


# ## Scaling the data

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


# In[26]:


print(scaler.mean_)


# In[27]:


X_train_tf = scaler.transform(X_train)
X_train_tf


# In[28]:


X_train_tf_df = pd.DataFrame(X_train_tf, columns = outliers_columns)
X_train_tf_df


# In[29]:


X_train_tf_df.fillna(X_train_tf_df.mean(),inplace=True)


# ## Support Vector Classifier(SVC)

# In[30]:


from sklearn.svm import SVC
svc = SVC()


# In[31]:


svc.fit(X_train_tf_df,y_train)


# In[32]:


# Train Accuracy
svc.score(X_train_tf_df, y_train)


# In[33]:


## Test data

X_test_tf = scaler.transform(X_test)
X_test_tf


# In[34]:


X_test_tf_df = pd.DataFrame(X_test_tf,columns=outliers_columns)


# In[35]:


X_test_tf_df.isnull().sum()


# In[36]:


X_test_tf_df.fillna(X_test_tf_df.mean(),inplace=True)


# In[37]:


y_pred = svc.predict(X_test_tf_df)


# In[38]:


y_test


# In[39]:


## Test Accuracy

from sklearn.metrics import classification_report, confusion_matrix


# In[40]:


print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print('\n')
print('Classification Report')
print(classification_report(y_test,y_pred))


# # Lets hypertune the model

# In[41]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel':['linear','rbf']}


# In[42]:


from sklearn.svm import SVC
svc = SVC()


# In[43]:


grid_model = GridSearchCV(svc, param_grid,verbose=3)
grid_model.fit(X_train_tf_df, y_train)


# In[44]:


print(grid_model.best_params_)


# In[45]:


grid_model.score(X_train_tf_df, y_train)


# In[46]:


# Test Data

from sklearn.metrics import classification_report, confusion_matrix


# In[47]:


grid_model_pred = grid_model.predict(X_test_tf_df)
print("Confusion Matrix")
print(confusion_matrix(y_test, grid_model_pred))
print('\n')
print("Classification report")
print(classification_report(y_test, grid_model_pred))


# ## Sampling to handle imbalanced dataset

# In[48]:


X.shape


# In[49]:


X.isnull().sum()


# In[50]:


X.fillna(X.mean(),inplace=True)

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks!pip install imblearn
# In[51]:


import imblearn
from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy = 'minority',random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(X,y)
oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)


# In[52]:


pip install -U imbalanced-learn


# In[53]:


oversampled_X.shape


# In[54]:


oversampled_Y.shape


# In[55]:


## Since our dfset shape has changed we need to perform train_test_split again


# In[56]:


X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(oversampled_X,
oversampled_Y, test_size=0.25, random_state=44)


# In[59]:


# Applying standardization to our model
scaler.fit(X_train_sm)
X_train_sm_tf = scaler.transform(X_train_sm)
X_test_sm_tf = scaler.transform(X_test_sm)


# In[60]:


grid_model.fit(X_train_sm_tf, y_train_sm)


# In[61]:


## Train Accuracy
grid_model.score(X_train_sm_tf, y_train_sm)


# In[62]:


# Test Data 
# Prediction on test data

y_pred_grid_sm = grid_model.predict(X_test_sm_tf)


# In[63]:


# Test Accuracy
from sklearn.metrics import classification_report, confusion_matrix


# In[64]:


print("Confusion Matrix")
print(confusion_matrix(y_test_sm,y_pred_grid_sm))
print("Classification Report")
print(classification_report(y_test_sm,y_pred_grid_sm))


# ## Logistic Regressionn Model

# In[65]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_tf, y_train)


# In[66]:


## Train Accuracy
lr.score(X_train_tf,y_train)


# In[67]:


# Test Data
y_pred_lr = lr.predict(X_test_tf)


# In[68]:


## Test Accuracy
from sklearn.metrics import classification_report, confusion_matrix


# In[72]:


print("COnfusion Matrix")
print(confusion_matrix(y_test,y_pred_lr))
print('\n')
print("Classification report")
print(classification_report(y_test,y_pred_lr))


# ## Conclusion

# In[82]:


from sklearn.metrics import accuracy_score
print("Final Test Accuracy Score of all the Models")
print('\n')
print('SVC : ',round(accuracy_score(y_test,y_pred),2)*100)
print('\n')
print('SVC after hyperparmeter tunning : ',round(accuracy_score(y_test,grid_model_pred)))
print('\n')
print('SVC after sampling using hypertunned model : ',round(accuracy_score(y_test_sm,y_pred_grid_sm),2)*100)
print('\n')
print('Logistic Regression : ',round(accuracy_score(y_test, y_pred_lr),2)*100)


# In[ ]:




