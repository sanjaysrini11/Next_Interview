#!/usr/bin/env python
# coding: utf-8

# In[1]:
st.text("Part 1")
import streamlit as st
import json
import re

c = {"orders":[{"id":1},{"id":2},{"id":3},{"id":4},{"id":5},{"id":6},{"id":7},{"id":8},{"id":9},{"id":10},{"id":11},{"id":648},{"id":649},{"id":650},{"id":651},{"id":652},{"id":653}],"errors":[{"code":3,"message":"[PHP Warning #2] count(): Parameter must be an array or an object that implements Countable (153)"}]}
y = json.dumps(c)


r = re.findall('{"id":(.+?)}',y)
r.extend(re.findall('"code":(.\d+?)',y))
arr = [int(i) for i in r]
st.text("regex to extract all the numbers with orange color background from the below text in italics:\n")
st.text(arr)
 
st.text("Part 2")
st.text("Here is the data set that contains the history of customer booking in a hotel.")





from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV
import streamlit as st
from sklearn import metrics
from sklearn.metrics import mean_squared_error,roc_auc_score,roc_curve,classification_report,confusion_matrix,plot_confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from mpl_toolkits import mplot3d
pd.options.display.max_columns = None

# Suppress warnings
import warnings; warnings.filterwarnings('ignore')

# Visualize Tree
from sklearn.tree import export_graphviz
# from IPython.display import Image
from os import system

# Display settings
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000


# In[2]:



df_train = pd.read_csv("train_data_evaluation_part_2.csv")
st.write(df_train.head())


# In[7]:


st.write(df_train.shape)


# In[8]:


st.write(df_train.info())


# In[9]:


st.write(df_train.isnull().sum())


# In[10]:


st.write(df_train.describe())


# In[11]:


for column in df_train.columns:
    if df_train[column].dtype == 'object':
        st.write((column.upper(),': ',df_train[column].nunique()))
        st.write((df_train[column].value_counts().sort_values()))
        st.write(('\n'))


# In[12]:


df_train['BookingsCanceled'].value_counts()


# In[13]:


df_train['BookingsNoShowed'].value_counts()


# In[14]:


df_train['BookingsCheckedIn'].value_counts()


# In[15]:


for column in df_train.columns:
    if df_train[column].dtype != 'object':
        mean = df_train[column].mean()
        df_train[column] = df_train[column].fillna(mean)    
        
df_train.isnull().sum() 


# In[16]:


splot_cols=df_train.columns
for i in splot_cols:
    if df_train[i].dtype != 'object' and df_train[i].isnull().values.any() != True:
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(df_train[i])
        st.pyplot(fig)


# In[17]:


fig = plt.figure(figsize=(10, 4))
sns.scatterplot(x="BookingsCheckedIn",y="BookingsCanceled",data=df_train)
st.pyplot(fig)


# In[18]:

fig = plt.figure(figsize=(10, 4))
sns.scatterplot(x="BookingsCheckedIn",y="BookingsNoShowed",data=df_train)
st.pyplot(fig)


# In[19]:

# fig = plt.figure()
# df_train.hist()
# st.pyplot(fig)


# In[20]:


# fig = plt.figure()
# sns.pairplot(df_train,diag_kind='kde',palette="tab10")
# st.pyplot(fig)


# In[21]:


fig = plt.figure(figsize=(10, 8))
sns.countplot(x = 'BookingsCheckedIn',data=df_train)
st.pyplot(fig)


# In[22]:


fig = plt.figure(figsize=(10, 8))
sns.countplot(x='DistributionChannel',data=df_train)
st.pyplot(fig)


# In[23]:

fig = plt.figure(figsize=(10, 8))
sns.countplot(x='MarketSegment',data=df_train)
st.pyplot(fig)


# In[24]:


fig = plt.figure(figsize=(10, 8))
ax = sns.stripplot(x='MarketSegment', y='RoomNights', data=df_train)
plt.title('Graph')
st.pyplot(fig)


# ###### More rooms where booked by the customers due to direct market segment and corporate booking.

# In[25]:


fig = plt.figure(figsize=(10, 8))
ax = sns.stripplot(x='DistributionChannel', y='RoomNights', data=df_train)
plt.title('Graph')
st.pyplot(fig)


# More rooms where booked for the corporate people.

# In[26]:


st.write(df_train.columns)


# In[27]:


new_df_train = df_train.drop(['Unnamed: 0','ID','Nationality'],axis=1)
st.write(new_df_train.head())


# In[28]:

st.write("categorical data")
df_cat_train = new_df_train.select_dtypes(include = ['object'])
df_num_train = new_df_train.select_dtypes(include = ['float64','int64'])
st.write(df_cat_train.head())


# In[29]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
for i in df_cat_train.columns:
#     print(i)
    df_cat_train[i]= label_encoder.fit_transform(df_cat_train[i])
st.write("Label Encoding",df_cat_train.head())


# In[30]:


df_train_copy = pd.concat([df_cat_train,df_num_train],axis=1)
st.write(df_train_copy.head())


# In[31]:


st.write(df_train_copy['BookingsCheckedIn'].value_counts())


# In[32]:


df_train_copy['BookingsCheckedIn'] = np.where((df_train_copy['BookingsCheckedIn'] > 0), 1, 0)
st.write(df_train_copy['BookingsCheckedIn'].value_counts())


# In[33]:


st.write(df_train_copy.corr())


# In[34]:


fig = plt.figure(figsize=(20,10))
sns.heatmap(df_train_copy.corr(), annot=True,mask=np.triu(df_train_copy.corr(),+1))
st.pyplot(fig)


# In[35]:


x = df_train_copy.drop('BookingsCheckedIn', axis=1)
y = df_train_copy['BookingsCheckedIn']


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(df_train_copy, y, test_size=0.3 , random_state=42,stratify=y)
st.write(x_train.shape)


# In[37]:


from sklearn.neural_network import MLPClassifier
ANN_wgs = MLPClassifier(random_state=42)
st.write(ANN_wgs.fit(x_train, y_train))


# In[38]:


ytrain_predict = ANN_wgs.predict(x_train)
st.write(ytrain_predict)


# In[39]:


ytest_predict = ANN_wgs.predict(x_test)
st.write(ytest_predict)


# In[40]:


# st.write((classification_report(y_train, ytrain_predict),'-\nline'))
st.text('Model Report:\n ' + classification_report(y_train, ytrain_predict))


# In[41]:


# st.write((classification_report(y_test, ytest_predict),'-\nline'))
st.text('Model Report:\n ' + classification_report(y_test, ytest_predict))

# ### ANN with grid search

# In[42]:


param_grid_ANN = { 'hidden_layer_sizes': [93,95,97,99,102,105],
                  'activation': [ 'relu','adam'], 
                  'max_iter': [10000,12500], 
                  'solver': ['adam'], 
                  'tol': [0.00001,0.0001], }
ANN = MLPClassifier(random_state=42)


# In[43]:


grid_search_ANN = GridSearchCV(estimator = ANN, param_grid = param_grid_ANN, cv = 3)
st.write(grid_search_ANN.fit(x_train, y_train))


# In[44]:


st.write(grid_search_ANN.best_params_)


# In[45]:


best_grid_ANN = grid_search_ANN.best_estimator_
st.write(best_grid_ANN)


# In[46]:


ytrain_predict = best_grid_ANN.predict(x_train)
st.write(ytrain_predict)


# In[47]:


ytest_predict = best_grid_ANN.predict(x_test)
st.write(ytest_predict)


# In[48]:


# st.write(print(classification_report(y_train, ytrain_predict),'-\nline'))
st.text('Model Report:\n ' + classification_report(y_train, ytrain_predict))

# In[49]:


# st.write(print(classification_report(y_test, ytest_predict),'-\nline'))
st.text('Model Report:\n ' + classification_report(y_test, ytest_predict))

# In[ ]:





# In[50]:
st.write("test data")

df_test = pd.read_csv("test_data_evaluation_part2.csv")
st.write(df_test.head())


# In[51]:


st.write(df_test.shape)


# In[52]:


st.write(df_test.info())


# In[53]:


st.write(df_test['BookingsCanceled'].value_counts())


# In[54]:


st.write(df_test['BookingsNoShowed'].value_counts())


# In[55]:


st.write(df_test['BookingsCheckedIn'].value_counts())


# In[56]:


st.write(df_test.describe())


# In[57]:


for column in df_test.columns:
    if df_test[column].dtype != 'object':
        mean = df_test[column].mean()
        df_test[column] = df_test[column].fillna(mean)    
        
st.write(df_test.isnull().sum() )


# In[58]:


test_file_ids = df_test['ID']
new_df_test = df_test.drop(['Unnamed: 0','ID','Nationality'],axis=1)
st.write(new_df_test.head())


# In[59]:


df_cat_test = new_df_test.select_dtypes(include = ['object'])
df_num_test = new_df_test.select_dtypes(include = ['float64','int64'])
st.write(df_cat_test.head())


# In[60]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
for i in df_cat_test.columns:
#     print(i)
    df_cat_test[i]= label_encoder.fit_transform(df_cat_test[i])
st.write(df_cat_test.head())


# In[61]:


df_test_copy = pd.concat([df_cat_test,df_num_test],axis=1)
st.write(df_test_copy.head())


# In[62]:


st.write(df_test_copy['BookingsCheckedIn'].value_counts())


# In[63]:


df_test_copy['BookingsCheckedIn'] = np.where((df_test_copy['BookingsCheckedIn'] > 0), 1, 0)
st.write(df_test_copy['BookingsCheckedIn'].value_counts())


# In[64]:


final_predictions = best_grid_ANN.predict(df_test_copy)
submission=pd.DataFrame([test_file_ids,final_predictions]).T
st.write("predicted output ")
st.write(submission["Unnamed 0"].value_counts())
submission.rename(columns={"Unnamed 0": "BookingsCheckedIn"},inplace=True)
submission.to_csv('submission.csv',index = False)
files.download('submission.csv')

