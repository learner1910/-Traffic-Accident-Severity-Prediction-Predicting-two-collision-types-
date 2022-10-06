#!/usr/bin/env python
# coding: utf-8

# <h2>1. Import libraries<h2>

# In[2]:


get_ipython().system('pip install pydotplus')


# In[4]:


get_ipython().system('pip install folium')


# In[1]:


import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import json # library to handle JSON files
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from scipy import stats
import scipy as sp
import random


# Matplotlib and associated plotting modules
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
from io import StringIO
import itertools
from sklearn.model_selection import train_test_split

# import k-means from clustering stage
from sklearn.cluster import KMeans

# import Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# import SVM
from sklearn import svm

import folium # map rendering library

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import webbrowser
from folium import plugins
get_ipython().run_line_magic('matplotlib', 'inline')




print("Libraries imported.")


# <h2>2. Load data from CSV file<h2>

# In[3]:


df = pd.read_csv("/Users/toshiniagrawal/Desktop/Data-Collisions.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.dtypes


# #### Let's see how many severity codes there are
# 
# 1 - Property damage only collision
# 
# 2 - Injury collision

# In[7]:


df['SEVERITYCODE'].value_counts(normalize = True)


# In[8]:


df['SEVERITYCODE'].value_counts(normalize = True).plot(kind = 'bar')
plt.show()


# We now have a baseline for the rate of accidents which include injury. For purposes of discussion, we will say that 70% of the accidents result in only property damage, while roughly 30% result in injury.
# 
# We can use this baseline to compare normalized rates of various conditions.
# 

# <h2>3. Data Cleaning<h2>
#     
# See if there is any NaN

# In[9]:


missing_data = df.isnull()
missing_data.head(5)


# In[10]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# ### Junction Types

# In[11]:


df['JUNCTIONTYPE'].value_counts()


# In[12]:


# Replace NaN with "Unknown"
df['JUNCTIONTYPE'].replace(np.nan, 'Unknown', inplace=True)


# In[13]:


df['JUNCTIONTYPE'].value_counts()


# In[14]:


df_junctiontype = df.groupby(['JUNCTIONTYPE'])['SEVERITYCODE'].value_counts()
df_junctiontype


# In[15]:


df_junctiontype.plot(kind = 'bar')
plt.show


# In[16]:


df_junctiontype_norm = df.groupby(['JUNCTIONTYPE'])['SEVERITYCODE'].value_counts(normalize = True)
df_junctiontype_norm


# In[17]:


df_junctiontype_norm.plot(kind = 'bar')
plt.show


# In[18]:


fig_junctiontype = plt.figure() # create figure

ax_junc = fig_junctiontype.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_junc_norm = fig_junctiontype.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**


df_junctiontype.plot(kind='bar', figsize = (10,6), ax=ax_junc) # add to subplot 1
ax_junc.set_title('Junction Types and Severity')
ax_junc.set_xlabel('Junction Type')



df_junctiontype_norm.plot(kind='bar', figsize = (10,6), ax=ax_junc_norm) # add to subplot 1
ax_junc_norm.set_title('Junction Types and Severity (Normalized)')
ax_junc_norm.set_xlabel('Junction Type')

plt.show()


# In[19]:


f, ax = plt.subplots(figsize=(25, 10))
sns.countplot(x='JUNCTIONTYPE', hue='SEVERITYCODE',data=df,order=df['JUNCTIONTYPE'].value_counts().index)


# The majority of accidents occur at mid-block (not related to intersection). However, more injury collision occur at intersection (intersection related).  

# ### Light Conditions

# In[20]:


df['LIGHTCOND'].value_counts()


# In[21]:


# Replace NaN with "Unknown"
df['LIGHTCOND'].replace(np.nan, 'Unknown', inplace=True)
df['LIGHTCOND'].value_counts()


# In[22]:


df_light = df.groupby(['LIGHTCOND'])['SEVERITYCODE'].value_counts()
df_light


# In[23]:


df_light.plot(kind = 'bar')
plt.show


# In[24]:


df_light_norm = df.groupby(['LIGHTCOND'])['SEVERITYCODE'].value_counts(normalize = True)
df_light_norm


# In[25]:


df_light_norm.plot(kind = 'bar')
plt.show


# In[26]:


fig_light = plt.figure() # create figure

ax_light = fig_light.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_light_norm = fig_light.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**


df_light.plot(kind='bar', figsize = (10,6), ax=ax_light) # add to subplot 1
ax_light.set_title('Light Conditions and Severity')
ax_light.set_xlabel('Light Conditions')



df_light_norm.plot(kind='bar', figsize = (10,6), ax=ax_light_norm) # add to subplot 1
ax_light_norm.set_title('Light Conditions and Severity (Normalized)')
ax_light_norm.set_xlabel('Light Conditions')

plt.show()


# In[27]:


f, ax = plt.subplots(figsize=(25, 10))
sns.countplot(x='LIGHTCOND', hue='SEVERITYCODE',data=df,order=df['LIGHTCOND'].value_counts().index)


# The majority of accidents occur in Daylight, and Dark - Street Lights On.
# Lighting conditions appear to generally follow the base rate with some variance.

# ### Collision Types

# In[28]:


df['COLLISIONTYPE'].value_counts()


# In[29]:


# Replace NaN with "Unknown"
df['COLLISIONTYPE'].replace(np.nan, 'Unknown', inplace=True)
df['COLLISIONTYPE'].value_counts()


# In[30]:


df_collision = df.groupby(['COLLISIONTYPE'])['SEVERITYCODE'].value_counts()
df_collision


# In[31]:


df_collision_norm = df.groupby(['COLLISIONTYPE'])['SEVERITYCODE'].value_counts(normalize = True)
df_collision_norm


# In[32]:


fig_collision = plt.figure() # create figure

ax_coll = fig_collision.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_coll_norm = fig_collision.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**


df_collision.plot(kind='bar', figsize = (10,6), ax=ax_coll) # add to subplot 1
ax_coll.set_title('Collision Types and Severity')
ax_coll.set_xlabel('Collision Type')



df_collision_norm.plot(kind='bar', figsize = (10,6), ax=ax_coll_norm) # add to subplot 1
ax_coll_norm.set_title('Collision Types and Severity (Normalized)')
ax_coll_norm.set_xlabel('Collision Type')

plt.show()


# In[33]:


f, ax = plt.subplots(figsize=(25, 10))
sns.countplot(x='COLLISIONTYPE', hue='SEVERITYCODE',data=df,order=df['COLLISIONTYPE'].value_counts().index)


# There appear to be a few collision types which have a disproportionate result:
# 
# Angles - 39.3% injury
# 
# Head on - 43.1% injury
# 
# Parked car - 94.5% property damage only
# 
# Pedestrian - 89.9% injury
# 
# Cycles - 87.6% injury
# 
# The type of collision appears to have a significant impact on the likelihood of injury.

# ### Weather

# In[34]:


df['WEATHER'].value_counts()


# In[35]:


# Replace NaN with "Unknown"
df['WEATHER'].replace(np.nan, 'Unknown', inplace=True)
df['WEATHER'].value_counts()


# In[36]:


df_weather = df.groupby(['WEATHER'])['SEVERITYCODE'].value_counts()
df_weather


# In[37]:


df_weather_norm = df.groupby(['WEATHER'])['SEVERITYCODE'].value_counts(normalize = True)
df_weather_norm


# In[38]:


fig_weather = plt.figure() # create figure

ax_weather = fig_weather.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_weather_norm = fig_weather.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**


df_weather.plot(kind='bar', figsize = (10,6), ax=ax_weather) # add to subplot 1
ax_weather.set_title('Weather Types and Severity')
ax_weather.set_xlabel('Weather Type')



df_weather_norm.plot(kind='bar', figsize = (10,6), ax=ax_weather_norm) # add to subplot 1
ax_weather_norm.set_title('Weather Types and Severity (Normalized)')
ax_weather_norm.set_xlabel('Weather Type')

plt.show()


# In[39]:


f, ax = plt.subplots(figsize=(25, 10))
sns.countplot(x='WEATHER', hue='SEVERITYCODE',data=df,order=df['WEATHER'].value_counts().index)


# Most weather instances are Clear, Raining, Overcast.
# 
# All of these have a slightly higher percentage of injury. I initially found it surprising that adverse weather conditions did not result in greater injury. Perhaps this is due to more alert, careful driving in adverse conditions.
# 
# I was initially surprised to see that </i> Partly Cloudy had a 60% rate of personal injury, but with only 5 instances, the sample size is too small to really draw any meaningful conclusions.

# ### Speeding

# Whether or not speeding was a factor in the collision. (Y/N)

# In[40]:


df['SPEEDING'].value_counts()


# In[41]:


# Replace NaN with "N"
df['SPEEDING'].replace(np.nan, 'N', inplace=True)
df['SPEEDING'].value_counts()


# In[42]:


df_speed = df.groupby(['SPEEDING'])['SEVERITYCODE'].value_counts()
df_speed


# In[43]:


df_speed_norm = df.groupby(['SPEEDING'])['SEVERITYCODE'].value_counts(normalize = True)
df_speed_norm


# In[44]:


df_speed_norm.plot(kind = 'bar')
plt.show()


# In[45]:


sns.countplot(x='SPEEDING', hue='SEVERITYCODE',data=df,order=df['SPEEDING'].value_counts().index)


# With only 9,333 accidents reflecting speeding, we will need to update 'Nan' to 'N', so all accidents will be taken into account.
# 
# Speed appears to have a higher likelihood of involving injury when compared to the baseline - 37.8%</i>

# ### Road Conditions

# In[46]:


df['ROADCOND'].value_counts()


# In[47]:


# Replace NaN with "Unknown"
df['ROADCOND'].replace(np.nan, 'Unknown', inplace=True)
df['ROADCOND'].value_counts()


# In[48]:


df_cond = df.groupby(['ROADCOND'])['SEVERITYCODE'].value_counts()
df_cond


# In[49]:


df_cond_norm = df.groupby(['ROADCOND'])['SEVERITYCODE'].value_counts(normalize = True)
df_cond_norm


# In[50]:


fig_conditions = plt.figure() # create figure

ax_cond = fig_conditions.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_cond_norm = fig_conditions.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

df_cond.plot(kind='bar', figsize = (10,6), ax=ax_cond) # add to subplot 1
ax_cond.set_title('Road Condition Types and Severity')
ax_cond.set_xlabel('Road Condition Type')


df_cond_norm.plot(kind='bar', figsize = (10,6), ax=ax_cond_norm) # add to subplot 1
ax_cond_norm.set_title('Road Condition Types and Severity (Normalized)')
ax_cond_norm.set_xlabel('Road Condition Type')

plt.show()


# In[51]:


f, ax = plt.subplots(figsize=(25, 10))
sns.countplot(x='ROADCOND', hue='SEVERITYCODE',data=df,order=df['ROADCOND'].value_counts().index)


# Road conditions appear to have similar results to weather. Worse road conditions may result in different driving habits which may have higher probability of property damage, but with more careful driving, may minimize the injury risks.

# ### Inattention
# 
# Whether or not collision was due to inattention. (Y/N)

# In[52]:


df['INATTENTIONIND'].value_counts()


# In[53]:


# Replace NaN with "N"
df['INATTENTIONIND'].replace(np.nan, 'N', inplace=True)
df['INATTENTIONIND'].value_counts()


# In[54]:


df_attention = df.groupby(['INATTENTIONIND'])['SEVERITYCODE'].value_counts()
df_attention


# In[55]:


df_attention_norm = df.groupby(['INATTENTIONIND'])['SEVERITYCODE'].value_counts(normalize = True)
df_attention_norm


# In[56]:


fig_attention = plt.figure() # create figure

ax_attention = fig_attention.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_attention_norm = fig_attention.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

df_attention.plot(kind='bar', figsize = (10,6), ax=ax_attention) # add to subplot 1
ax_attention.set_title('Attention Types and Severity')
ax_attention.set_xlabel('Attention Type')


df_attention_norm.plot(kind='bar', figsize = (10,6), ax=ax_attention_norm) # add to subplot 1
ax_attention_norm.set_title('Attention Types and Severity (Normalized)')
ax_attention_norm.set_xlabel('Attention Type')

plt.show()


# In[57]:


sns.countplot(x='INATTENTIONIND', hue='SEVERITYCODE',data=df,order=df['INATTENTIONIND'].value_counts().index)


# With only 29,805 accidents reflecting 'inattention,' we will need to update 'Nan' to 'N', so all accidents will be taken into account.
# 
# Inattention has a higher percentage of injury than the baseline.

# ### Hit parked car
#  
# Whether or not the collision involved hitting a parked car. (Y/N)

# In[58]:


df['HITPARKEDCAR'].value_counts()


# In[59]:


df_parked = df.groupby(['HITPARKEDCAR'])['SEVERITYCODE'].value_counts()
df_parked


# In[60]:


df_parked_norm = df.groupby(['HITPARKEDCAR'])['SEVERITYCODE'].value_counts(normalize = True)
df_parked_norm


# In[61]:


fig_parked = plt.figure() # create figure

ax_parked = fig_parked.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax_parked_norm = fig_parked.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

df_parked.plot(kind='bar', figsize = (10,6), ax=ax_parked) # add to subplot 1
ax_parked.set_title('parked Types and Severity')
ax_parked.set_xlabel('parked Type')


df_parked_norm.plot(kind='bar', figsize = (10,6), ax=ax_parked_norm) # add to subplot 1
ax_parked_norm.set_title('parked Types and Severity (Normalized)')
ax_parked_norm.set_xlabel('parked Type')

plt.show()


# In[62]:


sns.countplot(x='HITPARKEDCAR', hue='SEVERITYCODE',data=df,order=df['HITPARKEDCAR'].value_counts().index)


# Hitting a parked car results in only property damage 93.8% of the time. Not hitting a parked car follows the baseline.
# 

# <h2>5. Final Feature Selection for Modeling<h2>

# In[63]:


df1 = df[['SEVERITYCODE','COLLISIONTYPE','JUNCTIONTYPE','SPEEDING','INATTENTIONIND','WEATHER','ROADCOND','LIGHTCOND','HITPARKEDCAR']]


# In[64]:


df1.head()


# In[65]:


y = df1['SEVERITYCODE'].values
y[0:5]


# In[66]:



df1['ROADCOND'].replace(to_replace=['Other','Unknown','Dry','Wet','Ice','Snow/Slush','Standing Water','Sand/Mud/Dirt','Oil'],value=[0,0,1,2,3,3,3,2,2],inplace=True)
df1['WEATHER'].replace(to_replace=['Other','Unknown','Clear','Raining','Overcast','Snowing','Fog/Smog/Smoke','Sleet/Hail/Freezing Rain','Blowing Sand/Dirt','Severe Crosswind','Partly Cloudy'],value=[0,0,1,3,2,3,3,3,3,3,2],inplace=True)
df1['LIGHTCOND'].replace(to_replace=['Other','Unknown','Daylight','Dark - Street Lights On','Dusk','Dawn','Dark - No Street Lights','Dark - Street Lights Off','Dark - Unknown Lighting'],value=[0,0,1,3,2,2,4,4,4],inplace=True)
from sklearn.preprocessing import LabelEncoder
#creating an instance of labelencoder
labelencoder = LabelEncoder()
#assigning numerical values
df1[['COLLISIONTYPE']] = labelencoder.fit_transform(df[['COLLISIONTYPE']])
df1[['JUNCTIONTYPE']] = labelencoder.fit_transform(df[['JUNCTIONTYPE']])
df1[['SPEEDING']] = labelencoder.fit_transform(df[['SPEEDING']])
df1[['INATTENTIONIND']] = labelencoder.fit_transform(df[['INATTENTIONIND']])
df1[['HITPARKEDCAR']] = labelencoder.fit_transform(df[['HITPARKEDCAR']])
#Display DataFrame
df1.head()


# ### Normalize the dataset

# In[67]:


X = df1[['COLLISIONTYPE','JUNCTIONTYPE','SPEEDING','INATTENTIONIND','WEATHER','ROADCOND','LIGHTCOND','HITPARKEDCAR']]
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## Methodology

# ## Analysis

# <h3>Train/Test Split<h3>

# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <h3>Decision Tree<h3>

# In[69]:


Tree = DecisionTreeClassifier(criterion="entropy", max_depth =5)
Tree.fit(X_train,y_train)
Tree


# In[70]:


#Train Model and Predict
DTyhat=Tree.predict(X_test)
print(DTyhat[0:5])
print(y_test[0:5])


# In[71]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, DTyhat))


# In[72]:


from sklearn.metrics import classification_report
print (classification_report(y_test, DTyhat))


# In[73]:


from six import StringIO
#from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


dot_data = StringIO()
filename = "accidenttree.png"
featureNames = df1.columns[1:9]
targetNames = df1["SEVERITYCODE"].unique().tolist()
out=tree.export_graphviz(Tree,feature_names=featureNames,out_file=dot_data,class_names=True,filled=True,special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
plt.figure(figsize=(40, 40))
img = mpimg.imread(filename)
plt.imshow(img,interpolation='nearest')


# <h3>KNN<h3>

# After balancing SEVERITYCODE feature, and standardizing the input feature, the data has been ready for building machine learning models. I have employed three machine learning models: K Nearest Neighbour (KNN) Decision Tree Linear Regression After importing necessary packages and splitting preprocessed data into test and train sets, for each machine learning model, I have built and evaluated the model and shown the results as follow:

# In[75]:


from sklearn.neighbors import KNeighborsClassifier


# In[100]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[102]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[111]:


k = 18
knn=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
knn_y_pred=knn.predict(X_test)
knn_y_pred[0:5]


# In[112]:


print("KNN's Accuracy: ", metrics.accuracy_score(y_test, knn_y_pred))


# In[113]:


from sklearn.metrics import classification_report,confusion_matrix


# In[114]:


print(confusion_matrix(y_test,knn_y_pred))
print(classification_report(y_test,knn_y_pred))


# <h3>Linear Regression<h3>
# 

# In[119]:


#Building the LR Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR=LogisticRegression(C=6,solver="liblinear").fit(X_train,y_train)
LR


# In[102]:


#Train Model and predict
LRyhat=LR.predict(X_test)
LRyhat


# In[112]:


print("LinearRegression's Accuracy: ", metrics.accuracy_score(y_test, LRyhat))


# In[113]:


yhat_prob=LR.predict_proba(X_test)
yhat_prob


# In[114]:


print(confusion_matrix(y_test,LRyhat))
print(classification_report(y_test,LRyhat))


# <h3>Random Forest<h3>

# In[106]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


# In[107]:


rfc


# In[108]:


y_pred


# In[109]:


print("RandomForest's Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[110]:


from sklearn.metrics import classification_report


# In[111]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[117]:


import pandas as pd
data = [['KNN',0.7442,0.84],['Decision Tree',0.7385,0.83],['Linear Regresion',0.6999,0.82],['Random Forest',0.7506,0.84]]
df = pd.DataFrame(data,columns=['Mode','Accuracy','F1 Score'])
print(df)


# Based on the above table, Random Forest is the best model to predict car accident severity. Despite that, KNN is better as Random forest is comaparatively slower and less intrepretable than KNN . 

# In[ ]:




