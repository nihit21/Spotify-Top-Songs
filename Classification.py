#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Necessary Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Load the Dataset
train_data = pd.read_csv(r"C:\Users\aravi\Downloads\CS98XClassificationTrain.csv")
train_data.head()


# In[4]:


#Display number of elements in each column of the dataset
train_data.count()


# In[5]:


#Display number of null elements in each column of the dataset
train_data.isnull().sum()


# In[6]:


#Replcace the Null values with the most frequent genre
train_data['top genre'].replace('',np.NaN)
train_data['top genre'].fillna('adult standards')

#train_data = train_data.dropna(axis=0)


# In[7]:


#Encode 'top genre' using label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['top genre'] = le.fit_transform(train_data['top genre'])
train_data.head(10)


# In[8]:


#Frequency distribution of elements in the dataset
import matplotlib.pyplot as plt
train_data.hist(bins=25, figsize=(20,20))
plt.show()


# In[9]:


#Create a Box Plot of features in the dataset
import seaborn as sns 
import matplotlib.style as style
style.use('seaborn-poster')
sns.boxplot(data=train_data.drop(['Id','title', 'year','artist', 'top genre'], axis=1))
plt.xlabel('Features')
plt.ylabel('Value')
plt.show()


# In[10]:


#Create a linear Corelation Matrix of the features in the dataset
plt.figure(figsize=(15,10))
sns.heatmap(train_data.corr(), annot=True,cmap='RdYlGn')
plt.title('Linear Correlation Matrix')
plt.show()


# In[11]:


from scipy import stats
import numpy as np
def get_outlier_counts(df, threshold):
    df = df.copy()
    
    # Get the z-score for specified threshold
    threshold_z_score = stats.norm.ppf(threshold)
    
    # Get the z-scores for each value in df
    z_score_df = pd.DataFrame(np.abs(stats.zscore(df)), columns=df.columns)
    
    # Compare df z_scores to the threshold and return the count of outliers in each column
    return (z_score_df > threshold_z_score).sum(axis=0)


# In[12]:


get_outlier_counts(train_data.drop(columns=['title','artist','Id']), 0.9999999)


# In[13]:


#Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Prepare the Data
X = train_data.drop(columns=['title','artist','top genre','Id'])
Y = train_data['top genre']


# In[14]:


#Scale the Data using Standard Scalar Function
scale = StandardScaler()
scaled_X=scale.fit_transform(X)


# In[15]:


#Define the function to train the models
def train_model(model, x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=50, test_size=0.1)
  model.fit(x_train,y_train)
  #score = model.score(x_test,y_test)
  #print(score)
  prediction = model.predict(x_test)
  print("Accuracy Score :", accuracy_score(y_test,prediction)*100)


# In[16]:


#Logistic Regresion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


one_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=10))
train_model(one_model,scaled_X,Y)


# In[17]:


#Bagging Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagg_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                            n_estimators=100, # number of base estimators in ensemble
                            max_samples=0.5,  # 50% of samples taken from existing instances
                            max_features=1.0, # 100% of features are taken
                            bootstrap_features=False, # it represent samples taken with feature replacement
                            oob_score=True,
                            random_state=10)

#model = BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=100, bootstrap=False, random_state=0)
train_model(bagg_model,scaled_X,Y)


# In[18]:


#Hard Voting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimatorsList = [
                  ('rf', RandomForestClassifier(n_estimators=30, random_state=10)),
                  ('svc',SVC()),
                  ('bg',BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                            n_estimators=100, # number of base estimators in ensemble
                            max_samples=0.5,  # 50% of samples taken from existing instances
                            max_features=1.0, # 100% of features are taken
                            bootstrap_features=False, # it represent samples taken with feature replacement
                            random_state=10))
                  ]

voting_model = VotingClassifier(estimators=estimatorsList,voting='hard')
train_model(voting_model,scaled_X, Y)


# In[19]:


#AdaBoost Classifer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=500, algorithm="SAMME.R", learning_rate=0.1)
train_model(ada_model,scaled_X, Y)


# In[23]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=500, max_depth=5, max_leaf_nodes=16, n_jobs=-1)

train_model(rf_model,scaled_X,Y)

#Feature Importance
feature_importances = rf_model.feature_importances_
print("Feature importances:\n{}".format(feature_importances))


# In[29]:


# Load the test file
test = pd.read_csv(r"C:\Users\aravi\Downloads\CS98XClassificationTest.csv")
test_copy = test.copy() # Create copy file to retain the original dataset
test_copy.drop(['title','artist','Id'], axis=1, inplace=True)
test_scaled = scale.fit_transform(test_copy) # fit the test_copy
prediction=le.inverse_transform(rf_model.predict(test_scaled))
test['top genre'] = prediction
#Note: CS98XClassificationTest.csv has to be added for each test run
test.to_csv(r"C:\Users\aravi\Downloads\CS98XClassificationTest.csv") # load the final predicted pop into original test dataset
test.head(5)


# In[ ]:




