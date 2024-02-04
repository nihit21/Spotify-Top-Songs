#!/usr/bin/env python
# coding: utf-8

# # Spotify Regression
# 

# #INTRODUCTION
# Spotify is a multimedia platform used worldwide which gives access to different content like songs, podcasts etc from all over the world. Most of the times we all wonder why we enjoy a particular song or how certain songs become popular over others.To find these answers we tried to came up with a solution after analyzing spotify datset. We followed few steps to analyze data like go through the data clealry, Visualized it using different python libraries, run regression models to get most relevant features  and then  select and fine tune a model.We continued with Decision Tree and Random Forest Regressor. We choose the model with the lowest RMSE score as the most representative one to use.

# # Regression

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#Import relevant packages for initial exploration
import pandas as pd
import numpy as np

import sklearn
print("Sklearn version:",sklearn.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# #3. Dataset Analysis
# The train dataset consists of 453 instances and 15 columns, 3 of which are object type (title, artist, top genre).
# 
# In out initial look we want to address few concerns.We wanted to uncover any issues with this dataset as soon as possible so that we could correct them in our data pre-processing. So, we checked if there are any null values in the dataset and we found there were 15 in 'top genre' column only which we can remove in prediction phase. When we searched for duplicates there was one which will not effect our analysis it was not removed. At last we dropped the 'id' column as it was not valuable to us.

# In[ ]:


#read the two csv files
train=pd.read_excel('/content/drive/MyDrive/CS98XRegressionTrain.xlsx')
test=pd.read_excel('/content/drive/MyDrive/CS98XRegressionTest.xlsx')


# In[ ]:


#print first 20 rows & dimensions
train.head(20)
#train.shape()


# In[ ]:


print(f'The size of the training dataset is {len(train)} records')
print(f'The size of the test dataset is {len(test)} records')


# The train dataset consists of 453 instances and 15 columns, 3 of which are object type (title, artist, top genre)

# In[ ]:


train.info()


# In[ ]:


train['top genre'].unique()


# In[ ]:


train.describe() #summary of numeric attributes
# Count is the same for all features. Difference in deviation (dB,spch vs dur)


# The test dataset consists of 114 instances and 14 columns, 3 of which are object type (title, artist, top genre)

# In[ ]:


test.info() #info for test data


# ## Check for NULL and Blank values in the datasets

# In[ ]:


#NULLs in test data
any_missing_data = test.isnull().any().any()
where_data_missing = test.isnull().any()

print("Is there any data missing?")
print(any_missing_data)
print("\nColumns where data is missing (True/False):\n")
print(where_data_missing)

how_much_missing = test.isnull().sum()
print("\nCount of missing datapoints in each category:\n")
print(how_much_missing)


# In[ ]:


#NULLs in train data
any_missing_data = train.isnull().any().any()
where_data_missing = train.isnull().any()

print("Is there any data missing?")
print(any_missing_data)
print("\nColumns where data is missing (True/False):\n")
print(where_data_missing)

how_much_missing = train.isnull().sum()
print("\nCount of missing datapoints in each category:\n")
print(how_much_missing)


# Top Genre column contains 15 null values.

# In[ ]:


#Let's see the rows where top genre is NULL
train[pd.isnull(train["top genre"])]


# In[ ]:


train['artist'].value_counts() #no of times an artist appears in the train dataset


# In[ ]:


train["top genre"].value_counts()


# In[ ]:


train['title'].value_counts() #no of times a song appears in the train set


# In[ ]:


train['artist'].nunique()


# In[ ]:


train["top genre"].nunique()


# In[ ]:


train['title'].nunique()


# From the above code chunks, we can see that there are 345 artists and 451 songs in our train dataset, making up a total of 86 different genres. Columns such as 'Id', 'title', 'artist','top genre' are of little analytical (or predictive) value and could be removed. By adopting this approaach, all 15 null values from 'top genre' would be automatically be made redundant too.

# In[ ]:


dup = train.iloc[:,1:]
duplicateRowsDF = dup[dup.duplicated(keep=False)]
duplicateRowsDF


# In[ ]:


train[train['title'] == "Take Good Care Of My Baby - 1990 Remastered"] # this looks like a duplicate


# In[ ]:


# train = train.drop(42) #drop one of them
# train[train["title"]=='Take Good Care Of My Baby - 1990 Remastered']


# In[ ]:


train[train['title'] == "Please Mr. Postman"] # same song re-issued by the Carpenters in 1975? Not classed as duplicated


# In[ ]:


# Drop duplicates rows
train.drop_duplicates(inplace=True, keep='first')


# ### Visualisation

#  We want to investigate the link between the numerous predictors and the response variable, popularity, in our exploratory data analysis. After cleaning the data, we produce some visualisations for further exploration and to check the distribution of training data..

# In[ ]:


# Plot distributions of the training/test set to check that they look similar.
train.hist(bins=50, figsize=(12,12))
test.hist(bins=50, figsize=(12,12))
plt.show()


# The last two graphs in order to see the relationship of our predictors with popularity.

# In[ ]:


#Plot the popularity distribution for train
train['pop'].hist(figsize=(10, 5), label = 'Training data')

plt.title('Popularity distribution')
plt.xlabel('Popularity Score')
plt.ylabel('Counts')
plt.legend(title = 'Dataset')
plt.show()


# In[ ]:


train['year'].hist()
plt.xlabel('Year')
plt.ylabel('Counts')
plt.show()


# In[ ]:


plt.subplots(figsize = (10,6))

ax = sns.scatterplot(x="dur",y='pop',hue='acous',data=train)
plt.title('Song popularity according to:', weight='bold')
#plt.xticks(rotation=0)
#plt.yticks(rotation=0)
plt.ylabel('Popularity')
plt.xlabel('Duration (in sec)')
ax.legend(title = 'Acousticness')

sns.set(font_scale=1) # font size


# The above visual of scatter plot indicates a potential linear (positive) relationship between pop-dur (population-duration). Now we further continued our analysis by selecting column like 'dur','db', 'enrgy, 'pop','spch',acous'and observed there relation with each other.

# In[ ]:


from pandas.plotting import scatter_matrix
attributes = ["dur","dB","nrgy","dnce","spch","bpm","acous","pop"]
scatter_matrix(train[attributes], figsize=(12,12))
plt.show()


# From the plot above, we can see that 'pop' is likely to have a linear relationship with 'acous' and 'dur' attributes.

# In[ ]:


fig , ax = plt.subplots(1,2,figsize = (20,5))
sns.histplot(train.dur, ax = ax[0], color = 'orange', kde = True, stat="count", linewidth=1.0, line_kws={"linewidth":3},
            edgecolor="black").set_title('Distribution (duration)', fontsize = 14)

sns.histplot(train.acous, ax = ax[1], color = 'teal', kde = True, stat="count", linewidth=1.0, line_kws={"linewidth":3},
            edgecolor = 'black').set_title('Distribution (acoustic)', fontsize = 14)


# In[ ]:


train.plot(kind="scatter", x="dur", y="pop") # Looking at potential relationships


# In[ ]:


train.plot(kind="scatter", x="acous", y="pop")


# ## Correlation

# Checking correlation between variables, listing them in descending order and producing a correlation matrix.

# In[ ]:


# check correlation between variables
corr=train.corr()
sns.heatmap(corr)
plt.show()


# The heatmap above reinforced the notion that the two predictors with the strongest association with popularity were 'acous' (0.46) and'dur'(0.36) . We thus had reason to believe that they would be influential and most important to include in our model building.

# In[ ]:


corr['pop'].sort_values(ascending=False)


# #linear Corelation Matrix

# In[ ]:


plt.figure(figsize=(10,10))

corr_matrix = sns.heatmap(corr, cmap="flare", annot = True, vmin=-1, vmax=1, center=0)
corr_matrix.set_title('Correlation Matrix', fontdict={'fontsize':15},  pad=12)


# Observations :
# 1.live is mostly correlated with most f the features
# 2.bpm and dnce are negatively corelated increase in one leads to decrease in one or vice versa

# ## Checking for Outliers

# An isolated data point that differs significantly from other points in the dataset is known as an outlier.
# With the help of boxplot will identify the points outside its whiskers those points will be outliers.

# In[ ]:


plt.figure(figsize=(18,10))
sns.set(style="whitegrid")
sns.boxplot(
    data = train.drop(['title','artist','top genre', 'year'], axis = 1), orient="h", palette="deep").set_title('Boxplot of Features', fontdict={'fontsize':15}, pad=12)

plt.xlabel('Value')
plt.ylabel('Features')
plt.show()


# In[ ]:


train.plot(kind='box',subplots=True, layout=(2,6), figsize=(16,6))
plt.show()


# 
# #Not sure if the outliers below should be removed?
# Outliers can be removed manually and further to this, DBSCAN clustering can be used on top of manual removal, to identify and detect potential anomalies (also the 'eps' parameter will need adjusted accordingly, looking at the scatterplots in previous section).

# In[ ]:


corr_matrix=train.corr()
corr_matrix['pop'].sort_values(ascending=False)


# In[ ]:


#adding dur^2 
train['dur2']=train['dur']**2
train.head(5)


# 4.Data Preparation

# 4.1 Stratified Sampling
# We added new feature to data like song age which how old the song is and then we decided to drop columns such 'Id','artist','title','top genre', because they have are not of analytical importance ('Id) or are of categorical nature ('artist','title','top genre'). OneHot Encoding would have created so many extra columns, making it difficult to manage. Also, as we noticed during the data exploration, 'top genre' had 15 missing values, so by dropping the column we are solving this issue. At last we fill the null values with 'adult standards'.
# In our original attempt, we included all the remaining columns, but we noticed that attributes with very low correlation to 'pop' such as val, live, bpm didn't add anything to the model, therefore decided to go with a simple model and add features if needed.

# In[ ]:


# Adding new feature to the data : How old the song is
train['Song Age']= 2023-train['year']


# In[ ]:


# Extracting the independent variables
features = list(train.drop(['Id', 'title','artist','year', 'pop', 'dur2'], axis = 1))
features=train[features]
features.head()


# In[ ]:


# Fill missing values in top_genre with the modal value
features['top genre']= features['top genre'].fillna('adult standards')


# In[ ]:


#dependent variables
y = train['pop']


# In[ ]:


#label encode the top genre feature
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
top_genre=pd.concat([train['top genre'], test['top genre']]).fillna('adult standards').to_frame()
le.fit(top_genre)
features['top genre']=le.transform(features['top genre'])


# Stratified sampling is necessary when some of the features are skewed. We want to make sure that our data are representative of the whole population. 'Acous' seems to be skewed and also has the highest correlation (-0.466537).
# Therefore, we decided to introduce a new column in the data frame and group the data into categories. We use this column to inform the new train/test split (similar to Week 1 California house prices project).

# In[ ]:


acousSet_Bins = round(train['acous'] / 10)*10 # ROUND TO THE NEAREST 10
acousSet_Bins.value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.3, random_state=42)
print(len(X_train), len(X_test))


# In[ ]:


from sklearn.preprocessing import StandardScaler
autoscaler = StandardScaler()


X_train = autoscaler.fit_transform(X_train)
X_test = autoscaler.transform(X_test)


#  5. Select a model and train it
# The process we are going to follow:
# 1 - Select a model i.e Logistic Regression - create an instance of it 2 - Train the model - calling the fit function 3 - Run the predictions and check performance (evaluation)

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## AdaBoost
from sklearn.ensemble import AdaBoostRegressor
## Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor


# # 5.1 Linear Regression
#  A variable's value can be predicted using linear regression analysis based on the value of another variable. The dependent variable is the one you want to be able to forecast. The independent variable is the one you're using to make a prediction about the value of the other variable.

# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred= lin_reg.predict(X_test)
# Root mean squared error
mean_squared_error(y_pred, y_test, squared=False)


# # Decision Tree Regression
# It is a decision making tool which makes it  possible to represent decisions and all of its potential consequences, including outcomes, input costs, and utility. Here dataset is broken into smaller subsets and by the side decision tree is developed.
# We calculate mean square error and try to minnimize it with every node.

# In[ ]:


attribs = list(train.drop(['title','artist','top genre', 'year', 'pop','bpm', 'val', 'live'], axis = 1))
attribs

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

depth_range = range(1, 20) #tree depth
popularity=train['pop']
features=train[attribs]

RMSE_list = [] #Create a list to store the RMSE for each model


for depth in depth_range:
    treeModel = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    MSE_score = cross_val_score(treeModel, train[attribs], train['pop'], scoring='neg_mean_squared_error')
    RMSE_list.append(np.mean(np.sqrt(-MSE_score)))

# Plot RMSE vs Tree Depth
plt.figure(figsize=(10,5))
plt.plot(depth_range, RMSE_list, color = 'teal', marker = 'o')
plt.xlabel('Tree Depth')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error per Tree Depth', fontsize = 13)


# In[ ]:


treeModel_2 = DecisionTreeRegressor(max_depth = 2, random_state = 42)
treeModel_2.fit(train[attribs], train['pop'])
test['pop_test'] = treeModel_2.predict(test[attribs])
test.head(10)
# A 2-node tree produces the lowest RMSE as per graph above.


# In[ ]:


RMSE_list


# In[ ]:


#test['pop'] = np.ceil(test['pop_test'])
#test['pop'] = np.floor(test['pop_test']) 
#test[['Id', 'pop']].to_csv(path_or_buf = 'Group10_CS986_model.csv', #generates a csv file with the results
#sep = ',',
#index = None
#)


# # Random Forest Regression_1
# Random Forests are ensembles of Decision Trees, usually trained via bagging, which introduces extra randomness when growing trees. Unlike Decision Trees, which search for the best feature when splitting a node, Random Forests search for the best feature among a random subset of features. The results is greater tree diversity which trades a higher bias for a lower variance, generally yielding an overall better model.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

depth_range = range(1, 15) #tree depth
popularity=train['pop']
features=train[attribs] # features to train

RMSE_list = [] #Create a list to store the RMSE for each model

for depth in depth_range:
    RandomForestModel = RandomForestRegressor(max_depth = depth, random_state = 42)
    MSE_score = cross_val_score(RandomForestModel, train[attribs], train['pop'], scoring='neg_mean_squared_error')
    RMSE_list.append(np.mean(np.sqrt(-MSE_score)))

# Plot RMSE vs Tree Depth
plt.figure(figsize=(10,5))
plt.plot(depth_range, RMSE_list, color = 'teal', marker = 'o')
plt.xlabel('Tree Depth')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error per Tree Depth', fontsize = 13)


# In[ ]:


RMSE_list


# In[ ]:


# TRAIN MODEL
modelRFR = RandomForestRegressor(max_depth = 3, random_state = 42)
modelRFR.fit(X_train_strat, y_train_strat)


# In[ ]:


print(np.mean(np.sqrt(-MSE_score)))


# In[ ]:


# PREDICTIONS BASED ON TESTING DATAb
tree_mod2=DecisionTreeRegressor(max_depth=3, random_state=42)
tree_mod2.fit(X_train_strat, y_train_strat)
test['y_test'] = modelRFR.predict(test[attribs])
test.head(5)


# In[ ]:


test['pop_test'] = modelRFR.predict(test[attribs])
test['pop'] = np.floor(test['pop_test'])
test.head(10)


# In[ ]:


#test['pop'] = np.ceil(test['y_test'])
#test['pop'] = np.floor(test['y_test']) 
#test[['Id', 'pop']].to_csv(path_or_buf = 'Group10_CS986_model2_regression.csv',
#sep = ',',
#index = None
#) 


# # Random Forest Regression_2

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
modelRFR = RandomForestRegressor(random_state = 42)
modelRFR.fit(X_train, y_train)
y_pred= modelRFR.predict(X_test)
# Root mean squared error
mean_squared_error(y_pred, y_test, squared=False)


# # **XGBOOST**
# Using XGBoost to get accurate results and highly scalable training methods to prevent overfitting.

# In[ ]:


import xgboost as xg
xgb_r = xg.XGBRegressor(objective ='reg:squarederror')
xgb_r.fit(X_train, y_train)
y_pred= xgb_r.predict(X_test)
# Root mean squared error
mean_squared_error(y_pred, y_test, squared=False)


# # **VotingRegressor**

# In[ ]:


# from sklearn.ensemble import VotingRegressor

# soft_voting = VotingRegressor(estimators=[('forest', modelRFR), ('tree', treeModel_2), ('forest', modelRFR)])

# scores = cross_val_score(soft_voting, X_train, y, scoring="neg_mean_squared_error", cv=5)
# soft_voting_rmse_scores = np.sqrt(-scores)
# print(soft_voting_rmse_scores.mean())

# soft_voting.fit(X_train, y)


# In[ ]:


from sklearn.ensemble import VotingRegressor

soft_voting = VotingRegressor(estimators=[('forest', modelRFR), ('Linear_Reg', lin_reg), ('xgb_r', xgb_r)])
soft_voting.fit(X_train, y_train)
y_pred= soft_voting.predict(X_test)
# Root mean squared error
mean_squared_error(y_pred, y_test, squared=False)


# # **TEST PREDICTION**

# ## Our Final Model
# Since Random Forest Regression gives the best model performance compared to other models, its best use for test prediction as Mean squared error value was 10.52 which was lowest.
# 

# In[ ]:


# Train on entire dataset
from sklearn.preprocessing import StandardScaler
autoscaler = StandardScaler()
X = autoscaler.fit_transform(features)

modelRFR = RandomForestRegressor(random_state = 42)
modelRFR.fit(X, y)


# **Data Processing for test data**

# In[ ]:


# Adding new feature to the test data : How old the song is
test ['Song Age']= 2023-test['year']


# In[ ]:


# Extracting the independent variables in test data
features_test = list(test.drop(['Id', 'title','artist','year'], axis = 1))
features_test=test[features_test]
features_test.head()


# In[ ]:


# Fill missing values in top_genre with the modal value
features_test['top genre']= features_test['top genre'].fillna('adult standards')


# In[ ]:


# label encode the top genre feature for test
features_test['top genre']=le.transform(features_test['top genre'])


# In[ ]:


# make prediction on test data
x=autoscaler.transform(features_test)
y_test_prediction = modelRFR.predict(x)


# In[ ]:


# Create test data frame with predicted values
test.drop('Song Age', inplace=True, axis=1)
y_test_prediction=pd.DataFrame(y_test_prediction, columns=['prediction'])
test_data_with_prediction=pd.concat( [test,y_test_prediction ], axis=1 )
test_data_with_prediction.head(20)


# Once we trained pur model we had high value of RMSE and index eror were shown when removed null value and the outliers so we came up with categorizing the null values. Random forest was the accurate model to predict the song popularity and by using XGboost and voting regressor our RMSE decreased it was not as low as expected it might be due to many reasons like overriding of data etc. Our model will perform better as we train it and few amendments in data exploration can leed to better predictions as we tried and got score from kaggle which was 7.58 better than the previous one.

# # Conclusion:
# Using highly connected feature for popularity prediction, can help in enhancing the model output. In addition to improve the performance we can do additional data cleansing and can use one hot encoder or dummy for encoding the character variables.
# It's important to note that the accuracy of the model will depend on the quality of the dataset and the features used for prediction. Additionally, the popularity of a song is a complex phenomenon that is influenced by many factors, so the model may not be able to capture all of the nuances that affect a song's popularity. Nonetheless, random forest regression can be a useful tool for predicting the popularity of songs to a certain extent.
