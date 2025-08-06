#!/usr/bin/env python
# coding: utf-8

# ### Import Necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set_style('whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import warnings
warnings.filterwarnings('ignore')


# ### Import Data Sets

# In[3]:


train_path = r"Z:\Sasindu\Data set\Sticker_Sales\train.csv"
test_path = r"Z:\Sasindu\Data set\Sticker_Sales\test.csv"


# In[4]:


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# In[5]:


train_df= pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


# In[6]:


train_df.head()


# In[7]:


test_df.head()


# In[8]:


train_df.info()


# In[9]:


test_df.info()


# In[10]:


train_df.isnull().sum()


# In[11]:


test_df.isnull().sum()


# In[12]:


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# ### EDA

# In[13]:


for col in train_df.columns:
    print(col,'---->',train_df[col].nunique())


# In[14]:


plt.figure(figsize=(12,8))
sns.histplot(train_df["num_sold"], kde=True)
plt.show()


# In[15]:


plt.figure(figsize=(12,8))
sns.boxplot(x=train_df["num_sold"])
plt.show()


# In[16]:


train_df["country"].value_counts()


# In[17]:


categorical_columns = ['country','store','product']

plt.figure(figsize=(15,10))
for  i,column in enumerate(categorical_columns,1):
    plt.subplot(3,1,i)
    sns.countplot(y = column, data = train_df, order = train_df[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Count')
    plt.ylabel(column)

plt.tight_layout()
plt.show()


# In[18]:


plt.figure(figsize=(15, 5))

for i,column in enumerate(categorical_columns, 1):
    plt.subplot(1,3,i)
    sns.histplot(data = train_df, x = "num_sold", hue =column , element ='step', bins=30)
    plt.title(f'{column} Distribution')
plt.tight_layout()
plt.show()    


# In[19]:


plt.figure(figsize=(15,5))
for i,column in enumerate(categorical_columns, 1):
    plt.subplot(1,3,i)
    sns.boxplot(y = train_df["num_sold"], x =train_df[column] ,hue =train_df[column])
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()


# In[20]:


train_df.groupby('country')['num_sold'].mean()


# In[21]:


train_df.groupby('store')['num_sold'].mean()


# In[22]:


train_df.groupby('product')['num_sold'].mean()


# ### Handling Missing values

# In[23]:


train_df['num_sold'] = train_df.groupby('country')['num_sold'].transform(lambda x: x.fillna(x.mean()))


# In[24]:


import plotnine as p9 
from plotnine import *


# In[25]:


ggplot(train_df, aes(x='date', y='num_sold')) + geom_line()


# In[26]:


ggplot(train_df, aes(x='date', y='num_sold',group='country' ,color='country'))+geom_point()+ theme_minimal()


# In[27]:


ggplot(train_df, aes(x='date', y='num_sold',group='store' ,color='store'))+geom_point()+ theme_minimal()


# In[28]:


ggplot(train_df, aes(x='date', y='num_sold',group='product' ,color='product'))+geom_point()+ theme_minimal()


# ### Feature Engineering On date column

# In[29]:


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek


# In[30]:


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek


# In[31]:


ggplot(train_df, aes(x='month', y='num_sold',group='year' ,color='year'))+geom_point()+scale_x_continuous(breaks=range(1, 13))+ theme_minimal() 


# In[32]:


train_df.head()


# ### Feature Engineering

# In[33]:


train_df["holiday"] = 0
test_df["holiday"] = 0


# In[34]:


train_df["country"].unique()
test_df["country"].unique()


# In[35]:


import holidays

ca_holidays = holidays.country_holidays('CA') # Canada
fi_holidays = holidays.country_holidays('FI') # Finland
it_holidays = holidays.country_holidays('IT') # Italy
ke_holidays = holidays.country_holidays('KE') # Kenya
no_holidays = holidays.country_holidays('NO') # Norway
sg_holidays = holidays.country_holidays('SG') # Singapore


# In[36]:


# The previous row-by-row .apply is inefficient. A vectorized approach is much faster.
holiday_map = {
    'Canada': ca_holidays, 'Finland': fi_holidays, 'Italy': it_holidays,
    'Kenya': ke_holidays, 'Norway': no_holidays, 'Singapore': sg_holidays
}
# Pre-computing sets of holiday timestamps for fast 'isin' lookup.
holiday_timestamp_sets = {
    country: set(pd.to_datetime(list(hdays.keys()))) for country, hdays in holiday_map.items()
}

# Create new DataFrames to match the original logic, which used .apply to create new ones.
df_train = train_df.copy()
df_test = test_df.copy()

# Vectorized holiday assignment for training data
train_normalized_dates = df_train['date'].dt.normalize()
for country, h_set in holiday_timestamp_sets.items():
    mask = (df_train['country'] == country) & (train_normalized_dates.isin(h_set))
    df_train.loc[mask, 'holiday'] = 1

# Vectorized holiday assignment for test data
test_normalized_dates = df_test['date'].dt.normalize()
for country, h_set in holiday_timestamp_sets.items():
    mask = (df_test['country'] == country) & (test_normalized_dates.isin(h_set))
    df_test.loc[mask, 'holiday'] = 1


# In[38]:


df_train


# #### One hot encoding

# In[39]:


df_train_encoded = pd.get_dummies(df_train, columns=['country','store','product'], dtype=np.uint8)
df_test_encoded = pd.get_dummies(df_test, columns=['country','store','product'], dtype=np.uint8)


# #### Sine Cosine Trensformation on date features

# In[40]:


def periodic_transform(dff,variable):
    dff[f"{variable}_SIN"] = np.sin(dff[variable] / dff[variable].max()*2*np.pi)
    dff[f"{variable}_COS"] = np.cos(dff[variable] / dff[variable].max()*2*np.pi)
    return dff


# In[41]:


cyclic_col = ['month','day','day_of_week']

for col in cyclic_col:
    df_train_final = periodic_transform(df_train_encoded, col)
    df_test_final = periodic_transform(df_test_encoded, col)


# In[42]:


df_train_final.columns


# In[43]:


df_test_final.columns


# #### Drop unwanted columns

# In[44]:


df_train_final = df_train_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)
df_test_final = df_test_final.drop(['month', 'day', 'day_of_week', 'date', 'id'], axis = 1)


# In[45]:


df_train_final.columns


# In[46]:


numeric_df = df_train_final.select_dtypes(include = ['number'])
corr_matrix = numeric_df.corr()


# In[47]:


print(corr_matrix['num_sold'].sort_values(ascending = False).to_string())


# In[48]:


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap = 'coolwarm', fmt = ".2f")
plt.show()


# In[49]:


x = df_train_final.drop(['num_sold'],axis =1)
y = df_train_final['num_sold']


# #### Split data

# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=42)


# #### Apply standerd scaler

# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


# In[54]:


df_test_scaled_final = mm.transform(df_test_final)


# ### Deploy model

# In[55]:


def model_acc(model):
    model.fit(x_train_scaled,y_train)
    acc = model.score(x_test_scaled,y_test)
    print(str(model)+'-->'+str(acc))


# In[56]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_acc(rf)


# ### Model Evaluation for Decision Tree

# In[57]:


y_test_pred = dt.predict(x_test_scaled)
y_test = y_test.values.flatten()
y_test_pred = y_test_pred.flatten()
final_df1 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[58]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error


# In[59]:


acc_train_dt = dt.score(x_train_scaled,y_train)
print("Model Score on Train set :",acc_train_dt)
print("Model Score on Test set :",dt.score(x_test_scaled,y_test))


# In[60]:


print(f'MAE: {mean_absolute_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'MSE: {mean_squared_error(final_df1["Prediction"],final_df1["Real"])}')
print(f'RMSE: {np.sqrt(mean_squared_error(final_df1["Prediction"],final_df1["Real"]))}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}')


# In[61]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df1['Real'])) ,y=final_df1['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df1['Prediction'])),y=final_df1['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions for Decision Tree')
plt.show()


# ### Model Evaluation for Random Forest

# In[62]:


y_test_pred = rf.predict(x_test_scaled)
y_test_pred = y_test_pred.flatten()
final_df2 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[63]:


acc_train = rf.score(x_train_scaled,y_train)
print(acc_train)


# In[64]:


print(f'MAE: {mean_absolute_error(final_df2["Prediction"],final_df2["Real"])}')
print(f'MSE: {mean_squared_error(final_df2["Prediction"],final_df2["Real"])}')
print(f'RMSE: {np.sqrt(mean_squared_error(final_df1["Prediction"],final_df1["Real"]))}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}')


# In[65]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df2['Real'])) ,y=final_df2['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df2['Prediction'])),y=final_df2['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions Random Forest')
plt.show()


# ### xgboost

# In[66]:


import xgboost as xgb


# In[67]:


train_data = xgb.DMatrix(x_train_scaled, label=y_train)
test_data = xgb.DMatrix(x_test_scaled, label=y_test)


# In[68]:


params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'learning_rate': 0.1,  # Step size shrinkage
    'max_depth': 5,  # Maximum depth of a tree
    'alpha': 10,  # L1 regularization term on weights
    'n_estimators': 100  # Number of boosting rounds (trees)
}


# In[69]:


model_xgb = xgb.train(params, train_data, num_boost_round=100)


# In[70]:


y_pred = model_xgb.predict(test_data)


# In[71]:


from sklearn.metrics import mean_absolute_error,r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')


# In[72]:


y_test_pred = y_pred.flatten()
final_df2 = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[73]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df2['Real'])) ,y=final_df2['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df2['Prediction'])),y=final_df2['Prediction'],color='red',label='Prediction')
ax.set_xlim([3000,3100])
plt.title('Real vs. Predictions XGBoost')
plt.show()


# ### Deploy Random Forest for test data set

# ##### When considering the Mean Absolute Percentage Error (MAPE) for machine learning models, the model with the lowest MAPE is considered the best. Among these models, the Random Forest model performs the best. Therefore, we can use it to predict the sticker prices of the test data.

# In[74]:


y_test_pred_rf = rf.predict(df_test_scaled_final)


# In[75]:


submission_df = pd.DataFrame({
    'id': df_test_encoded['id'],  # Extract 'id' column from the test DataFrame
    'Premium Amount': y_test_pred_rf  # Use the predictions from your model
})


# In[76]:


submission_df


# In[77]:


submission_df.to_csv('Submission.csv', index=False)
