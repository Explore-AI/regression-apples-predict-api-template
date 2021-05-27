"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')].drop(columns='Commodities')


train['Date'] = pd.to_datetime(train['Date'])
train['Day'] = train['Date'].dt.day
train['Month'] = train['Date'].dt.month
train.drop(['Date'], inplace = True, axis = 1)

train.columns = ['province', 'container', 'size_grade', 'weight_kg', 'low_price', 
                 'high_price', 'sales_total', 'total_qty_sold','total_kg_sold', 
                 'stock_on_hand', 'avg_price_per_kg', 'day', 'month']

dummy_df = pd.get_dummies(train,drop_first=False)

X = dummy_df.drop('avg_price_per_kg',axis=1)
y = dummy_df['avg_price_per_kg']

# Fit model
xgbmodel = XGBRegressor(max_depth=2,min_child_weight=13,subsample=1,colsample_bytree=1,
            objective='reg:squarederror',n_estimators=6000, learning_rate=0.3, random_state= 16)
print ("Training Model...")
xgbmodel.fit(X, y)

# Pickle model for use within our API
save_path = '../assets/trained-models/xgbmodel.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(xgbmodel, open(save_path,'wb'))
