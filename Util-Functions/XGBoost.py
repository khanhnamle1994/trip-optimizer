'''
Author: James Le
Filename: XGBoost.py
Description: Build a Gradient Boosted Decision Trees machine learning model
'''

# Import Packages
import pandas as pd
import numpy as np
import xgboost as xgb

import pickle
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

def featureEngineer(data):
    '''
    Function to pre-process and engineer features of the train data
    '''
    # Convert character variables to numeric
    f = lambda x: 0 if x == 'N' else 1
    data["store_and_fwd_flag"] = data["store_and_fwd_flag"].apply(lambda x: f(x))

    # Convert datetime strings into datetime
    data["dropoff_datetime"] = pd.to_datetime(data["dropoff_datetime"], format='%Y-%m-%d %H:%M:%S')
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')

    # Now construct other variables, like month, date, etc.
    data["pickup_month"] = data["pickup_datetime"].dt.month
    data["pickup_day"] = data["pickup_datetime"].dt.day
    data["pickup_weekday"] = data["pickup_datetime"].dt.weekday
    data["pickup_hour"] = data["pickup_datetime"].dt.hour
    data["pickup_minute"] = data["pickup_datetime"].dt.minute

    # Get latitude and longitude differences
    data["latitude_difference"] = data["dropoff_latitude"] - data["pickup_latitude"]
    data["longitude_difference"] = data["dropoff_longitude"] - data["pickup_longitude"]

    # Convert duration to minutes for easier interpretation
    data["trip_duration"] = data["trip_duration"].apply(lambda x: round(x/60))


    # Convert trip distance from longitude and latitude differences to Manhattan distance.
    data["trip_distance"] = 0.621371 * 6371 *
        (abs(2 * np.arctan2(
            np.sqrt(
                np.square(np.sin((abs(data["latitude_difference"]) * np.pi / 180) / 2))
            ), np.sqrt(
                1 - (np.square(np.sin((abs(data["latitude_difference"]) * np.pi / 180) / 2)))
            )
        )) +
        abs(2 * np.arctan2(
            np.sqrt(
                np.square(np.sin((abs(data["longitude_difference"]) * np.pi / 180) / 2))
            ), np.sqrt(
                1 - (np.square(np.sin((abs(data["longitude_difference"]) * np.pi / 180) / 2)))
            )
        )))

def rmsle(y_true, y_pred):
    '''
    Function to define evaluation metric
    >> Input: y_true -- ground truth labels, y_pred -- predicted labels
    >> Output: evaluation metric
    '''
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def XGBmodel(X, y):
    '''
    Function to train a XGBoost machine learning model on the data
    >> Input: X -- features, y -- label
    >> Output:
    '''
    # Split the train data into training, test, and valdiation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2019)

    # XGBoost parameters
    params = {
        'booster':            'gbtree',
        'objective':          'reg:linear',
        'learning_rate':      0.05,
        'max_depth':          14,
        'subsample':          0.9,
        'colsample_bytree':   0.7,
        'colsample_bylevel':  0.7,
        'silent':             1,
        'feval':              'rmsle'
    }

    # Define train and validation sets
    dtrain = xgb.DMatrix(X_train, np.log(y_train+1))
    dval = xgb.DMatrix(X_val, np.log(y_val+1))

    # this is for tracking the error
    watchlist = [(dval, 'eval'), (dtrain, 'train')]

    # Number of training rounds
    nrounds = 1000

    # Train model
    gbm = xgb.train(params, dtrain, num_boost_round = nrounds, evals = watchlist, verbose_eval = True)

    # Test predictions
    y_pred = np.exp(gbm.predict(xgb.DMatrix(X_test))) - 1

    # Use mean absolute error to get a basic estimate of the error
    mae = (abs(y_pred - y_test)).mean()

    # Take a look at feature importance
    feature_scores = gbm.get_fscore()
    # Feature scaling
    summ = 0
    for key in feature_scores:
        summ = summ + feature_scores[key]
    for key in feature_scores:
        feature_scores[key] = feature_scores[key] / summ

    print('Mean Absolute Error:', mae)
    print('Feature Importance:', feature_scores)

    return gbm

if __name__ == '__main__':
    # Read train file
    taxiDB = pd.read_csv('Taxi-Trip-Duration-Data/train.csv')

    # Engineer features
    featureEngineer(taxiDB)

    # Get features and labels for the data
    X = taxiDB.drop(["trip_duration", "id", "vendor_id", "pickup_datetime", "dropoff_datetime"], axis=1)
    y = taxiDB["trip_duration"]

    # Train XGB Model to our data
    model = XGBmodel(X, y)
    filename = "xgb_model.sav"
    pickle.dump(model, open(filename, 'wb'))
