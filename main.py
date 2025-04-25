import pandas as pd
from dbrepo.RestClient import RestClient
import os
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from dotenv import load_dotenv

load_dotenv("./secrets.env")

client = RestClient(endpoint=os.environ["DB_REPO_HOST"], username=os.environ["DB_REPO_USER"], password=os.environ["DB_REPO_PASSWORD"])
training_df: pd.DataFrame = client.get_identifier_data(identifier_id="7f9bf749-903d-4722-8dbf-11e66e6af731")
validation_df: pd.DataFrame = client.get_identifier_data(identifier_id="c4c52669-3eee-41dd-bfc8-71bb290c41aa")
test_df: pd.DataFrame = client.get_identifier_data(identifier_id="42629aa2-c5a0-45d8-94ee-229124323a97")


def feature_encodings(df: pd.DataFrame, categorical_col_names: list[str]):
    """
    Transforms categorical variables to one hot encodings and converts numberic objects to numbers
    :param categorical_col_names: names of categorical variables
    :param df: The input dataframe
    :return: The input dataframe with categorical columns transformed to one hot encodings
    """

    df_with_one_hots =  pd.get_dummies(df,columns=categorical_col_names)
    df_with_one_hots = df_with_one_hots.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    return df_with_one_hots



def get_dataset_features_and_targets(input_df:pd.DataFrame):
    """
    Takes an input dataframe drops unnecessary columns,
    transforms categorical variables to one hot encodings
     and constructs a matrix optimized for xgboost.
    :param input_df: The input dataframe
    :return: xgb.DMatrix containing features and targets,
    """

    features = input_df.drop(columns=["index_col", "date_and_hour", "regression_split","rented_bike_count"])
    categorical_cols = ['seasons', 'holiday', 'functioning_day']
    features = feature_encodings(features, categorical_col_names=categorical_cols)
    target = input_df['rented_bike_count']

    feature_and_target_matrix = xgb.DMatrix(features, label=target)

    return feature_and_target_matrix

dtrain = get_dataset_features_and_targets(training_df)
dval = get_dataset_features_and_targets(validation_df)
dtest = get_dataset_features_and_targets(test_df)

# Define model parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42
}

# Train the model with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=True
)

# Predict on test set
y_pred = model.predict(dtest)

# Evaluate performance

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

    return metrics


print(evaluate_model(dtest.get_label(), y_pred))
