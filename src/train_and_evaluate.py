# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from get_data import read_params
import argparse
import joblib
import json


def evaluate_metrics(y_test, y_pred):
    scoring= {
                'Accuracy_': accuracy_score(y_test, y_pred),
                'precision_' : precision_score(y_test,y_pred, average = 'macro'),
                'recall_' : recall_score(y_test,y_pred, average = 'macro'),
                'f1_Score_' : f1_score(y_test,y_pred, average = 'weighted')
             }
    

    return scoring


def training_evaluation(config_path):
    config = read_params(config_path)

    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    max_depth= config["estimators"]["XGBClassifier"]["params"]["max_depth"]
    learning_rate= config["estimators"]["XGBClassifier"]["params"]["learning_rate"]
    n_estimators= config["estimators"]["XGBClassifier"]["params"]["n_estimators"]
    n_jobs= config["estimators"]["XGBClassifier"]["params"]["n_jobs"]
    model_dir =config["model_dir"]
    target_col = config["base"]["target_col"]


    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    train_data = pd.read_csv(train_path, sep=",")
    test_data = pd.read_csv(test_path, sep=",")

    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]

    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]

    model =  XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores = evaluate_metrics(y_test , y_pred)
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)

    print(scores)

    

    

    with open(params_file, "w") as f:
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs
        }
        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)

    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training_evaluation(config_path=parsed_args.config)

