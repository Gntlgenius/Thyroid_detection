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
from sklearn.tree import DecisionTreeClassifier
import mlflow
from urllib.parse import urlparse




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
    max_depth= config["estimators"]["DecisionTree"]["params"]["max_depth"]
    criterion= config["estimators"]["DecisionTree"]["params"]["criterion"]
    min_samples_leaf= config["estimators"]["DecisionTree"]["params"]["min_samples_leaf"]
    model_dir =config["model_dir"]
    target_col = config["base"]["target_col"]


    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    train_data = pd.read_csv(train_path, sep=",")
    test_data = pd.read_csv(test_path, sep=",")

    X_train = np.array(train_data.drop(target_col, axis=1))
    y_train = np.array(train_data[target_col])

    X_test = np.array(test_data.drop(target_col, axis=1))
    y_test = np.array(test_data[target_col])

    # #XGB =  XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs)
    # clf = DecisionTreeClassifier(s)
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

    # scores = evaluate_metrics(y_test , y_pred)
    # with open(scores_file, "w") as f:
    #     json.dump(scores, f, indent=4)

    # print(scores)

    

    

    # with open(params_file, "w") as f:
    #     params = {
    #         "max_depth": max_depth,
    #         "learning_rate": learning_rate,
    #         "n_estimators": n_estimators,
    #         "n_jobs": n_jobs
    #     }
    #     json.dump(params, f, indent=4)


    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, "model.joblib")

    # joblib.dump(clf, model_path)

 ##################### #ML FLOW #######################################################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    registered_model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as mlops_run:

        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion)
        clf.fit(X_train, y_train)

        predicted_val = clf.predict(X_test)

        scores = evaluate_metrics(y_test, predicted_val)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("criterion", criterion)
        mlflow.log_metric("Accuracy", scores['Accuracy_'])
        mlflow.log_metric("Precision", scores['precision_'])
        mlflow.log_metric("Recall", scores['recall_'])
        mlflow.log_metric("F1_Score", scores['f1_Score_'])
      

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store !="file":
            mlflow.sklearn.log_model(clf, "model", registered_model_name=registered_model_name)
        else:
            mlflow.sklearn.load_model(clf, "model")

    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training_evaluation(config_path=parsed_args.config)

