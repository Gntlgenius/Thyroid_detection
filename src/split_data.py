import os
import pandas as pd
import argparse
from get_data import read_params
from sklearn.model_selection import train_test_split

def split_and_save_data(config_path):
    config = read_params(config_path)
    good_data = config["load_data"]["good_data_csv"]
    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    size = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(good_data, sep=",")

    train, test = train_test_split(df, test_size=size, random_state=random_state)

    train.to_csv(train_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_path, sep=",", index=False, encoding="utf-8")



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)