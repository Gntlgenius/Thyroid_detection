from data_processing import check_missing_val, ColsToDrop, replaceMissingValWithNAN, MapCategoricalVal
from data_processing import MissingValImputer, smoteImbalanceAndSplit
from get_data import read_params, get_data
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_process_data(config_path):
    config = read_params(config_path)

    df = get_data(config_path)

    check_missing_val(df, config_path)

    cols= ['TBG','TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']

    df = ColsToDrop(df, cols)

    df = replaceMissingValWithNAN(df)

    df = MapCategoricalVal(df)

    df = pd.get_dummies(df, columns=["referral_source"])

    le = LabelEncoder()

    df['Class'] = le.fit_transform(df['Class'])

    df = MissingValImputer(df)

    df = smoteImbalanceAndSplit(df, config_path)

    print(df.head())


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_process_data(config_path=parsed_args.config)