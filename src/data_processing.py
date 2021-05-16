import os
import pandas as pd
import numpy as np
import yaml
from get_data import read_params, get_data
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTENC, RandomOverSampler, KMeansSMOTE
import warnings
warnings.filterwarnings('ignore')



def check_missing_val(df, config_path):
    config = read_params(config_path)
    null_data_path = config['data_source']['null_dir']
    val = []
    col_ = []
    for column in df.columns:
        count = df[column][df[column]=="?"].count()
        if count > 0:
            val.append(count)
            col_.append(column)
    data = {}       
    for key in col_:
        for value in val:
            data[key]=value
            val.remove(value)
            break
    null = pd.DataFrame(data, index= range(0,1))
    if null.shape[1]>0:
        null.to_csv(null_data_path, sep=",", index=False)


    

def ColsToDrop(df, cols):
    df.drop(columns=cols, axis=1, inplace=True)
    return df
    

def replaceMissingValWithNAN(df):
    for col in df.columns:
        count = df[col].loc[df[col]=='?'].count()
        if count > 0:
            df[col] = df[col].replace("?", np.nan)
    return df

def MapCategoricalVal(df):
    for col in df.columns:
        if ((df[col].value_counts()).shape)[0] ==2:
            if 'M' in (df[col].unique()):
                df[col] = df[col].map({'F':0, 'M':1})
            else:
                df[col] = df[col].map({'f':0, 't':1})
    return df


def MissingValImputer(df):
    imputer = KNNImputer(n_neighbors=3, weights="uniform", missing_values=np.nan)

    new_array = imputer.fit_transform(df)

    df = pd.DataFrame(data=np.round(new_array), columns=df.columns)
        
    return df
    
def smoteImbalanceAndSplit(df, config_path):
    config = read_params(config_path)
    bal_data_path = config['data_source']['balanced_sampled_data']
    good_data_path = config['load_data']['good_data_csv']

    X = df.drop('Class', axis=1)
    y= df['Class']

    ros = RandomOverSampler()
    x_sampled, y_sampled = ros.fit_resample(X,y)

    (y_sampled.value_counts()).to_csv(bal_data_path, sep=",")

    processed_data = pd.concat([x_sampled, y_sampled], axis=1, sort=False)


    processed_data = shuffle(processed_data).reset_index(drop=True)

    processed_data.to_csv(good_data_path, sep=",", index=False)



    return  processed_data
       
    

