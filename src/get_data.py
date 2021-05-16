import os
import yaml
import argparse
import pandas as pd

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

#path =r"C:\Users\USER\projects\Thyroid_project\Training_Batch_Files"
def ReadJoinAllcsv(data_path):
    
        col = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
       'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
       'I131_treatment', 'psych', 'TSH_measured', 'TSH', 'T3_measured', 'T3',
       'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured', 'FTI',
       'TBG_measured', 'TBG', 'referral_source', 'Class', 'query_hypothyroid',
       'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary']


        df = pd.DataFrame(columns=col)

        all_files = (os.listdir(data_path))
        for data in all_files:
                my_csv = pd.read_csv(data_path+"\\"+data)
                df = pd.concat([df, my_csv], axis=0, sort=False)
            
        return df
        
   


def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['training_source']

    dataset = ReadJoinAllcsv(data_path)
    return dataset




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
