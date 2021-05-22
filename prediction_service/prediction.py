import yaml
import os
import joblib
import numpy as np
import json

params_path ="params.yaml"
schema_path = os.path.join("prediction_service", "schema.json")
model_dir = os.path.join("saved_models", "model.joblib")



class NotInRange(Exception):
    def __init__(self, message = "values entered not in range"):
        self.message = message
        super().__init__(self.message)

class NotInColumn(Exception):
    def __init__(self, message = "values entered not in column"):
        self.message = message
        super().__init__(self.message)


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predicter(data):
    model_path = model_dir
    model = joblib.load(model_path)
    prediction = model.predict(data)
    prediction= prediction[0]

    try:
        if 0 <= prediction <= 3:
            if prediction == 0:
                return "Detection of Possible Compensated Hypothyroid"
            elif prediction == 1:
                return "Negative"
            elif prediction == 2:
                return "Detection of Possible Primary Hypothyroid"
            else:
                return "Detection of Possible Secondary Hypothyroid"
            
        else:
            raise NotInRange
    except NotInRange:
        return "unexpected result"



def data_validation(dict_req):
    def col_validation(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInColumn


    def val_validation(col, val):
        schema = get_schema()

        if not (schema[col]['min'] <= float(dict_req[col]) <= schema[col]['max']):
            raise NotInRange

    for col, val in dict_req.items():
        col_validation(col)
        val_validation(col, val)

    return True



def form_response(dict_req):
    if data_validation(dict_req):
            data = dict_req.values()
            data = np.array([list(map(float, data))])
            model_path = model_dir
            response = predicter(data)
            return response
    

    
def api_response(dict_req):
    try:
        if data_validation(dict_req):
            data = np.array([list(dict_req.values())])
            response = predicter(data)
            response = {'response':response}
            return response

    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInColumn as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response
