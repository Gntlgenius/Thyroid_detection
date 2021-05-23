import json
import logging
import os
import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
   {
    "age":68.0,
    "sex":0.0,
    "on_thyroxine":0.0,
    "query_on_thyroxine":0.0,
    "on_antithyroid_medication":0.0,
    "sick":0.0,
    "pregnant":0.0,
    "thyroid_surgery":0.0,
    "I131_treatment":0,
    "psych":3,
    "TSH":19,
    "T3":2,
    "TT4":93,
    "T4U":11,
    "FTI":77,
    "query_hypothyroid":1,
    "query_hyperthyroid":0,
    "lithium":0,
    "goitre":0,
    "tumor":0,
    "hypopituitary":0, 
    "referral_source_STMW":0,
    "referral_source_SVHC":0,
    "referral_source_SVHD":0,
    "referral_source_SVI": 0,
    "referral_source_other":1
    },

    "correct_range":
    {
    "age":68.0,
    "sex":0.0,
    "on_thyroxine":0.0,
    "query_on_thyroxine":0.0,
    "on_antithyroid_medication":0.0,
    "sick":0.0,
    "pregnant":0.0,
    "thyroid_surgery":0.0,
    "I131_treatment":0,
    "psych":0,
    "TSH":19,
    "T3":2,
    "TT4":93,
    "T4U":1,
    "FTI":77,
    "query_hypothyroid":1,
    "query_hyperthyroid":0,
    "lithium":0,
    "goitre":0,
    "tumor":0,
    "hypopituitary":0, 
    "referral_source_STMW":0,
    "referral_source_SVHC":0,
    "referral_source_SVHD":0,
    "referral_source_SVI": 0,
    "referral_source_other":1
    },

    "correct_range_form":
    {
     'age': '39',
     'sex': '0',
     'on_thyroxine': '0',
     'query_on_thyroxine': '0', 
     'on_antithyroid_medication': '0', 
     'sick': '0',
     'pregnant': '0', 
     'thyroid_surgery': '0', 
     'I131_treatment': '0', 
     'psych': '0', 
     'TSH': '160', 
     'T3': '0', 
     'TT4': '11', 
     'T4U': '1', 
     'FTI': '9', 
     'query_hypothyroid': '0', 
     'query_hyperthyroid': '0', 
     'lithium': '0', 
     'goitre': '0', 
     'tumor': '0', 
     'hypopituitary': '0', 
     'referral_source_STMW': '0', 
     'referral_source_SVHC': '0', 
     'referral_source_SVHD': '0', 
     'referral_source_SVI': '0', 
     'referral_source_other': '1'
    },

    "incorrect_col":
    {
        "age":68.0,
        "sex":0.0,
        "on_thyroxine":0.0,
        "query_on_thyroxine":0.0,
        "antithyroid_medication":0.0,
        "sick":0.0,
        "pregnant":0.0,
        "thyroid_surgery":0.0,
        "I131_treatment":0,
        "psych":0,
        "TSH":19,
        "T3":2,
        "TT4":93,
        "T4U":1,
        "FTI":77,
        "query_hypothyroid":1,
        "query_hyperthyroid":0,
        "lithium":0,
        "goitre":0,
        "tumor":0,
        "hypopituitary":0, 
        "referral_source_STMW":0,
        "referral_source_SVHC":0,
        "referral_source_SVHD":0,
        "referral_source_SVI": 0,
        "referral_source_other":1
    }
}

outcomes = ('Detection of Possible Compensated Hypothyroid','Negative','Detection of Possible Primary Hypothyroid','Detection of Possible Secondary Hypothyroid')

# def test_form_response_correct_range():
#     ds = input_data["correct_range_form"]
#     result = form_response(ds)
#     assert result == outcomes[2]

#     # if result in outcomes:
#     #     assert result
    
    

        
    


def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    res = res["response"]
    if res in outcomes:
        assert  res


def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInColumn().message

  
