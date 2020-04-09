'''
Author : Imad Dabbura
Restful API for TIA model.
'''

import pandas as pd
from flask import Flask, jsonify, request, abort
from joblib import load

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def api_call():
    try:
        test_json = request.get_json()
        test_df = pd.read_json(test_json, orient='records')
    except Exception as e:
        raise e

    # Check if there is no data
    if test_df.empty:
        abort(400, 'Please provide data to get a TIA prediction.')
    patient_ids, X_test = preprocess_data(test_df)
    preds = predict(X_test)
    final_preds = pd.DataFrame(
        list(zip(patient_ids, preds)), columns=['PatientID', 'Predictions'])
    final_preds['Predictions'] = final_preds.Predictions.map({0: 'No', 1: 'Yes'})
    responses = jsonify(predictions=final_preds.to_json(orient='records'))
    responses.status_code = 200
    return responses


def preprocess_data(df):
    feat_names = load('../models/column-names')
    # Check if some of the fields are missing
    if set(feat_names) != set(df.columns.values.tolist()):
        missing_fields = set(feat_names).difference(
            set(df.columns.values.tolist()))
        abort(400, f'Please provide the following fields {missing_fields}')
    df = df[feat_names]
    patient_ids = df['PatientID']
    df = df.drop(columns=['PatientID'])
    return patient_ids, df.values


def predict(X_test):
    model = load('../models/final-model')
    preds = model.predict(X_test)
    return preds
