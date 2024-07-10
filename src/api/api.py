from fastapi import FastAPI
import sklearn
import sklearn.metrics as metrics
import pandas as pd 
from sklearn import ensemble
import joblib
import json
import numpy as np
from pydantic import BaseModel
import os
from datetime import datetime


g_rf_classifier = None
g_model_filename = './src/models/trained_model.joblib'
g_test_features = './src/models/test_features0.json'
#g_test_features = './src/models/test_features1.json'
#g_model_filename = 'D:/Development/Python/mar24_cmlops_accidents/src/models/trained_model.joblib'
#g_model_filename = '../models/trained_model.joblib'


api = FastAPI()


class Prediction(BaseModel):
    place:int
    catu:int
    sexe:int
    secu1:int
    year_acc:int
    victim_age:float
    catv:float
    obsm:float
    motor:float
    catr:int
    circ:float
    surf:float
    situ:float
    vma:float
    jour:int
    mois:int
    lum:int
    dep:int
    com:int
    agg_:int
    int:int
    atm:float
    col:float
    lat:float
    long:float
    hour:int
    nb_victim:int
    nb_vehicules:int


def predict(features: dict = None) -> int:
    global g_rf_classifier
    global g_model_filename    
    #input_df = pd.DataFrame([features])
    input_df = pd.json_normalize(json.load(open(g_test_features)))
    
    # Chargement du modèle
    if g_rf_classifier == None:
        g_rf_classifier = joblib.load(g_model_filename)
       
    prediction = g_rf_classifier.predict(input_df)
    return int(prediction[0])   # convert np.int64 to int to avoid json exception

def evaluate():
    X_test = pd.read_csv("./data/processed/X_test.csv")
    y_test = pd.read_csv("./data/processed/y_test.csv")

    y_test = y_test.values.ravel()
    
    model = joblib.load(g_model_filename)

    predictions = model.predict(X_test)
 
    prediction_csv = pd.DataFrame({"target": y_test, "predicted": predictions})
    prediction_filename = f'src/models/predictions/prediction_{datetime.today().strftime("%Y%m%d%H%M%S")}.csv'
    prediction_csv.to_csv(prediction_filename, index=False, header=True)

    mse = metrics.mean_squared_error(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions)
    
    metrics_filename = f'src/models/scores/scores_{datetime.today().strftime("%Y%m%d%H%M%S")}.json'
    with open(metrics_filename, "w") as fd:
        json.dump({"mse": mse, "precision": precision, "recall": recall, "f1": f1}, fd, indent=2)

    print(prediction_filename)


@api.get('/status')
def get_status():
    return {'status': 'ok'}


@api.post('/prediction')
def post_prediction(prediction: Prediction):
    """
    Exemple test avec curl
    curl -X 'POST' 'http://127.0.0.1:8000/prediction' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"place": 0, "catu": 0, "sexe": 0, "secu1": 0, "year_acc": 0, "victim_age": 0, "catv": 0, "obsm": 0, "motor": 0, "catr": 0,  "circ": 0,  "surf": 0,  "situ": 0,  "vma": 0,  "jour": 0, "mois": 0, "lum": 0, "dep": 0, "com": 0, "agg_": 0, "int": 0, "atm": 0, "col": 0, "lat": 0, "long": 0, "hour": 0, "nb_victim": 0, "nb_vehicules": 0}'
    """    
    l_prediction = predict(prediction.dict())
    return {'prediction':l_prediction}
    


if __name__ == '__main__':
    #
    #
    # Uniquement à des fins de debug en manuel (sans passer par le serveur FastAPI)
    #
    #
    """  
    features_4_prediction = {'place':10,
                             'catu':3,
                             'sexe':2,
                             'secu1':0.0,
                             'year_acc':2021,
                             'victim_age':19.0,
                             'catv':2.0,
                             'obsm':1.0,
                             'motor':1.0,
                             'catr':4,
                             'circ':2.0,
                             'surf':1.0,
                             'situ':1.0,
                             'vma':30.0,
                             'jour':4,
                             'mois':11,
                             'lum':5,
                             'dep':59,
                             'com':59350,
                             'agg_':2,
                             'int':2,
                             'atm':0.0,
                             'col':6.0,
                             'lat':50.6325934047,
                             'long':3.0522062542,
                             'hour':22,
                             'nb_victim':4,
                             'nb_vehicules':1
                            }
    """   
    #ma_prediction = predict(features_4_prediction)
    ma_prediction = predict()
    print(ma_prediction)
    
    eval = evaluate()
    
    
    