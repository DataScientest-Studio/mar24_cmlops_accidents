FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./src/api/api.py /src/api/api.py
#Quid de la récupération du modèle ? -> gestion de l'artifact ?
# En attendant :
COPY ./src/models/trained_model.joblib /src/models/trained_model.joblib
