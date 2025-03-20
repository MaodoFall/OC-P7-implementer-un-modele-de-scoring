from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import pickle

app = FastAPI()

# Charger les données
train_data = pd.read_parquet("rest_api/data/data_train_feature_engine.parquet")
test_data = pd.read_parquet("rest_api/data/test_data_sample_1000.parquet")

# Ajouter la colonne client_id
train_data["client_id"] = range(1, len(train_data) + 1)
test_data["client_id"] = range(1, len(test_data) + 1)

# Séparer les features et la target dans les données d'entraînement
X_train = train_data.drop(["TARGET", "client_id"], axis=1)
y_train = train_data["TARGET"]

# Appliquer le scaler aux données d'entraînement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Charger le modèle pré-entraîné
model = pickle.load(open("rest_api/model/lgbm_model_prod.pkl","rb"))

# Créer un explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values_train = explainer(X_train_scaled, check_additivity=False)
global_shap_values = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(
    list(zip(X_train_scaled.columns, global_shap_values)), columns=["Features", "Importance_value"]
).sort_values(by="Importance_value", ascending=False)

class ClientData(BaseModel):
    client_id: int
    data: dict = {}

@app.get("/")
def home():
    return {"message": "API pour prédire la solvabilité du client"}

# Récupérer les infos d’un client
@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    client_data = test_data[test_data["client_id"] == client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    return client_data.to_dict(orient="records")[0]

# Prédiction
@app.post("/predict")
def predict(client_data: ClientData):
    client_id = client_data.client_id
    client_row = test_data[test_data["client_id"] == client_id]
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    info_client = client_row.drop("client_id", axis=1)
    info_client_scaled = scaler.transform(info_client)
    predicted_proba_default = model.predict_proba(info_client_scaled)[0][1]
    best_threshold = 0.54
    client_status = "Client non solvable" if predicted_proba_default >= best_threshold else "Client solvable"
    return {"probabilité": predicted_proba_default, "prédiction": client_status}

# Récupérer les 15 features les plus importantes globalement
@app.get("/feature_importance_global")
def global_feature_importance():
    features_top15 = global_shap_importance.head(15)
    return features_top15.set_index("Features")["Importance_value"].to_dict()

# Récupérer les 15 features les plus importantes localement
@app.get("/feature_importance_local/{client_id}")
def local_feature_importance(client_id: int):
    client_row = test_data[test_data["client_id"] == client_id]
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    info_client = client_row.drop("client_id", axis=1)
    info_client_scaled = scaler.transform(info_client)
    shap_values = explainer(info_client_scaled, check_additivity=False)
    local_shap_values = np.abs(shap_values.values[0])
    local_shap_importance = pd.DataFrame(
        list(zip(info_client.columns, local_shap_values)), columns=["Features", "Importance_value"]
    ).sort_values(by="Importance_value", ascending=False)
    return local_shap_importance.set_index("Features")["Importance_value"].to_dict()

# Récupérer une ou deux caractéristiques
@app.get("/feature_distribution/{feature}")
def feature_distribution(feature: str):
    if feature not in test_data.columns:
        raise HTTPException(status_code=400, detail="Feature not found in dataset")
    return {feature: test_data[feature].dropna().tolist()}

# Mettre à jour les infos d’un client
@app.put("/client_info/{client_id}")
def update_client_info(client_id: int, client_data: dict):
    global test_data
    if client_id not in test_data["client_id"].tolist():
        raise HTTPException(status_code=404, detail="Client not found")
    test_data.loc[test_data["client_id"] == client_id, list(client_data.keys())] = list(client_data.values())
    prediction = predict(ClientData(client_id=client_id, data=client_data))
    return {"message": "Client informations updated", "prediction": prediction}

# Ajouter un nouveau client
@app.post("/client_info")
def new_client(client_data: dict):
    global test_data
    new_client_id = len(test_data) + 1
    client_data["client_id"] = new_client_id
    test_data = pd.concat([test_data, pd.DataFrame(client_data, index=[0])], ignore_index=True)
    prediction = predict(ClientData(client_id=new_client_id, data=client_data))
    return {"message": "New client added", "client_id": new_client_id, "prediction": prediction}
