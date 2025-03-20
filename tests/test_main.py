import pytest
from fastapi.testclient import TestClient
from rest_api.main import app

client = TestClient(app)

@pytest.fixture
def sample_client_data():
    """Données réduites pour un client fictif"""
    return {
        "client_id": 5062,
        "data": {
            "DAYS_BIRTH": -12000,
            "AMT_CREDIT": 50000.0,
            "INCOME_CREDIT_PERC": 0.8,
            "EXT_SOURCE_2": 0.5
        }
    }

def test_home():
    """Test de la route principale"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API pour prédire la solvabilité du client"}

def test_get_client_info_existing():
    """Test récupération d'un client existant"""
    response = client.get("/client_info/5062")
    assert response.status_code == 200
    assert "client_id" in response.json()

def test_get_client_info_not_found():
    """Test récupération d'un client inexistant"""
    response = client.get("/client_info/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Client not found"

def test_predict_existing_client(sample_client_data):
    """Test de la prédiction pour un client existant"""
    response = client.post("/predict", json=sample_client_data)
    assert response.status_code == 200
    json_data = response.json()
    assert "probabilité" in json_data
    assert "prédiction" in json_data

def test_predict_client_not_found():
    """Test prédiction avec un client inexistant"""
    response = client.post("/predict", json={"client_id": 99999, "data": {}})
    assert response.status_code == 404
    assert response.json()["detail"] == "Client not found"

def test_global_feature_importance():
    """Test récupération des features les plus importantes globalement"""
    response = client.get("/feature_importance_global")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_local_feature_importance_existing():
    """Test récupération des features importantes pour un client existant"""
    response = client.get("/feature_importance_local/5062")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_local_feature_importance_not_found():
    """Test récupération des features pour un client inexistant"""
    response = client.get("/feature_importance_local/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Client not found"
