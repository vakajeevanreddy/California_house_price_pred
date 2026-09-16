import pytest
from fastapi.testclient import TestClient
from App.app import app

@pytest.fixture
def client():
    """Context manager fixture ensures FastAPI lifespan runs to load models."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_payload():
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "preprocessor_loaded" in data
    assert "model_loaded" in data

def test_predict_success(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["prediction_dollar"], float)
    assert data["prediction_dollar"] > 0

def test_predict_missing_required_field(client, valid_payload):
    invalid_payload = valid_payload.copy()
    del invalid_payload["median_income"]

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_negative_room_count(client, valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["total_rooms"] = -10.0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_invalid_data_types(client, valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["longitude"] = "Invalid-String-coord"

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422