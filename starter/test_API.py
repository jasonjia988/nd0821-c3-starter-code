from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Weclome to use this API!"


def test_inference_response_schema():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/inference", json=data)
    assert response.status_code == 200

    body = response.json()
    # contract: response must contain a "prediction" field which is a string
    assert "prediction" in body
    assert isinstance(body["prediction"], str)


def test_inference_invalid_type():
    data = {
        "age": "thirty-nine",  # invalid type, should be int
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    response = client.post("/inference", json=data)
    assert response.status_code == 422
    body = response.json()
    # "age" should appear in error location
    assert any("age" in str(err["loc"]) for err in body["detail"])
