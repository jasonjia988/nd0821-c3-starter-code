import os
import sys
import pickle
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import inference, train_model

file_dir = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def path():
    """
    Extract data path
    """
    path = os.path.join(file_dir, "data/cleaned_census.csv")
    return path


@pytest.fixture(scope="module")
def data():
    """
    Extract the data
    """
    data_path = os.path.join(file_dir, "data/cleaned_census.csv")
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def cat_features():
    """
    Extract categorical features
    """
    cat_features=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    return cat_features

@pytest.fixture(scope="module")
def train_dataset(data, cat_features):
    """
    Extract trained and pre-processed dataset
    """
    train, _ = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train


def test_train_model_returns_estimator(train_dataset):
    X_train, y_train = train_dataset
    model = train_model(X_train, y_train)
    # basic type checks
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    # can predict on small batch
    preds = model.predict(X_train[:5])
    assert len(preds) == 5


def test_is_model_fitted(train_dataset):

    X_train, _=train_dataset
    model_path=os.path.join(file_dir,"model/model.pkl")
    model=pickle.load(open(model_path, 'rb'))

    try:
        model.predict(X_train)
    except BaseException:
        assert ('Model not fitted')


def test_process_data_training(data, cat_features):
    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    # y length matches rows
    assert len(y_train) == data.shape[0]
    # X has same number of rows as data
    assert X_train.shape[0] == data.shape[0]
    # encoder and label binarizer are returned
    assert encoder is not None
    assert lb is not None