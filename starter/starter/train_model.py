# Script to train machine learning model.

from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
from slice import slice_func

import pandas as pd
import pickle
import os
import sys

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(file_dir)

# Add code to load in the data.
data = pd.read_csv(os.path.join(parent_dir, 'data/cleaned_census.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

model = train_model(X_train, y_train)

# Save models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

with open(os.path.join(MODEL_DIR, "label_binarizer.pkl"), "wb") as f:
    pickle.dump(lb, f)

# Inference and evaluation.
train_pred = inference(model, X_train)
test_pred = inference(model, X_test)
precision, recall, f_one = compute_model_metrics(y_test, test_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f_one}")

# Slicing
slice_func(model, encoder, lb, data, 'workclass', categorical_features=cat_features)