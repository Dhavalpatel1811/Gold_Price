# -*- coding: utf-8 -*-
"""
Trains an LSTM model to predict the next-day scaled closing price of gold
from the past 30 days of OHLCV data.

Input:  data/processed/gold_features.xlsx
Output: models/artifacts/gold_lstm_model.keras
        models/artifacts/gold_scaler.pkl
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "gold_features.xlsx")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 30  # look back 30 days to predict the next day
FEATURES = ["Open_scaled", "High_scaled", "Low_scaled", "Close_scaled", "Volume_scaled"]

# 1. Load data
df = pd.read_excel(DATA_PATH)
if "Close_scaled" not in df.columns:
    raise ValueError("Column 'Close_scaled' not found in the dataset.")
df = df.dropna(subset=["Close_scaled"]).reset_index(drop=True)

# 2. Build sequences
data = df[FEATURES].values
X, y = [], []
for i in range(SEQUENCE_LENGTH, len(data)):
    X.append(data[i - SEQUENCE_LENGTH:i])
    y.append(data[i, FEATURES.index("Close_scaled")])
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 3. Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# 4. Train
model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=1)

# 5. Save model + scaler
model.save(os.path.join(ARTIFACTS_DIR, "gold_lstm_model.keras"))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[FEATURES])
with open(os.path.join(ARTIFACTS_DIR, "gold_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Model saved to", os.path.join(ARTIFACTS_DIR, "gold_lstm_model.keras"))
print("Scaler saved to", os.path.join(ARTIFACTS_DIR, "gold_scaler.pkl"))

# 6. Quick evaluation plot
y_pred = model.predict(X_test)
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red")
    plt.title("Gold Price Prediction (LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Scaled Price")
    plt.legend()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "lstm_eval_plot.png"))
    print("Evaluation plot saved.")
except ImportError:
    pass
