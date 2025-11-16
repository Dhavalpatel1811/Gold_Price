# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

# ======================================
# 1️⃣ Load Data
# ======================================
df = pd.read_excel("../data/gold_features.xlsx")

# Use the scaled close price for prediction
if "Close_scaled" not in df.columns:
    raise ValueError("Column 'Close_scaled' not found in the dataset.")

# Drop NaN rows
df = df.dropna(subset=["Close_scaled"]).reset_index(drop=True)

# ======================================
# 2️⃣ Create sequences for LSTM
# ======================================
sequence_length = 30  # use past 30 days to predict next day
features = ["Open_scaled", "High_scaled", "Low_scaled", "Close_scaled", "Volume_scaled"]

data = df[features].values
X, y = [], []

for i in range(sequence_length, len(data)):
    X.append(data[i-sequence_length:i])
    y.append(data[i, features.index("Close_scaled")])

X, y = np.array(X), np.array(y)

# Split Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("✅ X_train shape:", X_train.shape)
print("✅ X_test shape:", X_test.shape)

# ======================================
# 3️⃣ Build the LSTM Model
# ======================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ======================================
# 4️⃣ Train the Model
# ======================================
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ======================================
# 5️⃣ Save Model and Scaler
# ======================================
model.save("gold_lstm_model.keras")

# Save the scaler used for features (optional)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[features])

with open("gold_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model saved as 'gold_lstm_model.keras'")
print("✅ Scaler saved as 'gold_scaler.pkl'")

# ======================================
# 6️⃣ Evaluate Model
# ======================================
y_pred = model.predict(X_test)

# Convert back to original price scale (optional)
# You can inverse transform using your scaler if needed

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title("Gold Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Scaled Price")
plt.legend()
plt.show()
