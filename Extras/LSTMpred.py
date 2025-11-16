# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ----------------------------
# 1️⃣ Get Gold Data
# ----------------------------
gold = yf.Ticker("GC=F")
data = gold.history(start="2025-08-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))

# Vérifier si 'Adj Close' existe
if 'Adj Close' not in data.columns:
    data['Adj Close'] = data['Close']

data = data[['Close', 'Adj Close']]
print("Latest data:\n", data.head(2))

# ----------------------------
# 2️⃣ Feature Scaling
# ----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# ----------------------------
# 3️⃣ Prepare sequences for LSTM
# ----------------------------
sequence_length = 20  # nombre de jours à regarder pour prédire le suivant
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # séquence de 20 jours
    y.append(scaled_data[i, 0])                 # Close du jour suivant

X, y = np.array(X), np.array(y)
print("Shape X:", X.shape, "Shape y:", y.shape)

# ----------------------------
# 4️⃣ Split Train/Test
# ----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 5️⃣ Build LSTM Model
# ----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ----------------------------
# 6️⃣ Train the model
# ----------------------------
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# ----------------------------
# 7️⃣ Predictions
# ----------------------------
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.concatenate([predicted, np.zeros((predicted.shape[0],1))], axis=1))[:,0]
real_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1), np.zeros((y_test.shape[0],1))], axis=1))[:,0]

# ----------------------------
# 8️⃣ Visualisation
# ----------------------------
plt.figure(figsize=(12,6))
plt.plot(real_prices, color='blue', label='Actual Gold Price')
plt.plot(predicted_prices, color='red', label='Predicted Gold Price (LSTM)')
plt.title('Gold Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
