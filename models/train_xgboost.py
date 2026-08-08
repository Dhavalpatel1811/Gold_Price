# -*- coding: utf-8 -*-
"""
Trains an XGBoost classifier that predicts next-day price direction
(up/down) from a news-sentiment score.

Input:  data/raw/gold_news_data.csv (must contain 'sentiment_score',
        'Close', and 'Next_Day_Close' columns)
Output: models/artifacts/gold_price_sentiment_model.pkl
"""
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "gold_news_data.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
df["Price_Direction"] = (df["Next_Day_Close"] > df["Close"]).astype(int)

X = df[["sentiment_score"]]
y = df["Price_Direction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 4. Save
out_path = os.path.join(ARTIFACTS_DIR, "gold_price_sentiment_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model, f)
print("Model saved to", out_path)
