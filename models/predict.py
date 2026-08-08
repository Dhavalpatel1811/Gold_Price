# -*- coding: utf-8 -*-
"""
Hybrid gold price predictor.

Combines three signals into one prediction:
  1. FinBERT sentiment analysis on the latest gold-related news headline
  2. XGBoost classifier -> predicted price direction from that sentiment
  3. LSTM model -> predicted next-day price from the last 30 days of OHLCV data
  4. SHAP -> explains which factor drove the XGBoost prediction

Requires a NEWSAPI_KEY environment variable (see .env.example).
Get a free key at https://newsapi.org.

Run from the repo root:
    python models/predict.py
"""
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import shap
import torch
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from transformers import BertForSequenceClassification, BertTokenizer

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "gold_features.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log_step(name, start=None):
    if start:
        print(f"done: {name} ({time.time() - start:.2f}s)")
    else:
        print(f"\n{name}...")
        return time.time()


start_time = time.time()

# ---------------------------------------------------------------
# 1. Load trained models
# ---------------------------------------------------------------
t0 = log_step("Loading models")
with open(os.path.join(ARTIFACTS_DIR, "gold_price_sentiment_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)
lstm_model = load_model(os.path.join(ARTIFACTS_DIR, "gold_lstm_model.keras"))
with open(os.path.join(ARTIFACTS_DIR, "gold_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
log_step("Loading models", t0)

# ---------------------------------------------------------------
# 2. FinBERT sentiment analysis
# ---------------------------------------------------------------
t1 = log_step("Loading FinBERT sentiment model")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)
log_step("Loading FinBERT sentiment model", t1)


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = finbert(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    label_id = torch.argmax(probs).item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[label_id], float(probs[0][label_id])


# ---------------------------------------------------------------
# 3. Fetch latest gold news
# ---------------------------------------------------------------
t2 = log_step("Fetching latest gold news")
api_key = os.environ.get("NEWSAPI_KEY")
if not api_key:
    raise RuntimeError(
        "NEWSAPI_KEY is not set. Copy .env.example to .env and add your key "
        "(get a free one at https://newsapi.org)."
    )

query = '("gold price" OR "gold market" OR "gold futures" OR "gold demand") AND NOT bitcoin AND NOT crypto'
url = (
    f"https://newsapi.org/v2/everything?q={query}"
    f"&sortBy=publishedAt&language=en&pageSize=1&apiKey={api_key}"
)
response = requests.get(url).json()
article = response.get("articles", [None])[0]

if article:
    news = article["title"]
    news_date = article["publishedAt"].split("T")[0]
else:
    news = "No relevant news found"
    news_date = datetime.now().strftime("%Y-%m-%d")

print(f"News: {news} ({news_date})")
log_step("Fetching latest gold news", t2)

# ---------------------------------------------------------------
# 4. XGBoost news-based prediction
# ---------------------------------------------------------------
t3 = log_step("Predicting news-based sentiment direction")
sent_label, sent_score = get_sentiment(news)
X_news = pd.DataFrame([{"sentiment_score": sent_score}])
xgb_pred = int(xgb_model.predict(X_news)[0])
xgb_conf = float(xgb_model.predict_proba(X_news)[0][xgb_pred])
direction_text = "UP" if xgb_pred == 1 else "DOWN"
log_step("Predicting news-based sentiment direction", t3)

# ---------------------------------------------------------------
# 5. LSTM time-series prediction
# ---------------------------------------------------------------
t4 = log_step("Predicting next-day gold price (LSTM)")
df = pd.read_excel(DATA_PATH)
features = ["Open_scaled", "High_scaled", "Low_scaled", "Close_scaled", "Volume_scaled"]
data = df[features].values
sequence_len = 30
X_input = np.array([data[-sequence_len:]])
lstm_pred = float(lstm_model.predict(X_input)[0][0])
log_step("Predicting next-day gold price (LSTM)", t4)

predicted_price = (
    scaler.inverse_transform(np.concatenate([np.zeros((1, 4)), [[lstm_pred]]], axis=1))[0, 4]
    if lstm_pred
    else lstm_pred
)

# ---------------------------------------------------------------
# 6. Combine predictions
# ---------------------------------------------------------------
combined_direction = "UP" if (xgb_pred == 1 and lstm_pred > df["Close_scaled"].iloc[-1]) else "DOWN"
confidence = (xgb_conf + 0.5) / 2  # simple heuristic average

# ---------------------------------------------------------------
# 7. SHAP explanation for the XGBoost component
# ---------------------------------------------------------------
t5 = log_step("Explaining prediction with SHAP")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_news)
feature_impact = shap_values.values[0][0]
reason = (
    f"Positive sentiment pushed prediction UP (+{feature_impact:.3f})"
    if feature_impact > 0
    else f"Negative sentiment pulled prediction DOWN ({feature_impact:.3f})"
)
log_step("Explaining prediction with SHAP", t5)

# ---------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------
t6 = log_step("Saving results")
txt_file = os.path.join(OUTPUT_DIR, "prediction_latest.txt")
csv_file = os.path.join(OUTPUT_DIR, "prediction_data.csv")

with open(txt_file, "w", encoding="utf-8") as f:
    f.write("=== HYBRID GOLD PRICE PREDICTION REPORT ===\n\n")
    f.write(f"Date: {news_date}\n")
    f.write(f"News: {news}\n\n")
    f.write(f"Sentiment: {sent_label} ({sent_score:.3f})\n")
    f.write(f"XGBoost Prediction: {direction_text} (Confidence: {xgb_conf:.2f})\n")
    f.write(f"LSTM Predicted Price: {predicted_price:.2f}\n")
    f.write(f"Hybrid Final Prediction: {combined_direction}\n\n")
    f.write(f"Explanation: {reason}\n")
    f.write(f"Completed in {time.time() - start_time:.2f}s\n")
    f.write("=" * 55)

if os.path.exists(csv_file):
    df_csv = pd.read_csv(csv_file)
else:
    df_csv = pd.DataFrame(columns=[
        "date", "news", "sentiment_label", "sentiment_score",
        "xgb_pred", "xgb_conf", "lstm_pred", "combined_direction",
    ])

if not df_csv.empty and news_date in df_csv["date"].values:
    df_csv = df_csv[df_csv["date"] != news_date]

new_row = pd.DataFrame([{
    "date": news_date,
    "news": news,
    "sentiment_label": sent_label,
    "sentiment_score": sent_score,
    "xgb_pred": direction_text,
    "xgb_conf": xgb_conf,
    "lstm_pred": lstm_pred,
    "combined_direction": combined_direction,
}])
df_csv = pd.concat([df_csv, new_row], ignore_index=True)
df_csv.to_csv(csv_file, index=False)

log_step("Saving results", t6)
print(f"\nTXT saved: {txt_file}")
print(f"CSV updated: {csv_file}")
print(f"Done in {time.time() - start_time:.2f}s")
