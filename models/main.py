# hybrid_gold_predictor.py
import os
import time
import requests
import pandas as pd
import numpy as np
import pickle
import torch
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# =========================================
# Helper: Progress Logger
# =========================================
def log_step(name, start=None):
    if start:
        print(f"✅ {name} done in {time.time() - start:.2f}s")
    else:
        print(f"\n🚀 {name}...")
        return time.time()

start_time = time.time()

# =========================================
# STEP 1: Load Models
# =========================================
t0 = log_step("Loading models")

# Use absolute-safe paths relative to the current script
base_dir = os.path.dirname(__file__)

xgb_path = os.path.join(base_dir, "gold_price_sentiment_model.pkl")
lstm_path = os.path.join(base_dir, "gold_lstm_model.keras")
scaler_path = os.path.join(base_dir, "gold_scaler.pkl")

# Load models safely
with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

lstm_model = load_model(lstm_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

log_step("Loading models", t0)

# =========================================
# STEP 2: FinBERT Sentiment Analysis
# =========================================
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

# =========================================
# STEP 3: Fetch Latest News
# =========================================
t2 = log_step("Fetching latest gold news")
api_key = "b30c47dafbc14719873177edf61253cc"
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

print(f"📰 {news} ({news_date})")
log_step("Fetching latest gold news", t2)

# =========================================
# STEP 4: XGBoost News-Based Prediction
# =========================================
t3 = log_step("Predicting news-based sentiment")
sent_label, sent_score = get_sentiment(news)
X_news = pd.DataFrame([{"sentiment_score": sent_score}])
xgb_pred = int(xgb_model.predict(X_news)[0])
xgb_conf = float(xgb_model.predict_proba(X_news)[0][xgb_pred])
direction_text = "UP 📈" if xgb_pred == 1 else "DOWN 📉"
log_step("Predicting news-based sentiment", t3)

# =========================================
# STEP 5: LSTM Time-Series Prediction
# =========================================
t4 = log_step("Predicting next-day gold price (LSTM)")
# Load Excel using a stable absolute path
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gold_features.xlsx")
df = pd.read_excel(os.path.abspath(data_path))
features = ["Open_scaled", "High_scaled", "Low_scaled", "Close_scaled", "Volume_scaled"]
data = df[features].values
sequence_len = 30
X_input = np.array([data[-sequence_len:]])
lstm_pred = float(lstm_model.predict(X_input)[0][0])
log_step("Predicting next-day gold price (LSTM)", t4)

# Inverse scale predicted value
scaled_df = df[features]
scaled_close = scaled_df["Close_scaled"].values.reshape(-1, 1)
# reconstruct partial scaler inverse if needed
inv_close = scaler.inverse_transform(df[features])[:, 3]
predicted_price = scaler.inverse_transform(
    np.concatenate([np.zeros((1, 4)), [[lstm_pred]]], axis=1)
)[0, 4] if lstm_pred else lstm_pred

# =========================================
# STEP 6: Combine Predictions
# =========================================
combined_direction = "UP 📈" if ((xgb_pred == 1) and (lstm_pred > df["Close_scaled"].iloc[-1])) else "DOWN 📉"
confidence = (xgb_conf + 0.5) / 2  # simple average heuristic

# =========================================
# STEP 7: SHAP Explanation (XGBoost part)
# =========================================
t5 = log_step("Explaining with SHAP")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_news)
feature_impact = shap_values.values[0][0]
reason = (
    f"Positive sentiment pushed prediction UP (+{feature_impact:.3f})"
    if feature_impact > 0 else
    f"Negative sentiment pulled prediction DOWN ({feature_impact:.3f})"
)
log_step("Explaining with SHAP", t5)

# =========================================
# STEP 8: Save Outputs
# =========================================
t6 = log_step("Saving results")

# ✅ Folder: data/predictions (auto-created)
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "predictions")
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ✅ File paths
txt_file = os.path.join(output_dir, "prediction_latest.txt")  # always overwritten
csv_file = os.path.join(output_dir, "prediction_data.csv")    # appended daily

# ✅ Write TXT report (always replaces old one)
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("=== HYBRID GOLD PRICE PREDICTION REPORT ===\n\n")
    f.write(f"📅 Date: {news_date}\n")
    f.write(f"📰 News: {news}\n\n")
    f.write(f"💬 Sentiment: {sent_label} ({sent_score:.3f})\n")
    f.write(f"🧠 XGBoost Prediction: {direction_text} (Confidence: {xgb_conf:.2f})\n")
    f.write(f"💰 LSTM Predicted Price: {predicted_price:.2f}\n")
    f.write(f"📊 Hybrid Final Prediction: {combined_direction}\n\n")
    f.write(f"📈 Explanation: {reason}\n")
    f.write(f"⏱️ Completed in {time.time() - start_time:.2f}s\n")
    f.write("="*55)

# ✅ Append results to CSV (preserve history)
if os.path.exists(csv_file):
    df_csv = pd.read_csv(csv_file)
else:
    df_csv = pd.DataFrame(columns=[
        "date", "news", "sentiment_label", "sentiment_score",
        "xgb_pred", "xgb_conf", "lstm_pred", "combined_direction"
    ])

# Avoid duplicate entries for the same date
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
    "combined_direction": combined_direction
}])

df_csv = pd.concat([df_csv, new_row], ignore_index=True)
df_csv.to_csv(csv_file, index=False)

# ✅ Log and print success
log_step("Saving results", t6)
print(f"\n✅ TXT saved (replaced): {txt_file}")
print(f"✅ CSV updated (appended): {csv_file}")
print(f"🏁 Done in {time.time() - start_time:.2f}s")
