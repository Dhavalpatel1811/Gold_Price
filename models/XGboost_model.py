# train_model.py
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load historical gold news data with sentiment scores
df = pd.read_csv("../data/Gold News Data.csv")

# Ensure 'sentiment_score' and 'Price_Direction' exist
df['Price_Direction'] = (df['Next_Day_Close'] > df['Close']).astype(int)

# Features and target
X = df[['sentiment_score']]
y = df['Price_Direction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save the model
with open("gold_price_sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ gold_price_sentiment_model.pkl created successfully!")
