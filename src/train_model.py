import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
DATA_DIR = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

STOCK_DATA = os.path.join(DATA_DIR, "stocks_all_preprocessed.csv")
SENTIMENT_DATA = os.path.join(DATA_DIR, "news_sentiment.csv")

# Load stock indicator data
df_stock = pd.read_csv(STOCK_DATA)

# Load sentiment data
df_sent = pd.read_csv(SENTIMENT_DATA)

# Convert Date columns
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_sent['Date'] = pd.to_datetime(df_sent['Date'])

# Merge sentiment with stock data
df_merged = pd.merge(df_stock, df_sent[['Date','Stock','Sentiment']],
                     left_on=['Date','Ticker'],
                     right_on=['Date','Stock'],
                     how='left')

# Fill missing sentiment with neutral (0)
df_merged['Sentiment'] = df_merged['Sentiment'].fillna(0)

# --- Create Intraday Target ---
# Trend = 1 if next candle close > current close else 0
df_merged['NextClose'] = df_merged.groupby('Ticker')['Close'].shift(-1)
df_merged['Trend'] = (df_merged['NextClose'] > df_merged['Close']).astype(int)

# Drop last row of each stock where NextClose is NaN
df_merged = df_merged.dropna(subset=['NextClose'])

# --- Features ---
feature_cols = [
    'Open','High','Low','Close','Volume',
    'MA10','MA20','EMA10','EMA20','RSI','MACD',
    'Sentiment'
]

X = df_merged[feature_cols]
y = df_merged['Trend']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# --- XGBoost Model ---
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "intraday_xgboost.pkl"))
print("Model saved to models/intraday_xgboost.pkl")
