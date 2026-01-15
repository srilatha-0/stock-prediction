import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------
# Paths
# -----------------------
DATA_DIR = "../data"
INPUT_NEWS_CSV = os.path.join(DATA_DIR, "news_headlines.csv")  # your file
OUTPUT_SENTIMENT_CSV = os.path.join(DATA_DIR, "news_sentiment.csv")

# -----------------------
# Load news data
# -----------------------
if not os.path.exists(INPUT_NEWS_CSV):
    raise FileNotFoundError(f"News CSV not found: {INPUT_NEWS_CSV}")

df_news = pd.read_csv(INPUT_NEWS_CSV)

# Strip spaces from column names
df_news.columns = df_news.columns.str.strip()

# Make sure required columns exist
if 'Date' not in df_news.columns or 'Headline' not in df_news.columns:
    raise ValueError("CSV must have 'Date' and 'Headline' columns")

# Convert Date to datetime
df_news['Date'] = pd.to_datetime(df_news['Date'], dayfirst=True)

# Drop rows with empty headlines
df_news = df_news.dropna(subset=['Headline']).reset_index(drop=True)

# -----------------------
# Load FinBERT
# -----------------------
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# -----------------------
# Compute sentiment
# -----------------------
def get_sentiment(text):
    try:
        result = finbert(text[:512])  # truncate to 512 tokens
        return result[0]['label']  # POSITIVE, NEGATIVE, NEUTRAL
    except:
        return 'NEUTRAL'

print("Computing sentiment for news headlines...")
df_news['Sentiment'] = df_news['Headline'].apply(get_sentiment)

# -----------------------
# Save sentiment CSV
# -----------------------
df_news.to_csv(OUTPUT_SENTIMENT_CSV, index=False)
print(f"✅ News sentiment saved: {OUTPUT_SENTIMENT_CSV}")
print(f"Columns: {df_news.columns.tolist()}")
print(f"Rows: {len(df_news)}")
