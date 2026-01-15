import streamlit as st
import pandas as pd
import os
import time

DATA_PATH = "../data/daily_predictions.csv"

st.set_page_config(page_title="Intraday Stock Predictor", layout="centered")

st.title("📈 NIFTY50 Intraday Top Gainers Prediction")

st.write("XGBoost model + Technical Indicators + News Sentiment")

# Refresh button
if st.button("Refresh Predictions"):
    os.system("python ../src/predict.py")
    time.sleep(2)

# Load predictions
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    df['Up_Probability'] = (df['Up_Probability']*100).round(2)

    st.subheader("Top 15 Stocks Likely to Rise Today")
    st.dataframe(df, use_container_width=True)

else:
    st.warning("Run prediction first: python src/predict.py")
