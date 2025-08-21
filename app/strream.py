import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import string

# Initial setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
st.set_page_config(page_title="Climate Change Modeling", layout="wide")

# Load datasets
climate_data = pd.read_csv("../data/global.csv").dropna(subset=["Total"]).fillna(method="ffill")
comment_data = pd.read_csv("../data/climate_nasa.csv").dropna(subset=["text"])

# Feature engineering for climate data
def engineer_features(df):
    df = df.copy()
    df['Total Fuel'] = df[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
    df['Gas Fuel Ratio'] = df['Gas Fuel'] / df['Total Fuel']
    df['Decade'] = (df['Year'] // 10) * 10
    return df

climate_data = engineer_features(climate_data)
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring', 'Total Fuel', 'Gas Fuel Ratio', 'Decade']
X = climate_data[features]
y = climate_data['Total']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression().fit(X_scaled, y)

# Word cleaning for NLP
def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

comment_data['clean_text'] = comment_data['text'].apply(clean_text)

# Sentiment classification
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

comment_data['Sentiment'] = comment_data['clean_text'].apply(get_sentiment)
sentiment_counts = comment_data['Sentiment'].value_counts()

# Streamlit App UI
st.title("🌍 Climate Change Modeling & Sentiment App")
tab1, tab2, tab3, tab4 = st.tabs(["📈 CO₂ Prediction", "🧪 Scenario Simulator", "💬 Sentiment Analysis", "☁️ Word Cloud"])

# ----------------------------- TAB 1 -----------------------------
with tab1:
    st.header("📈 Predict Global CO₂ Emissions")
    st.write("Enter projected values for the year:")

    year = st.number_input("Year", min_value=2020, max_value=2100, value=2025)
    gas = st.number_input("Gas Fuel", value=1700)
    liquid = st.number_input("Liquid Fuel", value=2800)
    solid = st.number_input("Solid Fuel", value=4000)
    cement = st.number_input("Cement", value=500)
    flaring = st.number_input("Gas Flaring", value=90)

    df_input = pd.DataFrame({
        'Year': [year],
        'Gas Fuel': [gas],
        'Liquid Fuel': [liquid],
        'Solid Fuel': [solid],
        'Cement': [cement],
        'Gas Flaring': [flaring]
    })
    df_input = engineer_features(df_input)
    X_input = scaler.transform(df_input[features])
    pred = model.predict(X_input)[0]
    st.success(f"🔮 Predicted Global CO₂ Emissions in {year}: **{pred:.2f} million tonnes**")

# ----------------------------- TAB 2 -----------------------------
with tab2:
    st.header("🧪 Scenario Simulation")
    scenarios = {
        '🔵 Base Scenario': [1700, 2800, 4000, 500, 90],
        '🔺 Gas +20%': [1700*1.2, 2800, 4000, 500, 90],
        '🔻 Cement -50%': [1700, 2800, 4000, 500*0.5, 90],
        '⚡ All Fuels +10%': [1700*1.1, 2800*1.1, 4000*1.1, 500*1.1, 90*1.1],
        '🍃 Green Policy -25%': [1700*0.75, 2800*0.75, 4000*0.75, 500*0.75, 90*0.75],
    }
    results = []
    for label, vals in scenarios.items():
        df_s = pd.DataFrame({
            'Year': [2025],
            'Gas Fuel': [vals[0]],
            'Liquid Fuel': [vals[1]],
            'Solid Fuel': [vals[2]],
            'Cement': [vals[3]],
            'Gas Flaring': [vals[4]]
        })
        df_s = engineer_features(df_s)
        X_s = scaler.transform(df_s[features])
        pred = model.predict(X_s)[0]
        results.append((label, pred))

    df_results = pd.DataFrame(results, columns=["Scenario", "Predicted CO₂"])
    st.dataframe(df_results)

    st.bar_chart(data=df_results.set_index("Scenario"))

# ----------------------------- TAB 3 -----------------------------
with tab3:
    st.header("💬 Sentiment Analysis on NASA Comments")
    st.write("Below is the breakdown of public sentiment:")

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(sentiment_counts)
    with col2:
        plt.figure(figsize=(4, 4))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=["green", "gray", "red"])
        st.pyplot(plt)

# ----------------------------- TAB 4 -----------------------------
with tab4:
    st.header("☁️ Word Cloud from Comments")
    all_words = " ".join(comment_data['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
