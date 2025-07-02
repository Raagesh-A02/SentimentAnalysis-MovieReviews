# app/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os, sys

# Add root to path so utils can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.text_preprocessing import preprocess_text
from utils.sentiment_predictor import get_sentiment


# Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_reviews.csv")
    df['vader_sentiment'] = df['cleaned_review'].apply(get_sentiment)
    return df

# Generate WordCloud
def show_wordcloud(text_data):
    text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Main UI
def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    st.title("Sentiment Analysis on Movie Reviews")

    df = load_data()

    tab1, tab2, tab3 = st.tabs([" Distribution", " WordCloud", " Predict"])

    # Tab 1: Sentiment Distribution
    with tab1:
        st.subheader("Sentiment Distribution (Rule-based NLP)")
        count_data = df['vader_sentiment'].value_counts().rename({0: 'Negative', 1: 'Positive'})
        fig = px.pie(values=count_data.values, names=count_data.index, title="Sentiment Pie Chart", color_discrete_sequence=['red', 'green'])
        st.plotly_chart(fig)

    # Tab 2: WordCloud
    with tab2:
        st.subheader("Top Words in Reviews")
        show_wordcloud(df['cleaned_review'])

    # Tab 3: Predict Sentiment
    with tab3:
        st.subheader("Test Sentiment on Your Review")
        user_input = st.text_area("Enter a movie review:")
        if user_input:
            cleaned = preprocess_text(user_input)
            pred = get_sentiment(cleaned)
            label = "Positive " if pred == 1 else "Negative "
            st.markdown(f"### Sentiment: {label}")

if __name__ == "__main__":
    main()
