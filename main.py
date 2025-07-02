# main.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from utils.text_preprocessing import preprocess_text

def main():
    # Load cleaned dataset
    print("Loading dataset...")
    df = pd.read_csv("data/MovieReview-Dataset.csv")

    # Clean reviews (again in case you're testing fresh)
    print("Preprocessing reviews...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # Save cleaned reviews (optional)
    df.to_csv("data/cleaned_reviews.csv", index=False)
    print("Cleaned data saved to data/cleaned_reviews.csv")

    # TF-IDF Vectorization
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_review'])
    y = df['sentiment']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("Model and vectorizer saved to models/")

if __name__ == "__main__":
    main()
