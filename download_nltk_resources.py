# download_nltk_resources.py
import nltk

required = ["punkt", "punkt_tab", "stopwords", "wordnet"]

for res in required:
    nltk.download(res)

print("✅ NLTK resources downloaded successfully.")
