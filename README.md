
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbinTAbey/TwitterSentimentAnalysisNLP/blob/main/TwitterSentimentAnalysis.ipynb)

This project performs sentiment analysis on Twitter data using natural language processing (NLP) techniques and machine learning models. It classifies tweets as positive or negative using the Sentiment140 dataset.

---

## 🔍 Project Overview

- Dataset: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Goal: Classify tweets based on sentiment
  Tech Stack: Python, Pandas, scikit-learn, NLTK, TfidfVectorizer

---

## 📊 Workflow

1. Data Loading
   Load the Sentiment140 dataset from CSV.

2. Text Preprocessing
   - Lowercasing  
   - Tokenization  
   - Stopword removal  
   - Stemming (PorterStemmer)

3. Feature Extraction
   - TF-IDF Vectorization

4. Model Building
   - Logistic Regression / Naive Bayes  
   - Train-Test Split  
   - Accuracy Evaluation

---

## 📂 File Structure
TwitterSentimentAnalysisNLP
# TwitterSentimentAnalysis.ipynb # Colab notebook
# training.1600000.processed.noemoticon.csv # Dataset (Twitter sentiment analysis via Kaggle)

---

## 🚀 How to Run

1. Click the badge above to open in Google Colab.  
2. Upload your `kaggle.json` file if loading dataset via API.  
3. Run all cells.

---

## 📌 Acknowledgements

- Tutorial followed from [GeeksforGeeks](https://www.geeksforgeeks.org/)
- Dataset by [@kazanova](https://www.kaggle.com/datasets/kazanova)

---

## 🧠 Future Improvements

- Add sentiment visualization (word clouds, pie charts)  
- Deploy model via Flask or Streamlit  
- Add multiclass sentiment labels (neutral)

---

## 🤝 Connect

Made with ❤️ by [Abin Abey](https://github.com/AbinTAbey)

