# 🔍 Backlink Spam Classifier

A Streamlit web app to detect spammy backlinks using a trained machine learning model.

## 📦 How to Use

1. Upload a CSV file containing a list of URLs (first column only).
2. The app will extract features from the URLs.
3. It will classify them using a pre-trained model (`spam_model.pkl`).
4. Download the results with `Spam` / `Not Spam` labels.

## 🧠 Model Used

- RandomForestClassifier (Scikit-learn)
- Feature engineered from domain and URL structure

## 📂 Files in this Repo

- `app.py` — Streamlit app entry point
- `requirements.txt` — Python dependencies
- `spam_model.pkl` — Trained model (upload yours)
- `sample_input.csv` — Sample URL list to test

## 🚀 Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub and select this repo
4. Use `app.py` as entry file