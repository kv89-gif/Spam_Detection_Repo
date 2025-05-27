
import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
import re

st.title("🔎 Backlink Spam Classifier")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

model = load_model()

# Feature extraction
def extract_features_df(df):
    def normalize_url(url):
        return f"https://{url}" if not str(url).startswith(('http://', 'https://')) else url

    def is_ip(domain): return 1 if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', domain) else 0
    def domain_length(domain): return len(domain)
    def contains_numbers(domain): return 1 if re.search(r'\d', domain) else 0
    def get_tld(domain): parts = domain.split('.'); return parts[-1] if len(parts) > 1 else 'unknown'
    def path_depth(url): return urlparse(url).path.count('/')
    def has_keywords(url): return 1 if any(k in url.lower() for k in ['free', 'casino', 'loan', 'bonus']) else 0
    def cyrillic_in_url(url): return 1 if re.search(r'[\u0400-\u04FF]', url) else 0

    df['url'] = df.iloc[:, 0].apply(normalize_url)
    df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
    df['is_ip'] = df['domain'].apply(is_ip)
    df['domain_length'] = df['domain'].apply(domain_length)
    df['contains_numbers'] = df['domain'].apply(contains_numbers)
    df['tld'] = df['domain'].apply(get_tld)
    df['path_depth'] = df['url'].apply(path_depth)
    df['has_keywords'] = df['url'].apply(has_keywords)
    df['cyrillic_in_url'] = df['url'].apply(cyrillic_in_url)

    df = pd.get_dummies(df, columns=['tld'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    return df[model.feature_names_in_]

uploaded_file = st.file_uploader("📤 Upload CSV of URLs", type=["csv"])

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")

        feature_df = extract_features_df(raw_df.copy())
        predictions = model.predict(feature_df)

        raw_df['prediction'] = ["Spam" if pred == 1 else "Not Spam" for pred in predictions]
        st.dataframe(raw_df)

        csv_out = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv_out, "classified_urls.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
