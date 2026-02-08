import streamlit as st
import torch
import pandas as pd
import requests
import re
from transformers import pipeline
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(page_title="AI Product Review Analyzer", layout="wide")
st.title("🛒 AI Product Review Analyzer")

# ==================================================
# LOAD PRE-TRAINED MODEL
# ==================================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

sentiment_pipe = load_model()

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_platform_name(url):
    domain = urlparse(url).netloc.replace("www.", "")
    return domain.split(".")[0].capitalize()

def extract_product_name(soup):
    h1 = soup.find("h1")
    if h1 and len(h1.get_text(strip=True)) < 120:
        return h1.get_text(strip=True)

    if soup.title:
        title = soup.title.string
        title = re.split(r"[-|]", title)[0]
        return title.strip()

    return "Unknown Product"

def is_genuine_review(text):
    text_lower = text.lower()

    noise_patterns = [
        r"₹\d+", r"\$\d+", r"\d+% off", r"offer", r"deal",
        r"buy at", r"price", r"mrp", r"discount",
        r"delivery", r"fulfilled", r"location",
        r"road", r"village", r"return", r"replacement",
        r"login", r"signup", r"policy", r"terms"
    ]

    for pattern in noise_patterns:
        if re.search(pattern, text_lower):
            return False

    review_keywords = [
        "good", "bad", "excellent", "poor", "quality",
        "comfortable", "worth", "experience", "problem",
        "issue", "fit", "using", "recommend", "love", "hate"
    ]

    return any(word in text_lower for word in review_keywords) and 50 < len(text) < 500

def scrape_reviews_and_product(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    product_name = extract_product_name(soup)

    blocks = soup.find_all(["p", "div", "span"])
    reviews = []

    for block in blocks:
        text = block.get_text(strip=True)
        if is_genuine_review(text):
            reviews.append(text)

    return list(set(reviews))[:50], product_name

def score_and_verdict(pos_percent):
    if pos_percent >= 80:
        return 10, "✅ Strongly Recommended to Buy"
    elif pos_percent >= 60:
        return 8, "👍 Recommended to Buy"
    elif pos_percent >= 40:
        return 6, "⚠️ Buy with Caution"
    else:
        return 3, "❌ Not Recommended to Buy"

# ==================================================
# INPUT SELECTION
# ==================================================
st.subheader("📥 Select Review Input Method")

input_method = st.radio(
    "Choose input type:",
    ["Paste Product URL", "Upload Review CSV"],
    horizontal=True
)

reviews = []
platform = "CSV Dataset"
product_name = "Uploaded Product Reviews"

# ==================================================
# URL INPUT
# ==================================================
if input_method == "Paste Product URL":
    product_url = st.text_input("Enter product URL")

    if product_url:
        try:
            platform = get_platform_name(product_url)
            with st.spinner("🌐 Extracting product details & reviews..."):
                reviews, product_name = scrape_reviews_and_product(product_url)

            if not reviews:
                st.error("❌ No genuine customer reviews found.")

        except Exception as e:
            st.error(f"❌ Error fetching page: {e}")

# ==================================================
# CSV INPUT
# ==================================================
else:
    uploaded_file = st.file_uploader(
        "Upload CSV file (required column: 'text', optional: 'product')",
        type="csv"
    )

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        df_input.columns = df_input.columns.str.strip().str.lower()

        if "text" not in df_input.columns:
            st.error("❌ CSV must contain a column named 'text'")
        else:
            reviews = df_input["text"].dropna().astype(str).tolist()

            if "product" in df_input.columns:
                product_name = df_input["product"].dropna().iloc[0]

# ==================================================
# SENTIMENT ANALYSIS
# ==================================================
if reviews:
    st.divider()
    st.subheader("🤖 Sentiment Analysis")

    sample_size = min(len(reviews), 50)

    with st.spinner("Analyzing sentiment..."):
        results = sentiment_pipe(
            reviews[:sample_size],
            batch_size=16,
            truncation=True
        )

    analysis_data = []

    for review, result in zip(reviews[:sample_size], results):
        confidence = round(result["score"] * 100, 2)

        if confidence < 65:
            sentiment = "NEUTRAL"
        else:
            sentiment = result["label"]

        analysis_data.append({
            "Review": review,
            "Sentiment": sentiment,
            "Confidence (%)": confidence
        })

    df_results = pd.DataFrame(analysis_data)

    # ==================================================
    # METRICS
    # ==================================================
    sentiment_counts = Counter(df_results["Sentiment"])
    pos_count = sentiment_counts.get("POSITIVE", 0)
    neg_count = sentiment_counts.get("NEGATIVE", 0)
    neu_count = sentiment_counts.get("NEUTRAL", 0)

    total = len(df_results)
    pos_percent = (pos_count / total) * 100
    score, verdict = score_and_verdict(pos_percent)

    # ==================================================
    # DASHBOARD SUMMARY
    # ==================================================
    st.subheader("📦 Product Overview")
    st.markdown(f"### **{product_name}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Platform", platform)
    c2.metric("Positive", pos_count)
    c3.metric("Negative", neg_count)
    c4.metric("Neutral", neu_count)
    c5.metric("Score / 10", score)

    st.subheader("🛒 Purchase Recommendation")
    st.markdown(f"### {verdict}")

    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(sentiment_counts)

    # ==================================================
    # TABLE VIEW (BOTTOM)
    # ==================================================
    st.divider()
    st.subheader("📝 Classified Customer Reviews")

    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True
    )



        #python -m streamlit run app.py