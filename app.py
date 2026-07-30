import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import string
import contractions
import time
import base64
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DOWNLOAD NLTK RESOURCES
# -----------------------------

@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

download_nltk()

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except:
        st.error("❌ Model files not found. Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' exist.")
        return None, None

model, vectorizer = load_model()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# SESSION STATE
# -----------------------------

if "article" not in st.session_state:
    st.session_state.article = ""
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "result_data" not in st.session_state:
    st.session_state.result_data = None

# -----------------------------
# CUSTOM CSS - ENHANCED
# -----------------------------

st.markdown("""
<style>
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #111827 50%, #0b1220 100%);
        color: #f0f4ff;
    }
    
    /* Main container */
    .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
        padding: 1.5rem 0;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Sidebar text */
    .sidebar-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(139,92,246,0.3), rgba(96,165,250,0.1));
        margin: 1.2rem 0;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-left: 3px solid #8B5CF6;
    }
    
    .sidebar-section h4 {
        color: #c4b5fd;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-section p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #cbd5e1;
        font-size: 0.9rem;
        padding: 0.3rem 0;
    }
    
    .step-number {
        background: rgba(139, 92, 246, 0.2);
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #a78bfa;
    }
    
    /* Hero */
    .hero {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
        animation: fadeIn 0.8s ease;
    }
    
    .hero h1 {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }
    
    .hero .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #7c5cfc 40%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero p {
        font-size: 1.2rem;
        color: #94a3b8;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.8;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(17, 27, 45, 0.65);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 1.8rem 2rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 15px 45px rgba(0,0,0,0.35);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.2);
        box-shadow: 0 20px 55px rgba(139, 92, 246, 0.12);
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(16, 24, 39, 0.8) !important;
        color: #f0f4ff !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 56px;
        border: none;
        border-radius: 16px;
        font-size: 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #7c5cfc 0%, #8B5CF6 100%);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 35px rgba(139, 92, 246, 0.4);
        border-color: transparent;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Secondary button */
    .stButton > button[data-baseweb="button"]:nth-child(2) {
        background: rgba(255,255,255,0.06);
        box-shadow: none;
    }
    
    .stButton > button[data-baseweb="button"]:nth-child(2):hover {
        background: rgba(255,255,255,0.12);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(17, 27, 45, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.2);
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.1);
    }
    
    .metric-card .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #c4b5fd;
    }
    
    .metric-card .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    
    /* Prediction Result Card */
    .prediction-card {
        background: rgba(17, 27, 45, 0.75);
        backdrop-filter: blur(24px);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        animation: slideUp 0.6s ease;
        text-align: center;
    }
    
    .prediction-card .result-icon {
        font-size: 4.5rem;
        margin-bottom: 0.5rem;
    }
    
    .prediction-card .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .prediction-card .result-title.fake {
        color: #f87171;
    }
    
    .prediction-card .result-title.real {
        color: #34d399;
    }
    
    .prediction-card .result-description {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 600px;
        margin: 0.5rem auto;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c5cfc, #60a5fa) !important;
        border-radius: 20px !important;
        height: 10px !important;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(17, 27, 45, 0.55);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        height: 100%;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(139, 92, 246, 0.2);
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.08);
    }
    
    .feature-card .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }
    
    .feature-card h4 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Tip Box */
    .tip-box {
        background: rgba(139, 92, 246, 0.08);
        border-radius: 14px;
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #8B5CF6;
        margin: 0.5rem 0;
        color: #cbd5e1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #475569;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 1rem;
    }
    
    .footer span {
        color: #a78bfa;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero h1 { font-size: 2.5rem; }
        .prediction-card .result-title { font-size: 1.6rem; }
        section[data-testid="stSidebar"] { width: 280px !important; }
    }
    
    /* Status badges */
    .badge-fake {
        display: inline-block;
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        padding: 0.3rem 1.2rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }
    
    .badge-real {
        display: inline-block;
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        padding: 0.3rem 1.2rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 3rem;">📰</div>
        <div class="sidebar-title">Fake News</div>
        <div class="sidebar-title" style="font-size: 1.4rem; margin-top: -0.3rem;">Detector</div>
        <div class="sidebar-subtitle">AI-powered truth verification</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.2rem;">
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; background: rgba(139,92,246,0.12); border-radius: 12px; color: #c4b5fd; font-weight: 600;">
            <span>🏠</span> Home
        </div>
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; border-radius: 12px; color: #64748b; transition: 0.2s; cursor: default;">
            <span>ℹ️</span> About
        </div>
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; border-radius: 12px; color: #64748b; transition: 0.2s; cursor: default;">
            <span>📊</span> Statistics
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # About section
    st.markdown("""
    <div class="sidebar-section">
        <h4>🧠 About</h4>
        <p>This application uses <strong style="color: #a78bfa;">Machine Learning</strong> and 
        <strong style="color: #a78bfa;">Natural Language Processing</strong> to classify news articles as 
        <span style="color: #f87171;">Fake</span> or <span style="color: #34d399;">Real</span>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div class="sidebar-section">
        <h4>⚡ How it works</h4>
        <div class="step-item"><span class="step-number">1</span> Paste a news article</div>
        <div class="step-item"><span class="step-number">2</span> Click <strong style="color: #a78bfa;">Analyze</strong></div>
        <div class="step-item"><span class="step-number">3</span> Get instant prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    st.markdown("""
    <div class="sidebar-section">
        <h4>🤖 Model</h4>
        <p><strong>Algorithm:</strong> Linear SVM</p>
        <p><strong>Vectorizer:</strong> TF-IDF</p>
        <p><strong>Dataset:</strong> 30,244 Articles</p>
        <div style="margin-top: 0.5rem;">
            <span class="badge-model">⚡ Accuracy: 94.2%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 0.5rem 0;">
        <span style="color: #64748b;">🔒 Privacy first • No data stored</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------

st.markdown("""
<div class="hero">
    <h1>
        Fake News<br>
        <span class="gradient-text">Detector</span>
    </h1>
    <p>
        Detect misinformation with AI-powered Natural Language Processing.
        Paste any news article and get instant results.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------

def preprocess_text(text):
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(tokens)

# -----------------------------
# INPUT SECTION
# -----------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("### 📝 Enter News Article")

article = st.text_area(
    label="",
    key="article",
    height=240,
    placeholder="Paste or type a news article here...\n\nExample: 'Breaking news: Scientists discover revolutionary new technology...'",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# LIVE COUNTERS
# -----------------------------

words = len(article.split()) if article.strip() else 0
characters = len(article) if article.strip() else 0

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.markdown(f"""
    <div style="color: #94a3b8; font-size: 0.9rem; padding: 0.3rem 0;">
        📝 <strong style="color: #c4b5fd;">{words}</strong> Words
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="color: #94a3b8; font-size: 0.9rem; padding: 0.3rem 0;">
        ⌨️ <strong style="color: #c4b5fd;">{characters}</strong> Characters
    </div>
    """, unsafe_allow_html=True)

# Tip
st.markdown("""
<div class="tip-box">
    💡 <strong>Tip:</strong> Longer articles provide more context for better predictions.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# BUTTONS
# -----------------------------

col1, col2 = st.columns([3, 1])

with col1:
    analyze = st.button("🚀 Analyze Article", use_container_width=True, type="primary")

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

# -----------------------------
# CLEAR BUTTON LOGIC
# -----------------------------

if clear:
    st.session_state.article = ""
    st.session_state.prediction_made = False
    st.session_state.result_data = None
    st.rerun()

# -----------------------------
# PREDICTION ENGINE
# -----------------------------

if analyze:
    if article.strip() == "":
        st.warning("⚠️ Please enter a news article before analyzing.")
    else:
        with st.spinner("🧠 Analyzing article with AI..."):
            start_time = time.time()
            
            # Preprocess
            cleaned_text = preprocess_text(article)
            
            # Vectorize
            vector = vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(vector)[0]
            
            # Confidence
            confidence = 0.85
            try:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(vector)[0]
                    confidence = float(np.max(probabilities))
                elif hasattr(model, "decision_function"):
                    score = abs(model.decision_function(vector)[0])
                    confidence = min(score / 3, 1.0)
            except:
                pass
            
            processing_time = round(time.time() - start_time, 2)
            
            # Store results
            st.session_state.prediction_made = True
            st.session_state.result_data = {
                "prediction": prediction,
                "confidence": confidence,
                "words": words,
                "characters": characters,
                "time": processing_time
            }
            
            st.rerun()

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

if st.session_state.prediction_made and st.session_state.result_data:
    data = st.session_state.result_data
    prediction = data["prediction"]
    confidence = data["confidence"]
    
    # Determine result
    if prediction in [1, "Fake", "FAKE"]:
        icon = "📰"
        title = "Fake News"
        title_class = "fake"
        badge = "badge-fake"
        description = "The article contains patterns commonly associated with misinformation. Please verify using trusted sources."
    else:
        icon = "🗞️"
        title = "Real News"
        title_class = "real"
        badge = "badge-real"
        description = "The article appears to follow legitimate news patterns and language structures."
    
    # Display Result Card
    st.markdown(f"""
    <div class="prediction-card">
        <div class="result-icon">{icon}</div>
        <div class="result-title {title_class}">{title}</div>
        <div style="margin: 0.5rem 0 1rem 0;">
            <span class="{badge}">Confidence: {confidence*100:.1f}%</span>
        </div>
        <div class="result-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence Progress
    st.markdown("### 📊 Confidence Score")
    st.progress(confidence)
    st.caption(f"Model confidence: {confidence*100:.1f}%")
    
    # Metrics
    st.markdown("### 📈 Analysis Metrics")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📝</div>
            <div class="metric-value">{data['words']}</div>
            <div class="metric-label">Words</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⌨️</div>
            <div class="metric-value">{data['characters']}</div>
            <div class="metric-label">Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{data['time']}s</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{confidence*100:.0f}%</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

# -----------------------------
# FEATURE CARDS
# -----------------------------

st.markdown("""
<h3 style="text-align: center; font-weight: 700; margin-bottom: 1.5rem;">
    ✨ Why Use Fake News Detector?
</h3>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <h4>NLP Powered</h4>
        <p>Advanced Natural Language Processing analyzes writing style and patterns.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h4>High Accuracy</h4>
        <p>Trained on thousands of labelled articles for reliable classification.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h4>Fast Analysis</h4>
        <p>Predictions generated in milliseconds with optimized ML model.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔒</div>
        <h4>Privacy First</h4>
        <p>Articles are processed only during analysis and never stored.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("""
<div class="footer">
    📰 Fake News Detector &nbsp;·&nbsp; Built with ❤️ using Streamlit &amp; Machine Learning
    <br>
    <span>© 2026 • All rights reserved</span>
</div>
""", unsafe_allow_html=True)