import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import string
import contractions
import time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ============================================================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NLTK RESOURCES
# ============================================================================

@st.cache_resource
def download_nltk():
    """Download required NLTK data packages."""
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

download_nltk()

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error(
            "❌ Model files not found. Please ensure 'fake_news_model.pkl' "
            "and 'tfidf_vectorizer.pkl' exist in the application directory."
        )
        return None, None

model, vectorizer = load_model()

# ============================================================================
# CONSTANTS
# ============================================================================

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ============================================================================
# SESSION STATE
# ============================================================================

if "article" not in st.session_state:
    st.session_state.article = ""
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "result_data" not in st.session_state:
    st.session_state.result_data = None

# ============================================================================
# SIDEBAR - NOW VISIBLE
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 3rem;">📰</div>
        <h1 style="font-size: 1.8rem; font-weight: 800; color: #f0f4ff; margin-bottom: 0.2rem;">
            Fake News
        </h1>
        <h2 style="font-size: 1.2rem; font-weight: 400; color: #94a3b8; margin-top: -0.3rem;">
            Detector
        </h2>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">
            AI-powered Fake News Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; 
                    background: rgba(139, 92, 246, 0.12); border-radius: 10px; 
                    color: #c4b5fd; font-weight: 600;">
            <span>🏠</span> Home
        </div>
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; 
                    border-radius: 10px; color: #94a3b8; transition: 0.2s; cursor: default;
                    margin-top: 0.2rem;">
            <span>ℹ️</span> About
        </div>
        <div style="display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; 
                    border-radius: 10px; color: #94a3b8; transition: 0.2s; cursor: default;
                    margin-top: 0.2rem;">
            <span>📊</span> Model Information
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About section
    st.markdown("""
    <div style="margin: 1rem 0;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; 
                    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem;">
            🧠 About
        </div>
        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; 
                    background: rgba(255, 255, 255, 0.02); border-radius: 12px; 
                    padding: 0.8rem 1rem; border-left: 2px solid rgba(139, 92, 246, 0.3);">
            This application uses <strong style="color: #a78bfa;">Machine Learning</strong> and 
            <strong style="color: #a78bfa;">Natural Language Processing</strong> to classify 
            news articles as <span style="color: #f87171;">Fake</span> or 
            <span style="color: #34d399;">Real</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div style="margin: 1rem 0;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; 
                    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem;">
            ⚡ How It Works
        </div>
        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; 
                    background: rgba(255, 255, 255, 0.02); border-radius: 12px; 
                    padding: 0.8rem 1rem; border-left: 2px solid rgba(139, 92, 246, 0.3);">
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.2rem 0;">
                <span style="display: inline-flex; align-items: center; justify-content: center; 
                            background: rgba(139, 92, 246, 0.15); border-radius: 50%; 
                            width: 22px; height: 22px; font-size: 0.65rem; font-weight: 700; 
                            color: #a78bfa; flex-shrink: 0;">1</span>
                Paste a news article
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.2rem 0;">
                <span style="display: inline-flex; align-items: center; justify-content: center; 
                            background: rgba(139, 92, 246, 0.15); border-radius: 50%; 
                            width: 22px; height: 22px; font-size: 0.65rem; font-weight: 700; 
                            color: #a78bfa; flex-shrink: 0;">2</span>
                Click <strong style="color: #a78bfa;">Analyze</strong>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.2rem 0;">
                <span style="display: inline-flex; align-items: center; justify-content: center; 
                            background: rgba(139, 92, 246, 0.15); border-radius: 50%; 
                            width: 22px; height: 22px; font-size: 0.65rem; font-weight: 700; 
                            color: #a78bfa; flex-shrink: 0;">3</span>
                View instant prediction
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model information
    st.markdown("""
    <div style="margin: 1rem 0;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; 
                    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem;">
            🤖 Model Information
        </div>
        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; 
                    background: rgba(255, 255, 255, 0.02); border-radius: 12px; 
                    padding: 0.8rem 1rem; border-left: 2px solid rgba(139, 92, 246, 0.3);">
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: #64748b;">Algorithm</span>
                <span style="color: #c4b5fd; font-weight: 500;">Logistic Regression</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: #64748b;">Vectorizer</span>
                <span style="color: #c4b5fd; font-weight: 500;">TF-IDF</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: #64748b;">Dataset</span>
                <span style="color: #c4b5fd; font-weight: 500;">30,244 Articles</span>
            </div>
            <div style="margin-top: 0.5rem; text-align: center;">
                <span style="background: rgba(139,92,246,0.12); color: #a78bfa; 
                             padding: 0.2rem 1rem; border-radius: 40px; font-size: 0.8rem;
                             border: 1px solid rgba(139,92,246,0.15);">
                    ⚡ Accuracy: 94.2%
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 0.5rem 0;">
        <span style="color: #64748b;">🔒 Articles are analyzed locally and are not stored.</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #111827 50%, #0b1220 100%);
        color: #f0f4ff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.12);
        padding: 1.5rem 1rem;
        min-width: 280px !important;
    }
    
    /* Main container */
    .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Hero - WITH PURPLE GRADIENT */
    .hero {
        text-align: center;
        padding: 0.5rem 0 2rem 0;
        animation: fadeIn 0.8s ease;
    }
    
    .hero h1 {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -2px;
        margin-bottom: 0.25rem;
        line-height: 1.1;
        color: #f0f4ff;
    }
    
    .hero .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #7c5cfc 40%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero p {
        font-size: 1.15rem;
        color: #94a3b8;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.8;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(17, 27, 45, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.35);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.15);
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(16, 24, 39, 0.8) !important;
        color: #f0f4ff !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 52px;
        border: none;
        border-radius: 14px;
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
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        box-shadow: none;
        color: #94a3b8;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        box-shadow: none;
        transform: translateY(-2px);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: rgba(17, 27, 45, 0.75);
        backdrop-filter: blur(24px);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        animation: slideUp 0.6s ease;
        text-align: center;
    }
    
    .prediction-card .result-icon {
        font-size: 4.5rem;
        margin-bottom: 0.25rem;
    }
    
    .prediction-card .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.25rem 0;
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
        margin: 0.5rem auto 0;
    }
    
    /* Badges */
    .badge-fake {
        display: inline-block;
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
        padding: 0.3rem 1.2rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(248, 113, 113, 0.15);
    }
    
    .badge-real {
        display: inline-block;
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        padding: 0.3rem 1.2rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 211, 153, 0.15);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c5cfc, #60a5fa) !important;
        border-radius: 20px !important;
        height: 8px !important;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(17, 27, 45, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.04);
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
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.06);
    }
    
    .feature-card .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }
    
    .feature-card h4 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    
    .feature-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Tip Box */
    .tip-box {
        background: rgba(139, 92, 246, 0.06);
        border-radius: 14px;
        padding: 0.7rem 1.2rem;
        border-left: 4px solid #8B5CF6;
        margin: 0.5rem 0;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    
    .tip-box strong {
        color: #c4b5fd;
    }
    
    /* Counter */
    .counter {
        color: #94a3b8;
        font-size: 0.9rem;
        padding: 0.3rem 0;
    }
    
    .counter strong {
        color: #c4b5fd;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        color: #475569;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 1.5rem;
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
        .prediction-card { padding: 1.5rem 1rem; }
        .glass-card { padding: 1.5rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text: str) -> str:
    """Clean and preprocess input text for model prediction."""
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
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in STOP_WORDS
    ]
    
    return " ".join(tokens)

# ============================================================================
# HERO - WITH PURPLE GRADIENT
# ============================================================================

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

# ============================================================================
# INPUT SECTION
# ============================================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("### 📝 Enter News Article")

article = st.text_area(
    label="",
    key="article",
    height=240,
    placeholder=(
        "Paste or type a news article here...\n\n"
        "Example: 'Breaking news: Scientists discover revolutionary new technology...'"
    ),
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# COUNTERS & TIP
# ============================================================================

words = len(article.split()) if article.strip() else 0
characters = len(article) if article.strip() else 0

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="counter">
        📝 <strong>{words}</strong> Words
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="counter">
        ⌨️ <strong>{characters}</strong> Characters
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="tip-box">
    💡 <strong>Tip:</strong> Longer articles provide more context for better predictions.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# ACTION BUTTONS
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    analyze = st.button("🚀 Analyze Article", use_container_width=True, type="primary")

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

# ============================================================================
# CLEAR BUTTON LOGIC
# ============================================================================

if clear:
    st.session_state.article = ""
    st.session_state.prediction_made = False
    st.session_state.result_data = None
    st.rerun()

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

if analyze:
    if article.strip() == "":
        st.warning("⚠️ Please enter a news article before analyzing.")
    
    elif model is None or vectorizer is None:
        st.error("❌ Model not loaded. Please check the model files.")
    
    else:
        with st.spinner("🧠 Analyzing article with AI..."):
            start_time = time.time()
            
            # Preprocess
            cleaned_text = preprocess_text(article)
            
            # Vectorize
            vector = vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(vector)[0]
            
            # Calculate confidence
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

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.prediction_made and st.session_state.result_data:
    data = st.session_state.result_data
    prediction = data["prediction"]
    confidence = data["confidence"]
    
    # Map prediction to display values
    # 0 = Fake, 1 = Real
    if prediction == 0:
        icon = "📰"
        title = "Fake News"
        title_class = "fake"
        badge_class = "badge-fake"
        description = (
            "The article contains patterns commonly associated with misinformation. "
            "Please verify the information using trusted news sources."
        )
    else:
        icon = "🗞️"
        title = "Real News"
        title_class = "real"
        badge_class = "badge-real"
        description = (
            "The article appears to follow credible news patterns based on the "
            "machine learning model's analysis."
        )
    
    # Prediction Card
    st.markdown(f"""
    <div class="prediction-card">
        <div class="result-icon">{icon}</div>
        <div class="result-title {title_class}">{title}</div>
        <div style="margin: 12px 0 16px 0;">
            <span class="{badge_class}">Confidence: {confidence*100:.1f}%</span>
        </div>
        <div class="result-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence Meter
    st.markdown("### 📊 Confidence Score")
    st.progress(confidence)
    
    # Confidence level label
    if confidence >= 0.90:
        st.success("✅ Very High Confidence")
    elif confidence >= 0.70:
        st.info("📈 High Confidence")
    elif confidence >= 0.50:
        st.warning("📊 Moderate Confidence")
    else:
        st.error("⚠️ Low Confidence")
    
    # Analysis Metrics
    st.markdown("### 📈 Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Words", data["words"])
    
    with col2:
        st.metric("⌨️ Characters", data["characters"])
    
    with col3:
        st.metric("⚡ Processing Time", f"{data['time']}s")
    
    with col4:
        st.metric("🎯 Confidence", f"{confidence*100:.1f}%")
    
    st.divider()

# ============================================================================
# FEATURE CARDS
# ============================================================================

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
        <p>Advanced Natural Language Processing analyzes writing style and linguistic patterns.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h4>High Accuracy</h4>
        <p>Trained on thousands of labelled articles for reliable and consistent classification.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h4>Fast Analysis</h4>
        <p>Predictions generated in milliseconds using an optimized machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔒</div>
        <h4>Privacy First</h4>
        <p>Articles are processed only during analysis and are never permanently stored.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    📰 Fake News Detector &nbsp;·&nbsp; Built with ❤️ using Streamlit &amp; Machine Learning
    <br>
    <span>© 2026 • All rights reserved</span>
</div>
""", unsafe_allow_html=True)