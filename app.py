import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import os
from groq import Groq

# Load assets
@st.cache_resource
def load_models():
    clf = joblib.load('category_classifier.pkl')
    reg = joblib.load('price_regressor.pkl')
    cls_feats = joblib.load('cls_features.pkl')
    reg_feats = joblib.load('reg_features.pkl')
    market_stats = joblib.load('market_stats.pkl')
    return clf, reg, cls_feats, reg_feats, market_stats

def get_explanation(features, price, category, stats):
    # Prepare context for LLM
    median = stats['median_price']
    avg_ppsqft = stats['avg_price_sqft']
    
    prompt = f"""
    Act as a real estate analyst.
    Data:
    - Predicted: ${price:,.0f} ({category})
    - Props: {features}
    - Market: Median ${median:,.0f}, Avg $/sqft ${avg_ppsqft:.0f}
    
    Output strictly:
    1. Feature-based reason (1 sentence).
    2. Market context (1 sentence).
    3. Verdict (Realistic/High/Low).
    """

    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key: return "⚠️ Groq API Key missing."
        
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Insight unavailable: {str(e)}"

# Main App
st.set_page_config(page_title="Real Estate AI", layout="wide")

try:
    clf, reg, cls_cols, reg_cols, stats = load_models()
except:
    st.error("Models not found. Please train models first using the notebook.")
    st.stop()

st.title("Real Estate Valuation AI 🏡")

# Sidebar - Strict Inputs Only
st.sidebar.header("Property Details")
sqft = st.sidebar.number_input("SqFt Living", 300, 10000, 2000)
beds = st.sidebar.slider("Bedrooms", 0, 10, 3) 
baths = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) 
year = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
renov = st.sidebar.number_input("Year Renovated", 0, 2025, 0)
floors = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, 0.5)
view = st.sidebar.slider("View (0-4)", 0, 4, 0)
cond = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
water = st.sidebar.selectbox("Waterfront", [0, 1])
bsmt = st.sidebar.number_input("SqFt Basement", 0, 5000, 0)
above = st.sidebar.number_input("SqFt Above", 300, 10000, 2000)

if st.sidebar.button("Valuate"):
    # Raw features matching User request
    features = {
        'sqft_living': sqft,
        'bedrooms': beds,
        'bathrooms': baths,
        'yr_built': year,
        'yr_renovated': renov,
        'floors': floors,
        'view': view,
        'condition': cond,
        'waterfront': water,
        'sqft_basement': bsmt,
        'sqft_above': above
    }
    
    # Align features to model
    input_df = pd.DataFrame([features]).reindex(columns=reg_cols, fill_value=0)
    cls_input = pd.DataFrame([features]).reindex(columns=cls_cols, fill_value=0)

    c1, c2 = st.columns(2)
    
    try:
        # Pure ML Model Prediction
        log_price = reg.predict(input_df)[0]
        price = np.expm1(log_price)
        
        cat = clf.predict(cls_input)[0]
        
        c1.subheader(f"Category: {cat}")
        c2.metric("Valuation", f"${price:,.0f}")
        
        st.divider()
        with st.spinner("Analyzing..."):
            insight = get_explanation(features, price, cat, stats)
            st.info(insight)
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
