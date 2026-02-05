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
    Act as a real estate analyst. Write a short, professional explanation for this valuation.
    Data:
    - Predicted Price: ${price:,.0f}
    - Category: {category}
    - Property: {features['sqft_living']} sqft, {features['bedrooms']} bed, {features['bathrooms']} bath, Grade {features['grade']}
    - Market Median: ${median:,.0f}
    - Market Avg $/sqft: ${avg_ppsqft:.0f}
    
    Output strictly:
    1. Reason for price (1 sentence).
    2. Market comparison (1 sentence).
    3. Final verdict (Realistic/High/Low).
    """

    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key: return "⚠️ Groq API Key missing."
        
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Insight unavailable: {str(e)}"

def validate_data(data):
    warnings = []
    # Basic sanity checks
    if data['grade'] > 10 and data['sqft_living'] < 1000:
        warnings.append("High grade with small area is unusual.")
        data['grade'] = 8
    if data['bedrooms'] > 0 and (data['sqft_living'] / data['bedrooms'] < 150):
        warnings.append(" Bedroom count seems high for this size.")
        data['bedrooms'] = int(data['sqft_living'] / 200)
    return warnings, data

def cap_price(price, data):
    # Prevent unrealistic outliers
    max_ppsf_map = {5: 150, 7: 300, 9: 500, 11: 800, 13: 1500}
    limit = next((v for k, v in max_ppsf_map.items() if data['grade'] <= k), 1500)
    
    if (price / data['sqft_living']) > limit:
        price = data['sqft_living'] * limit
    return price

# Main App
st.set_page_config(page_title="Real Estate AI", layout="wide")

try:
    clf, reg, cls_cols, reg_cols, stats = load_models()
except:
    st.error("Models failed to load. Please run the training notebook.")
    st.stop()

st.title("Real Estate Valuation AI 🏡")

# Sidebar
st.sidebar.header("Property Details")
s_sqft = st.sidebar.number_input("SqFt Living", 300, 10000, 2000)
s_grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
s_year = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
s_beds = st.sidebar.slider("Bedrooms", 0, 10, 3) 
s_baths = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) 
s_renov = st.sidebar.number_input("Renovated (Year)", 0, 2025, 0)
s_floors = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, 0.5)
s_water = st.sidebar.selectbox("Waterfront", [0, 1])
s_cond = st.sidebar.slider("Condition (1-5)", 1, 5, 3)

if st.sidebar.button("Valuate"):
    raw = {
        'sqft_living': s_sqft, 'grade': s_grade, 'yr_built': s_year,
        'bedrooms': s_beds, 'bathrooms': s_baths, 'yr_renovated': s_renov,
        'floors': s_floors, 'waterfront': s_water, 'condition': s_cond,
        'zipcode': 98000
    }
    
    warns, clean_data = validate_data(raw.copy())
    for w in warns: st.warning(w)
        
    # Feature Engineering
    features = clean_data.copy()
    features['house_age'] = 2025 - features['yr_built']
    features['has_renovated'] = 1 if features['yr_renovated'] > 0 else 0
    features['grade_sqft'] = features['grade'] * features['sqft_living']
    
    # Align features
    input_df = pd.DataFrame([features]).reindex(columns=reg_cols, fill_value=0)
    cls_input = pd.DataFrame([features]).reindex(columns=cls_cols, fill_value=0)

    c1, c2 = st.columns(2)
    
    # Predict
    try:
        # Price
        log_price = reg.predict(input_df)[0]
        price = cap_price(np.expm1(log_price), clean_data)
        
        # Category
        cat = clf.predict(cls_input)[0]
        
        c1.subheader(f"Category: {cat}")
        c2.metric("Valuation", f"${price:,.0f}")
        
        st.divider()
        with st.spinner("Analyzing..."):
            insight = get_explanation(clean_data, price, cat, stats)
            st.info(insight)
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
