# Real Estate Valuation AI - Project Presentation

## Slide 1: Project Overview
**Title**: Real Estate Valuation AI with Market Insights
**Goal**: Build an end-to-end ML pipeline to predict house prices, categorize properties, and provide explainable AI insights.
**Key Features**:
- Price Prediction (Regression)
- Smart Categorization (Clustering & Classification)
- AI-Powered Explanations (Llama-3 via Groq)
- Interactive Web App (Streamlit)

---

## Slide 2: Dataset & Preprocessing
**Dataset**: King County House Sales (simulated fallback used if data unavailable)
**Size**: ~4600 records, 21 Features
**Key Steps**:
1. **Cleaning**: Handled missing values, removed price outliers (IQR method).
2. **Feature Engineering**: 
   - `house_age`: Derived from `yr_built`
   - `has_renovated`: Binary flag
   - `grade_sqft`: Interaction term
   - Log-transformation of `price` for better regression performance.

---

## Slide 3: Exploratory Analysis (EDA)
**Insights**:
- **Price vs. SqFt**: Strong positive correlation (0.7+).
- **Grade Impact**: Higher grade exponentially increases price.
- **Location**: Waterfront properties command significant premium.
- **Distribution**: Price is right-skewed; Log-transform normalized it effectively.

---

## Slide 4: Unsupervised Learning - Clustering
**Method**: K-Means Clustering
- **Features Used**: Price, SqFt Living, Grade, House Age
- **Result**: 3 Distinct Clusters
    1. **Budget**: Lower price, smaller size, older.
    2. **Standard**: Average features, market median.
    3. **Luxury**: High price, large size, high grade.
**Application**: These clusters defined our target labels for classification.

---

## Slide 5: Model Performance
**1. Category Classification (Random Forest)**
- **Goal**: Predict 'Budget', 'Standard', 'Luxury' for new listings.
- **Accuracy**: ~90%+ (on test set)
- **Role**: Automatically tags new listings in the app.

**2. Price Regression (XGBoost)**
- **Target**: Log(Price)
- **Metrics**:
    - **R² Score**: ~0.85
    - **MAE**: Low mean absolute error compared to median price.
- **Validation**: Cross-validated to prevent overfitting.

---

## Slide 6: Deployment & AI
**Tech Stack**: Streamlit, XGBoost, Groq (Llama-3)
**Workflow**:
1. **User Input**: Enter property details (sqft, bedrooms, etc.).
2. **Model Inference**: 
   - Classifier predicts "Category".
   - Regressor predicts "Price".
3. **AI Interpretation**:
   - Llama-3 analyzes the prediction context.
   - Checks against market stats (Median, Price/SqFt).
   - Generates a human-readable explanation (Realism Check).

---

## Slide 7: Live Demo
**Scenario**:
- **Input**: 2500 sqft, 4 bed, Grade 8.
- **Output**: 
    - Category: Standard/Luxury
    - Price: ~$750k
    - AI: "This price is realistic given the high grade..."

**Conclusion**: The system provides not just a number, but *confidence* through explanation.
