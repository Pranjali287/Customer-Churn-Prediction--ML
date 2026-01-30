import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for bigger fonts and better UI
st.markdown("""
<style>
    /* Global Font scaling */
    html, body, [class*="st-"] {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
    }
    
    /* Input Labels - Increased size */
    .stTextInput > label, .stSelectbox > label, .stSlider > label, .stNumberInput > label, .stRadio > label {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #e0e0e0;
    }
    
    /* Standard Text & Paragraphs */
    p, .stMarkdown, .stText {
        font-size: 1.3rem !important;
    }
    
    /* Bigger Headers - Increased size */
    h1 { font-size: 4rem !important; }
    h2 { font-size: 3rem !important; }
    h3 { font-size: 2.2rem !important; }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.5rem !important;
        padding: 10px 24px !important;
    }
    
    /* Login Box */
    .login-container {
        padding: 50px;
        border-radius: 10px;
        background-color: #262730;
        margin-top: 50px;
        border: 1px solid #4c4c4c;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        color: black;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    
    .metric-card h1 { font-size: 4rem !important; }
    .metric-card h3 { font-size: 2rem !important; }
    .metric-card p { font-size: 1.3rem !important; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'page' not in st.session_state:
    st.session_state.page = 'input' # input or result
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# --- Load Models ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/churn_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        features = joblib.load('models/features.pkl')
        return model, preprocessor, features
    except FileNotFoundError:
        return None, None, None

model, preprocessor, features = load_artifacts()

# --- Page: Login ---
def login_page():
    # Centered layout using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🔐 Login to Customer Churn Predictor")
        st.markdown("### Secure Enterprise Access")
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="admin@company.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("Log In", type="primary", use_container_width=True)
            
            if submit_login:
                if email and password:
                    # Simulation of login
                    st.success("Authentication successful! Redirecting...")
                    st.session_state.authenticated = True
                    st.session_state.page = 'input' # Reset to input on login
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("⚠️ Please enter both email and password.")

# --- Page: Input Form ---
def input_page():
    # Header with Logout
    head_col1, head_col2 = st.columns([6, 1])
    with head_col1:
        st.title("🛡️ Customer Churn Predictor | New Analysis")
    with head_col2:
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

    st.markdown("---")
    
    if not features:
        st.error("Model artifacts not found. Please train the model first.")
        return

    # User Input Section
    st.subheader("📝 Customer Information")
    st.info("Please fill out the customer details in the sections below to generate a comprehensive risk assessment.")

    with st.form("churn_prediction_form"):
        
        # Section 1: Service Usage
        st.markdown("### 👤 1. Service Profile")
        st.markdown("Basic usage details about the customer's account.")
        
        tenure = st.slider("Tenure (How many months have they been with us?)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        
        st.markdown("---")

        # Section 2: Contract Info
        st.markdown("### 📄 2. Contract & Billing")
        st.markdown("Details regarding their subscription plan and payment habits.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col_c2:
            internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])

        st.markdown("---")

        # Section 3: Add-ons
        st.markdown("### 🛡️ 3. Security & Support")
        st.markdown("Additional services the customer is currently subscribed to.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            online_security = st.radio("Has Online Security?", ["No", "Yes", "No internet service"], horizontal=True)
        with col_s2:
            tech_support = st.radio("Has Tech Support?", ["No", "Yes", "No internet service"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Centered Submit Button
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            submit_prediction = st.form_submit_button("✨ Analyze Customer Risk", type="primary", use_container_width=True)

    if submit_prediction:
        # Save data to session state
        st.session_state.input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'TechSupport': tech_support,
            'PaymentMethod': payment_method
        }
        st.session_state.page = 'result'
        st.rerun()

# --- Page: Result Dashboard ---
def result_page():
    # Header with Navigation
    head_col1, head_col2 = st.columns([6, 1])
    with head_col1:
        st.title("📊 Analysis Dashboard")
    with head_col2:
        if st.button("New Analysis"):
            st.session_state.page = 'input'
            st.rerun()
            
    st.markdown("---")

    if st.session_state.input_data is None:
        st.warning("No data to analyze. Please go back and enter customer details.")
        return

    # Retrieve data
    input_data = pd.DataFrame([st.session_state.input_data])
    
    # Ensure column order matches
    input_data = input_data[features['all_columns']]
    
    try:
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)
        probability = model.predict_proba(input_processed)[0][1]
        
        # Result Columns
        # res_col1, res_col2 = st.columns([1, 2]) # Removed side-by-side
        
        # --- Architecture: Vertical Flow ---
        
        # 1. RISK SCORE SECTION
        st.markdown("---")
        
        # Determine Styling
        if probability > 0.5:
            risk_color = "#ff4b4b" # Red
            risk_label = "HIGH CHURN RISK"
            risk_bg = "rgba(255, 75, 75, 0.1)"
        else:
            risk_color = "#28a745" # Green
            risk_label = "LOW CHURN RISK"
            risk_bg = "rgba(40, 167, 69, 0.1)"
        
        # Centered Hero Section for Result
        st.markdown(f"""
        <div style="background-color: {risk_bg}; padding: 30px; border-radius: 15px; border: 2px solid {risk_color}; text-align: center; margin-bottom: 30px;">
            <h3 style="color: {risk_color}; margin-bottom: 10px;">PREDICTED STATUS</h3>
            <h1 style="color: {risk_color}; font-size: 5.5rem; margin: 0;">{probability:.1%}</h1>
            <p style="font-size: 1.2rem; color: #555; margin-top: 10px;">Probability that this customer will leave</p>
            <div style="background-color: {risk_color}; color: white; display: inline-block; padding: 10px 30px; border-radius: 30px; font-weight: bold; font-size: 1.5rem; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. RECOMMENDATIONS SECTION
        st.subheader("💡 AI-Powered Recommendations")
        
        if probability > 0.5:
            st.info("⚠️ **This customer requires immediate attention.** Based on their profile, here is a personalized retention strategy:")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1:
                st.markdown("""
                <div class="metric-card">
                    <h1>📞</h1>
                    <h3>Contact</h3>
                    <p>Schedule a check-in call within <b>24 hours</b> to discuss their experience.</p>
                </div>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.markdown("""
                <div class="metric-card">
                    <h1>🏷️</h1>
                    <h3>Incentivize</h3>
                    <p>Offer a <b>15% discount</b> for 3 months if they renew their contract.</p>
                </div>
                """, unsafe_allow_html=True)
            with rec_col3:
                st.markdown("""
                <div class="metric-card">
                    <h1>🔧</h1>
                    <h3>Support</h3>
                    <p>Escalate any open support tickets to <b>Priority Level 1</b>.</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.success("✅ **This customer is currently happy.** Use this opportunity to deepen the relationship:")
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.markdown("""
                <div class="metric-card">
                    <h1>📈</h1>
                    <h3>Upsell</h3>
                    <p>They might be interested in our <b>Premium Plan</b> or higher speeds.</p>
                </div>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.markdown("""
                <div class="metric-card">
                    <h1>❤️</h1>
                    <h3>Engage</h3>
                    <p>Send a personalized <b>Thank You</b> email with loyalty points.</p>
                </div>
                """, unsafe_allow_html=True)

        # 3. INTERACTIVE FEATURE ANALYSIS
        st.markdown("---")
        st.subheader("🔍 What is driving this prediction?")
        st.markdown("The chart below shows which factors are pushing the risk **UP (Red)** or pulling it **DOWN (Blue)**.")

        if hasattr(model, 'coef_'):
            try:
                feature_names = preprocessor.get_feature_names_out()
                coefs = model.coef_[0]
                
                # Create clean dataframe
                feat_df = pd.DataFrame({
                    'Feature': [name.split('__')[-1].replace('_', ' ') for name in feature_names],
                    'Impact': coefs
                })
                
                # Add color column based on impact
                feat_df['Type'] = feat_df['Impact'].apply(lambda x: 'High Risk Factor' if x > 0 else 'Retention Factor')
                
                # Sort by absolute magnitude to show most important first
                feat_df['AbsImpact'] = feat_df['Impact'].abs()
                feat_df = feat_df.sort_values('AbsImpact', ascending=False).head(10)
                
                # Use a specialized chart instead of generic bar_chart
                # We can use st.dataframe with bar logic or a better chart library if available
                # Sticking to st.bar_chart but customized via dataframe separation or just using Vega-Lite (via st.vega_lite_chart or st.altair_chart) is best for interactivity
                # Let's use a nice st.dataframe visualisation which is very clean and interactive
                
                st.markdown("### Top Influencing Factors")
                
                # We will create a display dataframe
                display_df = feat_df[['Feature', 'Impact']].copy()
                display_df['Impact Direction'] = display_df['Impact'].apply(lambda x: "🔴 Increases Churn" if x > 0 else "🔵 Reduces Churn")
                
                # Configuration for the progress bar visual
                st.dataframe(
                    display_df,
                    column_config={
                        "Impact": st.column_config.ProgressColumn(
                            "Impact Strength",
                            help="The magnitude of the effect",
                            format="%.2f",
                            min_value=-1.5,
                            max_value=1.5,
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption("Values closer to +1.0 meant strong churn risk. Values closer to -1.0 mean strong retention signal.")

            except Exception as e:
                st.write("Could not generate feature breakdown.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# --- App Logic Control Flow ---
if not st.session_state.authenticated:
    login_page()
else:
    if st.session_state.page == 'input':
        input_page()
    elif st.session_state.page == 'result':
        result_page()

