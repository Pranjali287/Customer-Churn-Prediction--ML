import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="centered"
)

# Load Models
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/churn_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        features = joblib.load('models/features.pkl')
        return model, preprocessor, features
    except FileNotFoundError:
        st.error("Error: Model artifacts not found. Please run train_model.py first.")
        return None, None, None

model, preprocessor, features = load_artifacts()

# App UI
st.title("🔮 Customer Churn Prediction App")
st.write("Enter input data to check if a customer is likely to churn.")

if features:
    # Create input form
    with st.form("churn_form"):
        st.subheader("Customer Details")
        
        # Numeric Inputs
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, 
                         help="How long the customer has stayed with the company")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0,
                                        help="Average monthly bill amount")
        
        # Categorical Inputs
        col1, col2 = st.columns(2)
        
        with col1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                                  help="Type of customer subscription contract")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                                          help="Type of internet service used by the customer")
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], help="Preferred payment method of the customer")
            
        with col2:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            
        # Submit Button
        submit = st.form_submit_button("Predict Churn")

    if submit:
        # Create DataFrame from input
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'Contract': [contract],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'TechSupport': [tech_support],
            'PaymentMethod': [payment_method]
        })
        
        # Ensure column order matches training
        input_data = input_data[features['all_columns']]
        
        try:
            # Preprocess
            input_processed = preprocessor.transform(input_data)
            
            # Predict
            prediction = model.predict(input_processed)
            probability = model.predict_proba(input_processed)[0][1]
            
            # Display Results
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Probability Progress Bar
            st.write("### Churn Risk Probability")
            st.progress(probability, text=f"Probability: {probability:.1%}")

            if probability > 0.5: # Or use prediction[0] == 1, but probability enables threshold tuning if needed later
                st.error(f"🟥 **High Risk of Churn**\n\n⚠️ This customer is likely to churn ({probability:.1%} probability)")
            else:
                st.success(f"🟩 **Low Risk of Churn**\n\n✅ This customer is likely to stay ({probability:.1%} probability)")
                
            # Feature Importance Visualization (If applicable to model)
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("Feature Importance")
                
                # Get transformed feature names
                # This is tricky with pipelines, but we can try to extract them if needed
                # For this MVP, we might skip complex mapping or show raw index importance
                importances = model.feature_importances_
                # Simple bar chart
                st.bar_chart(importances)
                st.caption("Feature contribution to the prediction model.")
            elif hasattr(model, 'coef_'):
                 st.markdown("---")
                 st.subheader("Feature Importance (Coefficients)")
                 st.caption("Positive values increase churn risk, while negative values reduce churn risk.")
                 
                 try:
                     # Extract feature names and coefficients
                     feature_names = preprocessor.get_feature_names_out()
                     coefs = model.coef_[0]
                     
                     # Create DataFrame for plotting and analysis
                     feat_df = pd.DataFrame({'Importance': coefs}, index=feature_names)
                     
                     # Clean feature names for display
                     # Remove pipeline prefixes like 'num__', 'cat_multi__'
                     feat_df.index = [name.split('__')[-1].replace('_', ' ') for name in feat_df.index]
                     
                     # Display Chart
                     st.bar_chart(feat_df)
                     
                     # Identify Top Churn Drivers (Highest Positive Coefficients)
                     st.subheader("Top Churn Drivers")
                     
                     # Filter > 0 (Churn Drivers) and Sort Descending
                     top_drivers = feat_df[feat_df['Importance'] > 0].sort_values(by='Importance', ascending=False).head(3)
                     
                     if not top_drivers.empty:
                         for name, row in top_drivers.iterrows():
                             st.write(f"• {name}")
                     else:
                         st.write("No specific churn drivers identified.")
                         
                 except Exception as e:
                     # Fallback in case of feature name mismatch
                     st.warning(f"Could not extract feature names: {e}")
                     st.bar_chart(model.coef_[0])

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Please train the model to enable predictions.")
