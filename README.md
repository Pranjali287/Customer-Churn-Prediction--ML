# 🔮 Customer Churn Prediction System

## 📌 Project Overview
The **Customer Churn Prediction System** is an end-to-end Machine Learning solution designed to help businesses reduce customer attrition. By analyzing customer demographics, service usage, and billing details, the system predicts the likelihood of a customer leaving (churning) and identifies key risk factors.

The project demonstrates a complete Data Science lifecycle: from data preprocessing and model training to deployment via an interactive **Streamlit** web application.

## ❓ Problem Statement
Customer churn is a critical challenge for subscription-based businesses. Acquiring a new customer can cost **5-25x more** than retaining an existing one. This project aims to:
1.  **Predict Churn Probability**: Accurately forecast if a customer is at risk.
2.  **Identify Key Drivers**: Determine which factors (e.g., Contract type, Monthly Charges) contribute most to churn.
3.  **Actionable Insights**: Provide a user-friendly tool for non-technical stakeholders to test scenarios.

## 📂 Project Structure
```text
customer-churn-prediction/
│── data/
│   └── cleaned_customer_churn.csv   # Cleaned dataset used for training
│── models/
│   ├── churn_model.pkl              # Trained Machine Learning Model
│   ├── preprocessor.pkl             # Scikit-learn Processing Pipeline
│   ├── features.pkl                 # Feature metadata
│   └── metrics.txt                  # Model performance logs
│── app.py                           # Streamlit Web Application
│── train_model.py                   # Automated Training Pipeline
│── requirements.txt                 # Project Dependencies
│── README.md                        # Documentation
```

## 📊 Dataset & Features
The model is trained on a telecom customer dataset including:
-   **Demographics**: Gender, Senior Citizen status, Partner, Dependents.
-   **Services**: Phone Service, Internet Service (DSL/Fiber), Online Security, Tech Support.
-   **Account Details**: Contract Type (Month-to-month vs. Yearly), Payment Method, Monthly Charges, Tenure.

## ⚙️ Model & Preprocessing
The solution uses robust data processing and machine learning techniques:
-   **Preprocessing**:
    -   **Label Encoding** for binary features.
    -   **One-Hot Encoding** for categorical variables (e.g., Payment Method).
    -   **Standard Scaling** for numerical features (Tenure, Monthly Charges).
-   **Models Evaluated**: Logistic Regression and Random Forest Classifier.
-   **Best Model**: Automatically selected based on **F1-Score** to balance Precision and Recall.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone <repository_url>
cd customer-churn-prediction
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to retrain the model on new data:
```bash
python train_model.py
```
*This script will process data in `data/`, train models, evaluate them, and save the best artifacts to `models/`.*

### 4. Run the Streamlit App
Launch the web interface:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🖥️ Streamlit App Features
-   **Real-time Prediction**: Adjust sliders and dropdowns to see instant churn probabilities.
-   **Risk Visualization**: Dynamic progress bar indicating high (red) or low (green) churn risk.
-   **Explainability**: Visualizes feature importance and lists the **Top 3 Churn Drivers** for the specific prediction.

## 🔮 Future Improvements
-   Integrate **SHAP values** for deeper local interpretability.
-   Add support for **Batch Predictions** via CSV upload.
-   Experiment with advanced models like **XGBoost** or **LightGBM**.
-   Dockerize the application for easier cloud deployment.

---
*Built with ❤️ using Python & Streamlit*
