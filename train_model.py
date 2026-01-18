import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH = 'data/cleaned_customer_churn.csv'
MODEL_PATH = 'models/churn_model.pkl'
METRICS_PATH = 'models/metrics.txt'

def load_data(file_path):
    """Load the dataset."""
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}. Please make sure the data exists.")
        return None

def preprocess_data(df):
    """
    Preprocess the data:
    - Encode binary features with Label Encoding
    - Encode multi-class features with One-Hot Encoding
    - Scale numerical features
    """
    # Separate target
    target = 'Churn'
    if target not in df.columns:
        print("❌ Error: Target column 'Churn' not found.")
        return None, None, None, None, None
    
    # Define features to use (subset for the app)
    # matching the UI requirements: tenure, MonthlyCharges, Contract, InternetService, OnlineSecurity, TechSupport, PaymentMethod
    # We ignore others for this production pipeline to ensure App compatibility
    feature_cols = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 
                   'OnlineSecurity', 'TechSupport', 'PaymentMethod']
    
    # Ensure columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"❌ Error: Missing columns in data: {missing}")
        return None, None, None, None, None

    X = df[feature_cols]
    y = df[target]

    # Convert Target 'Yes'/'No' to 1/0 if necessary
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, 'models/target_encoder.pkl')
    
    # Identify feature types for the reduced set
    numerical_cols = ['tenure', 'MonthlyCharges']
    categorical_cols = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
    
    # Check dataset structure to be sure
    binary_cols = [c for c in categorical_cols if X[c].nunique() <= 2]
    multi_cols = [c for c in categorical_cols if X[c].nunique() > 2]
    
    print(f"Selected Features: {feature_cols}")

    # Build Transformers
    # For binary: We use OrdinalEncoder to keep it 0/1
    # For multi-class: We use OneHotEncoder
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat_binary', Pipeline([('ordinal', OneHotEncoder(drop='if_binary'))]), binary_cols), # Handles binary efficiently
            ('cat_multi', OneHotEncoder(handle_unknown='ignore'), multi_cols)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest."""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} trained.")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return the best one."""
    best_model = None
    best_score = 0
    best_name = ""

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"\n📊 {name} Performance:")
        print(f" - Accuracy: {accuracy:.4f}")
        print(f" - Precision: {precision:.4f}")
        print(f" - Recall: {recall:.4f}")
        print(f" - F1 Score: {f1:.4f}")
        print(f" - ROC AUC: {roc_auc:.4f}")

        # Model selection logic (optimizing for Recall/F1 usually best for Churn, but using Accuracy/F1 as general proxy)
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} with F1-Score: {best_score:.4f}")
    
    # Save metrics
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"F1 Score: {best_score:.4f}\n")
        for metric, value in results[best_name].items():
            f.write(f"{metric}: {value:.4f}\n")
            
    return best_model, best_name, results

def main():
    # 1. Load Data
    df = load_data(DATA_PATH)
    if df is None:
        return

    # 2. Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    if preprocessor is None:
        return

    # Fit transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 3. Model Training
    models = train_models(X_train_processed, y_train)

    # 4. Evaluation
    best_model, best_name, results = evaluate_models(models, X_test_processed, y_test)

    # 5. Save Artifacts (Model + Preprocessor)
    # We save the pipeline: Preprocessor + Model
    # This makes the app cleaner as it just loads one object ideally, 
    # but we were asked to save encoders/scalers separate.
    # To strictly follow "Save encoders and scalers", we save the preprocessor object 
    # which contains all transformers.
    
    print("💾 Saving model and artifacts...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # Save column columns list for the app to know expected input order if needed
    feature_cols = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 
                   'OnlineSecurity', 'TechSupport', 'PaymentMethod']
    features = {
        'numerical': ['tenure', 'MonthlyCharges'], 
        'categorical': ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod'],
        'all_columns': feature_cols
    }
    joblib.dump(features, 'models/features.pkl')

    print("✅ Training complete. Model saving to models/ folder.")

if __name__ == "__main__":
    main()
