import pandas as pd
import numpy as np

# Create dummy data
np.random.seed(42)
n_samples = 100

data = {
    'customerID': [f'{i}' for i in range(n_samples)],
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
    'tenure': np.random.randint(1, 73, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
    'MonthlyCharges': np.random.uniform(18.0, 118.0, n_samples),
    'TotalCharges': np.random.uniform(18.0, 8000.0, n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples)
}

df = pd.DataFrame(data)
df.to_csv('data/cleaned_customer_churn.csv', index=False)
print("Dummy data created at data/cleaned_customer_churn.csv")
