import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# Define X and y
X = df[['YearsExperience']]
y = df['Salary']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully.")
