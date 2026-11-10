import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv("diabetes.csv")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Build Pipeline (Using the new sklearn version)
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# 4. Train
pipe.fit(X_train, y_train)

# 5. Save (This overwrites the broken file)
joblib.dump(pipe, "diabetes_model.pkl")
print("✅ New model trained and saved successfully with scikit-learn 1.9.1!")