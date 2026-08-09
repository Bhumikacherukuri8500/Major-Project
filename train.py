import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv(
    "dataset/patients_data_with_alerts.csv",
    encoding="latin1"
)

print("\nCOLUMNS IN DATASET:")
print(df.columns)

# ===============================
# 2. Target Column (ALERT)
# ===============================
target_column = "Temperature Alert"

# ===============================
# 3. Feature Selection (ONLY NUMERIC)
# ===============================
feature_columns = [
    'Heart Rate (bpm)',
    'SpO2 Level (%)',
    'Systolic Blood Pressure (mmHg)',
    'Diastolic Blood Pressure (mmHg)',
    'Body Temperature (°C)'
]

X = df[feature_columns]
y = df[target_column]

# ===============================
# 4. Encode Target
# ===============================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ===============================
# 5. Scale Features
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 7. STRONG MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 8. Evaluation
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🔥 Model Accuracy: {accuracy * 100:.2f}%")

# ===============================
# 9. Save Model
# ===============================
joblib.dump(model, "model/iomt_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("✅ Model, scaler, and encoder saved successfully!")
