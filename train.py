import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("soil_dataset.csv")

# Separate features & labels
X = df[["pH", "Moisture"]]
y = df["Label"]

# Encode labels: Normal → 1, Abnormal → 0
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# 2. Split Train/Test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# =========================
# 3. Train Random Forest
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 4. Evaluate Model
# =========================
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# =========================
# 5. Save Model
# =========================
joblib.dump(model, "soil_rf_model.pkl")
joblib.dump(le, "soil_label_encoder.pkl")

print("\n✅ Model saved as soil_rf_model.pkl")
print("✅ Label encoder saved as soil_label_encoder.pkl")
