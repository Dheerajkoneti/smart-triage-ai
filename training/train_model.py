import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.makedirs("../models", exist_ok=True)

df = pd.read_csv("../data/synthetic_data.csv")

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Encode Risk
label_encoder = LabelEncoder()
df["Risk_Level"] = label_encoder.fit_transform(df["Risk_Level"])

X = df[["Age", "Gender", "Blood_Pressure", "Heart_Rate", "Temperature"]]
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "../models/triage_model.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

print("Model saved successfully!")
