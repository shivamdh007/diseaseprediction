import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("disease.csv")

# Clean column names
df.columns = df.columns.str.strip()

# 🔧 Encode categorical features

binary_map = {"Yes": 1, "No": 0}

df["Fever"] = df["Fever"].map(binary_map)
df["Cough"] = df["Cough"].map(binary_map)
df["Fatigue"] = df["Fatigue"].map(binary_map)
df["Difficulty Breathing"] = df["Difficulty Breathing"].map(binary_map)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

df["Blood Pressure"] = df["Blood Pressure"].map({
    "Low": 0,
    "Normal": 1,
    "High": 2
})

df["Cholesterol Level"] = df["Cholesterol Level"].map({
    "Normal": 0,
    "High": 1
})

# 🎯 TARGET = Disease (MULTI-CLASS)

label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])

# Features
X = df[[
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Age",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level"
]]

# Target
y = df["Disease"]

# 🚀 Better model than Decision Tree
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model + encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(label_encoder, open("encoder.pkl", "wb"))

print("✅ Model trained successfully!")

