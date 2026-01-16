import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Sample data to simulate training
data = {
    "skill": [8, 6, 4, 7, 9, 3],
    "logic": [9, 7, 3, 8, 8, 4],
    "creativity": [2, 4, 9, 3, 3, 9],
    "teamwork": [6, 7, 6, 8, 7, 6],
    "career": [0, 1, 5, 2, 3, 5]  # Career labels
}
df = pd.DataFrame(data)

X = df.drop("career", axis=1)
y = df["career"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to model/ folder
os.makedirs("model", exist_ok=True)
with open("model/career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/career_model.pkl")
