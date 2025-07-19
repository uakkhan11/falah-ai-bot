# model_validation.ipynb

# ✅ Step 1: Load necessary libraries
import pandas as pd
import joblib
import numpy as np

# ✅ Step 2: Load your model
model = joblib.load("/root/falah-ai-bot/model.pkl")

# ✅ Step 3: Create dummy test data for features
test_data = pd.DataFrame({
    "RSI": [45, 55, 65, 75],
    "EMA10": [100, 105, 110, 115],
    "EMA21": [98, 100, 108, 112],
    "ATR": [2.0, 1.5, 1.8, 2.2],
    "VolumeChange": [1.1, 1.5, 1.3, 1.6]
})

# ✅ Step 4: Predict probabilities using model
probabilities = model.predict_proba(test_data)[:, 1]

# ✅ Step 5: Display results
for i, prob in enumerate(probabilities):
    ai_score = round(prob * 5, 2)
    print(f"Test Sample {i+1} - Probability: {prob:.4f} | AI Score: {ai_score}")
