import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger()

# ✅ Load Model & Preprocessor
preprocessor_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/model.pkl"

logger.info("🔄 Loading Preprocessor and Model...")
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)
with open(model_path, "rb") as f:
    model = pickle.load(f)
logger.info("✅ Model Loaded Successfully!")

expected_columns = [ "Time","V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        logger.info(f"📥 Received Data: {data}")

        # Convert form data to DataFrame
        df = pd.DataFrame([data])
        df = df.astype(float)
        df = df[expected_columns]

        # Transform data using preprocessor
        transformed_data = preprocessor.transform(df)
        prediction = model.predict(transformed_data)
        fraud_probability = model.predict_proba(transformed_data)[:, 1][0]

        result = {
            "prediction": int(prediction[0]),
            "fraud_probability": round(fraud_probability, 4),
            "message": "🚨 Fraudulent Transaction Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction"
        }

        logger.info(f"🔍 Prediction: {result['prediction']} | Fraud Probability: {result['fraud_probability']:.4f}")
        return render_template("result.html", result=result)

    except Exception as e:
        logger.error(f"❌ Prediction Failed: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
