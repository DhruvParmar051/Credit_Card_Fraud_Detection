"""
application.py: Flask app entry point for ML model inference.
"""

import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger()

# ✅ Load Model & Preprocessor
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
MODEL_PATH = "artifacts/model.pkl"

logger.info("🔄 Loading Preprocessor and Model...")
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

logger.info("✅ Model Loaded Successfully!")

EXPECTED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

@app.route("/", methods=["GET"])
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests and return model output."""
    try:
        data = request.form
        logger.info("📥 Received Data: %s", data)

        df = pd.DataFrame([data])
        df = df.astype(float)
        df = df[EXPECTED_COLUMNS]

        transformed_data = preprocessor.transform(df)
        prediction = model.predict(transformed_data)
        fraud_probability = model.predict_proba(transformed_data)[:, 1][0]

        result = {
            "prediction": int(prediction[0]),
            "fraud_probability": round(fraud_probability, 4),
            "message": (
                "🚨 Fraudulent Transaction Detected!"
                if prediction[0] == 1 else
                "✅ Legitimate Transaction"
            )
        }

        logger.info(
            "🔍 Prediction: %s | Fraud Probability: %.4f",
            result["prediction"],
            result["fraud_probability"]
        )

        return render_template("result.html", result=result)

    except (ValueError,KeyError) as e:
        logger.error("❌ Prediction Failed: %s", str(e))
        return jsonify({"error": "Prediction failed", "details": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
