# serve_model.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
print("📂 Loading model...")
model = joblib.load('best_credit_risk_model.pkl')
print("✅ Model loaded successfully!")

@app.route('/invocations', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data['dataframe_records'])
    prediction = model.predict(df)
    return jsonify(prediction.tolist())

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("="*50)
    print("🚀 Model Server Running")
    print("📍 http://127.0.0.1:5001")
    print("="*50)
    app.run(port=5001, host='0.0.0.0', debug=False)