# serve_model.py
import mlflow
import subprocess
import os

# Set tracking URI ke MLflow server
os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

# Run ID dari training Anda
RUN_ID = "cff0d5a0c69949978cfe893e53b4bd7c"
MODEL_PATH = f"runs:/{RUN_ID}/random_forest_model"

# Perintah untuk serve model
cmd = f'mlflow models serve -m "{MODEL_PATH}" --port 5001 --no-conda'

print("="*50)
print("🚀 Starting MLflow Model Server")
print("="*50)
print(f"Model: {MODEL_PATH}")
print(f"Port: 5001")
print("="*50)

# Jalankan serve
subprocess.run(cmd, shell=True)