# modelling.py
# ============================================
# MODEL TRAINING - CREDIT RISK SCORING
# Untuk Kriteria 2 - Basic (2 pts)
# ============================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
mlflow.sklearn.autolog()


# Set tracking URI ke local
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_Risk_Experiment")

print("="*60)
print("🚀 MODEL TRAINING - CREDIT RISK SCORING")
print("="*60)


# Load data hasil preprocessing
print("\n📂 Loading preprocessed data...")
X_train = pd.read_csv('../preprocessing/X_train.csv')
X_test = pd.read_csv('../preprocessing/X_test.csv')
y_train = pd.read_csv('../preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('../preprocessing/y_test.csv').values.ravel()

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   y_test shape: {y_test.shape}")

# Definisikan model yang akan dilatih
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Training dan logging dengan MLflow
for model_name, model in models.items():
    print(f"\n{'='*40}")
    print(f"📊 Training {model_name}...")
    print(f"{'='*40}")
    
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print(f"\n   📈 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Log parameters and metrics ke MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_state", 42)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log confusion matrix sebagai artifact
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   [[{cm[0,0]}, {cm[0,1]}]")
        print(f"    [{cm[1,0]}, {cm[1,1]}]]")
        
        # Save model
        mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))
        
        print(f"\n✅ Model {model_name} saved to MLflow!")
        
print("\n" + "="*60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\n📊 To view MLflow UI, run in terminal:")
print("   mlflow ui --port 5000")
print("   Then open: http://127.0.0.1:5000")