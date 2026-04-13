# modelling_tuning.py
# ============================================
# MODEL TRAINING WITH HYPERPARAMETER TUNING
# Untuk Kriteria 2 - Skilled (3 pts) / Advanced (4 pts)
# ============================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Tambahkan ini di bagian atas file, setelah import
import dagshub
import mlflow

# Initialize DagsHub
dagshub.init(repo_owner='Abdurrkhmn', repo_name='MLflow_Credit_Risk', mlflow=True)

# Set tracking URI explicitly
mlflow.set_tracking_uri(f"https://dagshub.com/Abdurrkhmn/MLflow_Credit_Risk.mlflow")
mlflow.set_experiment("Credit_Risk_Tuning")

print("="*60)
print("🚀 MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*60)

# Load data hasil preprocessing
# Load data hasil preprocessing
print("\n📂 Loading preprocessed data...")
X_train = pd.read_csv('../preprocessing/X_train.csv')
X_test = pd.read_csv('../preprocessing/X_test.csv')
y_train = pd.read_csv('../preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('../preprocessing/y_test.csv').values.ravel()

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
print("\n🔧 Performing Grid Search for best hyperparameters...")
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='roc_auc', 
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")

# Train best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 Test Set Performance:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# MLflow logging (manual logging)
with mlflow.start_run(run_name="RandomForest_Tuned"):
    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring_metric", "roc_auc")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log confusion matrix plot (extra artifact untuk advanced)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1])
    axes[1].set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_and_feature_importance.png')
    mlflow.log_artifact('confusion_matrix_and_feature_importance.png')
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")
    
    # Save model locally
    joblib.dump(best_model, 'best_credit_risk_model.pkl')
    mlflow.log_artifact('best_credit_risk_model.pkl')
    
    print("\n✅ Model saved to MLflow with all artifacts!")
    
    # Print MLflow run ID
    print(f"\n📌 MLflow Run ID: {mlflow.active_run().info.run_id}")

print("\n" + "="*60)
print("✅ HYPERPARAMETER TUNING COMPLETED!")
print("="*60)
print("\n📊 To view MLflow UI, run in terminal:")
print("   mlflow ui --port 5000")
print("   Then open: http://127.0.0.1:5000")

