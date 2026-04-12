# automate_Abdurrahman.py
# ============================================
# AUTOMATIC PREPROCESSING PIPELINE
# Untuk Kriteria 1 - Skilled & Advance (3-4 pts)
# ============================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load dataset dari file CSV
    """
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Handle missing values pada dataset
    """
    print("🔧 Handling missing values...")
    df_clean = df.copy()
    
    # Cek missing values sebelum
    missing_before = df_clean.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # Isi person_emp_length dengan median
    if 'person_emp_length' in df_clean.columns:
        df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(
            df_clean['person_emp_length'].median()
        )
    
    # Isi loan_int_rate dengan median
    if 'loan_int_rate' in df_clean.columns:
        df_clean['loan_int_rate'] = df_clean['loan_int_rate'].fillna(
            df_clean['loan_int_rate'].median()
        )
    
    # Isi sisa missing dengan forward fill (ffill) - PERBAIKAN UNTUK PANDAS 2.x
    # Method 'ffill' menggantikan method='ffill' yang deprecated
    df_clean = df_clean.ffill()
    
    # Jika masih ada missing values, isi dengan 0 atau nilai default
    if df_clean.isnull().sum().sum() > 0:
        print(f"   Still have missing values, filling with 0...")
        df_clean = df_clean.fillna(0)
    
    # Cek missing values setelah
    missing_after = df_clean.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
    return df_clean


def handle_outliers(df, columns, lower_percentile=0.01, upper_percentile=0.99):
    """
    Handle outliers dengan capping
    """
    print("🔧 Handling outliers...")
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            lower = df_clean[col].quantile(lower_percentile)
            upper = df_clean[col].quantile(upper_percentile)
            before_outliers = len(df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)])
            df_clean[col] = df_clean[col].clip(lower, upper)
            print(f"   {col}: capped {before_outliers} outliers")
    
    return df_clean


def filter_unrealistic_data(df):
    """
    Filter data yang tidak realistis
    """
    print("🔧 Filtering unrealistic data...")
    df_clean = df.copy()
    
    # Usia 18-100 tahun
    if 'person_age' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[(df_clean['person_age'] >= 18) & (df_clean['person_age'] <= 100)]
        after = len(df_clean)
        print(f"   Age filter: removed {before - after} rows")
    
    # Income positif (jika ada)
    if 'person_income' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['person_income'] > 0]
        after = len(df_clean)
        print(f"   Income filter: removed {before - after} rows")
    
    return df_clean


def encode_categorical(df):
    """
    Encoding untuk fitur kategorikal
    """
    print("🔧 Encoding categorical features...")
    df_clean = df.copy()
    encoders = {}
    
    # Label Encoding untuk binary column
    if 'cb_person_default_on_file' in df_clean.columns:
        le = LabelEncoder()
        df_clean['cb_person_default_on_file'] = le.fit_transform(
            df_clean['cb_person_default_on_file'].astype(str)
        )
        encoders['cb_person_default_on_file'] = le
        print("   cb_person_default_on_file: Y=1, N=0")
    
    # One-Hot Encoding untuk multi-category
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
    for col in categorical_cols:
        if col in df_clean.columns:
            # Handle possible NaN values
            df_clean[col] = df_clean[col].fillna('Unknown')
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean = df_clean.drop(col, axis=1)
            print(f"   {col}: one-hot encoded into {len(dummies.columns)} columns")
    
    return df_clean, encoders


def scale_features(X, numeric_features, scaler=None):
    """
    Standarisasi fitur numerik
    """
    print("🔧 Scaling features...")
    
    # Pastikan hanya kolom yang ada yang digunakan
    available_features = [col for col in numeric_features if col in X.columns]
    
    if len(available_features) == 0:
        print("   No numeric features to scale!")
        return X, scaler
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[available_features])
        print("   Fitted new scaler")
    else:
        X_scaled = scaler.transform(X[available_features])
        print("   Used existing scaler")
    
    # Konversi kembali ke DataFrame
    X_scaled_df = X.copy()
    X_scaled_df[available_features] = X_scaled
    
    return X_scaled_df, scaler


def preprocess_pipeline(input_path, output_path, train_mode=True, apply_smote=True):
    """
    Pipeline preprocessing lengkap
    
    Parameters:
    - input_path: path ke dataset raw
    - output_path: folder untuk menyimpan hasil
    - train_mode: True untuk training (split + SMOTE), False untuk inference
    - apply_smote: apakah akan apply SMOTE untuk imbalance handling
    """
    print("\n" + "="*60)
    print("🚀 STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    df = load_data(input_path)
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    
    # 3. Handle outliers
    outlier_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    df = handle_outliers(df, outlier_cols)
    
    # 4. Filter unrealistic data
    df = filter_unrealistic_data(df)
    
    # 5. Encode categorical
    df, encoders = encode_categorical(df)
    
    # 6. Pisahkan features dan target
    target_col = 'loan_status'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"\n   Features shape: {X.shape}")
    print(f"   Target distribution:\n   {y.value_counts().to_dict()}")
    
    # 7. Scale numeric features
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                        'cb_person_cred_hist_length']
    
    X, scaler = scale_features(X, numeric_features)
    
    # 8. Save scaler dan encoders
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(scaler, f'{output_path}/scaler.pkl')
    joblib.dump(encoders, f'{output_path}/encoders.pkl')
    print(f"\n💾 Saved scaler and encoders to {output_path}")
    
    # 9. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Split results:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    
    # 10. Apply SMOTE untuk training (opsional)
    if train_mode and apply_smote:
        print("\n🔧 Applying SMOTE for imbalance handling...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE applied: X_train shape = {X_train.shape}")
        print(f"   New target distribution: {dict(pd.Series(y_train).value_counts())}")
    
    # 11. Simpan hasil preprocessing
    X_train.to_csv(f'{output_path}/X_train.csv', index=False)
    X_test.to_csv(f'{output_path}/X_test.csv', index=False)
    y_train.to_csv(f'{output_path}/y_train.csv', index=False)
    y_test.to_csv(f'{output_path}/y_test.csv', index=False)
    
    # Simpan dataset lengkap yang sudah diproses
    df_processed = pd.concat([X, y], axis=1)
    df_processed.to_csv(f'{output_path}/dataset_processed.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETED!")
    print("="*60)
    print(f"\n📁 Output saved to: {output_path}")
    print(f"   - X_train.csv: {X_train.shape}")
    print(f"   - X_test.csv: {X_test.shape}")
    print(f"   - y_train.csv: {y_train.shape}")
    print(f"   - y_test.csv: {y_test.shape}")
    print(f"   - dataset_processed.csv: {df_processed.shape}")
    print(f"   - scaler.pkl, encoders.pkl")
    
    return X_train, X_test, y_train, y_test, scaler, encoders


def load_and_preprocess_inference(input_path, output_path, scaler_path, encoder_path):
    """
    Load data baru dan preprocessing untuk inference
    """
    print("\n" + "="*60)
    print("🚀 INFERENCE PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load scaler dan encoders yang sudah disimpan
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    
    # Load data baru
    df = load_data(input_path)
    
    # Apply preprocessing yang sama
    df = handle_missing_values(df)
    outlier_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    df = handle_outliers(df, outlier_cols)
    df = filter_unrealistic_data(df)
    df, _ = encode_categorical(df)
    
    # Pisahkan features dan target (jika ada)
    if 'loan_status' in df.columns:
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
    else:
        X = df
        y = None
    
    # Scale features
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                        'cb_person_cred_hist_length']
    X, _ = scale_features(X, numeric_features, scaler)
    
    # Simpan hasil
    os.makedirs(output_path, exist_ok=True)
    X.to_csv(f'{output_path}/X_inference.csv', index=False)
    if y is not None:
        y.to_csv(f'{output_path}/y_inference.csv', index=False)
    
    print(f"\n✅ Inference preprocessing completed!")
    return X, y


if __name__ == "__main__":
    # Jalankan pipeline saat file di-run langsung
    INPUT_PATH = "data_raw/credit_risk_dataset.csv"
    OUTPUT_PATH = "preprocessing"
    
    # Cek apakah file dataset ada
    if not os.path.exists(INPUT_PATH):
        print(f"❌ ERROR: File tidak ditemukan di {INPUT_PATH}")
        print("📁 Pastikan file CSV sudah ditempatkan di folder 'data_raw/'")
    else:
        preprocess_pipeline(
            input_path=INPUT_PATH, 
            output_path=OUTPUT_PATH,
            train_mode=True,
            apply_smote=True
        )