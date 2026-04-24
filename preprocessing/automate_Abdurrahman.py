# preprocessing/automate_Nama-siswa.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

def cap_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """Melakukan capping pada outliers berdasarkan persentil."""
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower, upper)
    return df

def preprocess_data(input_path, output_dir):
    """
    Menjalankan seluruh pipeline preprocessing.
    """
    print("="*60)
    print("Memulai Proses Preprocessing Otomatis")
    print("="*60)

    # 1. Load Data
    print(f"\n1. Memuat data dari {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   Data berhasil dimuat. Shape: {df.shape}")

    # 2. Handle Missing Values
    print("\n2. Menangani Missing Values...")
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    print("   Missing values telah diisi dengan median.")

    # 3. Handle Outliers
    print("\n3. Menangani Outliers (Capping)...")
    outlier_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    for col in outlier_cols:
        df = cap_outliers(df, col)
        print(f"   Capping applied on {col}")

    # 4. Handle Duplicates
    print("\n4. Menangani Duplikat...")
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"   {initial_len - len(df)} data duplikat dihapus. Shape sekarang: {df.shape}")

    # 5. Filter Unrealistic Data
    print("\n5. Memfilter data yang tidak realistis...")
    initial_len = len(df)
    df = df[(df['person_age'] >= 18) & (df['person_age'] <= 100)]
    print(f"   {initial_len - len(df)} data dihapus. Shape sekarang: {df.shape}")

    # 6. Encoding Data Kategorikal
    print("\n6. Encoding Fitur Kategorikal...")
    le_default = LabelEncoder()
    df['cb_person_default_on_file'] = le_default.fit_transform(df['cb_person_default_on_file'])
    
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"   Shape setelah encoding: {df.shape}")

    # 7. Pisahkan Features dan Target
    print("\n7. Memisahkan Features dan Target...")
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    print(f"   Features shape: {X.shape}, Target shape: {y.shape}")

    # 8. Feature Scaling
    print("\n8. Melakukan Feature Scaling...")
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                        'cb_person_cred_hist_length']
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    print("   Standarisasi selesai.")

    # 9. Split Train-Test
    print("\n9. Membagi data Train-Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 10. Simpan Hasil Preprocessing
    print(f"\n10. Menyimpan hasil preprocessing ke '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Simpan objek yang diperlukan untuk konsistensi di masa depan
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le_default, os.path.join(output_dir, 'label_encoder_default.pkl'))
    # Simpan nama kolom fitur untuk konsistensi saat inference
    joblib.dump(X.columns.tolist(), os.path.join(output_dir, 'feature_names.pkl'))
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING SELESAI!")
    print("="*60)


if __name__ == "__main__":
    # Tentukan path relatif terhadap root repository
    input_data_path = os.path.join('data_raw', 'credit_risk_dataset.csv')
    output_data_dir = os.path.join('preprocessing', 'credit_risk_dataset_preprocessed')
    preprocess_data(input_data_path, output_data_dir)
    
    #push ke github