"""
=============================================================================
BƯỚC 1, 2, 3: XÁC ĐỊNH BÀI TOÁN, THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Đề tài: Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe y tế
File dữ liệu gốc: CSDL_TieuDuong.csv
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font/UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# BƯỚC 1. XÁC ĐỊNH BÀI TOÁN
# ---------------------------------------------------------------------------
"""
1. Mục tiêu bài toán:
   Xây dựng mô hình Học máy (Machine Learning) hỗ trợ chẩn đoán và dự đoán 
   nguy cơ mắc bệnh tiểu đường của một cá nhân dựa trên 13 chỉ số thông tin 
   cá nhân và xét nghiệm sinh hóa máu.

2. Dạng bài toán:
   - Phân loại nhị phân (Binary Classification).
   - Đầu vào (Features - X): 13 thuộc tính:
     + Tuổi (Tuoi), Giới tính (GioiTinh)
     + Đường huyết (Glucose), HbA1C
     + Bilan lipid máu: Cholesterol, HDL, LDL, Triglycerid
     + Chức năng thận: Creatinine, Ure, Acid Uric
     + Men gan: AST, ALT
   - Đầu ra (Target - y): NguyCoTieuDuong (0: Không có nguy cơ / 1: Có nguy cơ).

3. Tiêu chí đặc thù y tế:
   Trong bài toán chẩn đoán y khoa, việc "bỏ sót người thực sự có bệnh" 
   (False Negative) là cực kỳ nguy hiểm. Do đó, bên cạnh Độ chính xác (Accuracy),
   chỉ số Độ nhạy (Recall) và F1-Score đối với lớp bệnh nhân (nhãn 1) là ưu tiên hàng đầu.
"""

def load_data(file_path="CSDL_TieuDuong.csv"):
    """
    BƯỚC 2: THU THẬP VÀ ĐỌC DỮ LIỆU Y TẾ THỰC TẾ
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"[Bước 2] Đọc thành công dữ liệu từ '{file_path}'. Kích thước ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df

def clean_and_preprocess_data(df):
    """
    BƯỚC 3. TIỀN XỬ LÝ DỮ LIỆU
    3.1. Làm sạch dữ liệu:
       - Đồng bộ dữ liệu nhãn mục tiêu và biến phân loại
       - Kiểm tra và xử lý dữ liệu lỗi logic y khoa
       - Kiểm tra và xử lý dữ liệu ngoại lai
       - Kiểm tra và điền khuyết thiếu (Missing values)
    3.2. Biến đổi dữ liệu:
       - Mã hóa biến phân loại (GioiTinh)
       - Chuẩn hóa các biến liên tục (StandardScaler)
    """
    print("\n" + "="*60)
    print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU")
    print("="*60)
    df_clean = df.copy()

    # -----------------------------------------------------------------------
    # 3.1.1. ĐỒNG BỘ DỮ LIỆU (Data Synchronization)
    # -----------------------------------------------------------------------
    # A. Đồng bộ cột mục tiêu NguyCoTieuDuong sang nhị phân 0 và 1:
    # Trong file gốc chứa: 'FALSE', 'TRUE', '1', '0', 'Có nguy cơ', 'Không có nguy cơ'
    target_mapping = {
        'TRUE': 1, 'True': 1, True: 1, '1': 1, 1: 1, 'Có nguy cơ': 1,
        'FALSE': 0, 'False': 0, False: 0, '0': 0, 0: 0, 'Không có nguy cơ': 0
    }
    df_clean['NguyCoTieuDuong'] = df_clean['NguyCoTieuDuong'].map(target_mapping)
    print(f"[3.1. Đồng bộ] Phân bố nhãn mục tiêu sau chuẩn hóa:\n{df_clean['NguyCoTieuDuong'].value_counts().to_dict()}")

    # B. Đồng bộ thuộc tính Giới tính (GioiTinh):
    # Chuẩn hóa 'Nam'/'Male' -> 1, 'Nữ'/'Female' -> 0
    gender_mapping = {
        'Nam': 1, 'Male': 1, 'nam': 1, 'male': 1,
        'Nữ': 0, 'Female': 0, 'nữ': 0, 'female': 0
    }
    df_clean['GioiTinh'] = df_clean['GioiTinh'].map(gender_mapping)

    # -----------------------------------------------------------------------
    # 3.1.2. KIỂM TRA VÀ XỬ LÝ DỮ LIỆU LỖI LOGIC Y KHOA (Medical Error Handling)
    # -----------------------------------------------------------------------
    # Lỗi 1: HbA1C có giá trị 8025.0 (lỗi nhập thiếu dấu thập phân của 8.025%)
    # Ngưỡng sinh lý HbA1c thông thường từ 3.5% đến 20.0%
    hba1c_errors = df_clean['HbA1C'] > 20.0
    if hba1c_errors.sum() > 0:
        print(f"[3.1. Sửa lỗi] Phát hiện {hba1c_errors.sum()} giá trị HbA1C vượt ngưỡng y khoa (>20%). Điều chỉnh chia 1000.")
        df_clean.loc[hba1c_errors, 'HbA1C'] = df_clean.loc[hba1c_errors, 'HbA1C'] / 1000.0

    # Lỗi 2: Cholesterol có giá trị 55.0 (lỗi nhập thiếu dấu phẩy thập phân của 5.5 mmol/L)
    # Ngưỡng Cholesterol bình thường là 2.5 - 15 mmol/L. Giá trị > 25 được điều chỉnh chia 10.
    chol_errors = df_clean['Cholesterol'] > 25.0
    if chol_errors.sum() > 0:
        print(f"[3.1. Sửa lỗi] Phát hiện {chol_errors.sum()} giá trị Cholesterol bất thường (>25). Điều chỉnh chia 10.")
        df_clean.loc[chol_errors, 'Cholesterol'] = df_clean.loc[chol_errors, 'Cholesterol'] / 10.0

    # Lỗi 3: Glucose có các giá trị > 50 (ví dụ 59.0, 61.0, 92.0, 750.0)
    # Đây là các bản ghi bị nhập nhầm đơn vị mg/dL thay vì mmol/L!
    # Hệ số quy đổi tiêu chuẩn y tế: 1 mmol/L = 18.0182 mg/dL.
    # Quy đổi: 59 mg/dL -> 3.27 mmol/L (bình thường, nhãn FALSE phù hợp); 750 mg/dL -> 41.62 mmol/L (đái tháo đường nặng, nhãn TRUE).
    glucose_unit_errors = df_clean['Glucose'] > 50.0
    if glucose_unit_errors.sum() > 0:
        print(f"[3.1. Sửa lỗi] Phát hiện {glucose_unit_errors.sum()} giá trị Glucose ghi nhầm đơn vị mg/dL (>50). Quy đổi sang mmol/L (/18.0182).")
        df_clean.loc[glucose_unit_errors, 'Glucose'] = df_clean.loc[glucose_unit_errors, 'Glucose'] / 18.0182

    # -----------------------------------------------------------------------
    # 3.1.3. KIỂM TRA VÀ XỬ LÝ DỮ LIỆU THIẾU (Missing Values Imputation)
    # -----------------------------------------------------------------------
    missing_summary = df_clean.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    print(f"[3.1. Dữ liệu thiếu] Các cột khuyết thiếu:\n{cols_with_missing.to_dict()}")

    # Điền khuyết thiếu cho Giới tính bằng Mode (Giá trị xuất hiện nhiều nhất)
    mode_gender = df_clean['GioiTinh'].mode()[0]
    df_clean['GioiTinh'] = df_clean['GioiTinh'].fillna(mode_gender)

    # Điền khuyết thiếu cho các chỉ số số học bằng Median (Trung vị)
    # Trung vị kháng lại tốt hơn giá trị ngoại lai so với Mean
    numeric_feature_cols = [
        'Tuoi', 'Glucose', 'HbA1C', 'Cholesterol', 'HDL', 'LDL',
        'Triglycerid', 'Creatinine', 'Ure', 'AcidUric', 'AST', 'ALT'
    ]
    imputation_dict = {'GioiTinh': mode_gender}
    for col in numeric_feature_cols:
        med_val = df_clean[col].median()
        imputation_dict[col] = med_val
        df_clean[col] = df_clean[col].fillna(med_val)

    # -----------------------------------------------------------------------
    # 3.1.4. XỬ LÝ GIÁ TRỊ NGOẠI LAI (Outlier Treatment)
    # -----------------------------------------------------------------------
    # Trong dữ liệu xét nghiệm y tế, các giá trị cực trị (ví dụ suy thận nặng làm Creatinine > 1000,
    # viêm gan cấp làm AST/ALT > 500) là các biểu hiện lâm sàng thực tế, không nên xóa dòng vì
    # làm mất thông tin người bệnh nguy cơ cao. Ta áp dụng phương pháp Winsorization (Clipping)
    # giới hạn tại phân vị 99.5% và 0.5% để giảm tác động tiêu cực đến mô hình nhạy cảm như Logistic Regression.
    clip_bounds = {}
    for col in numeric_feature_cols:
        lower_bound = float(df_clean[col].quantile(0.001))
        upper_bound = float(df_clean[col].quantile(0.999))
        clip_bounds[col] = (lower_bound, upper_bound)
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"[3.1. Ngoại lai] Đã xử lý giới hạn biên an toàn lâm sàng cho {len(numeric_feature_cols)} biến liên tục.")

    return df_clean, imputation_dict, clip_bounds

def split_and_scale_data(df_clean, test_size=0.30, val_ratio_in_test=0.50, random_state=42):
    """
    3.2. BIẾN ĐỔI VÀ 3.3. PHÂN CHIA DỮ LIỆU
    - Tỷ lệ: 70% Train, 30% Test (trong đó 15% Validation, 15% Test)
    - Phân tầng (Stratified Split) theo nhãn NguyCoTieuDuong để đảm bảo tỷ lệ lớp
    - Chuẩn hóa đặc trưng bằng StandardScaler (fit trên tập Train, transform cho Val và Test)
    """
    print("\n" + "="*60)
    print("3.3. PHÂN CHIA VÀ CHUẨN HÓA DỮ LIỆU")
    print("="*60)

    feature_cols = [
        'Tuoi', 'GioiTinh', 'Glucose', 'HbA1C', 'Cholesterol', 'HDL', 'LDL',
        'Triglycerid', 'Creatinine', 'Ure', 'AcidUric', 'AST', 'ALT'
    ]
    target_col = 'NguyCoTieuDuong'

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Bước phân chia 1: Chia 70% Train và 30% Tạm (Temp = Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Bước phân chia 2: Chia 30% Tạm thành 15% Val và 15% Test (tỷ lệ 50-50 của tập Tạm)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_in_test,
        random_state=random_state,
        stratify=y_temp
    )

    total_samples = len(df_clean)
    print(f"Tổng số mẫu: {total_samples}")
    print(f"- Tập Huấn luyện (Train) : {len(X_train)} mẫu ({len(X_train)/total_samples*100:.1f}%) | Dương tính: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"- Tập Đánh giá (Val)    : {len(X_val)} mẫu ({len(X_val)/total_samples*100:.1f}%) | Dương tính: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
    print(f"- Tập Kiểm thử (Test)    : {len(X_test)} mẫu ({len(X_test)/total_samples*100:.1f}%) | Dương tính: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # 3.2. Chuẩn hóa dữ liệu (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled
    }

def run_preprocessing_pipeline():
    """Hàm chạy toàn bộ pipeline tiền xử lý và lưu artifacts"""
    # 1. Đọc dữ liệu
    df_raw = load_data()

    # 2. Làm sạch
    df_cleaned, imputation_dict, clip_bounds = clean_and_preprocess_data(df_raw)

    # Lưu tập dữ liệu đã làm sạch
    df_cleaned.to_csv("CSDL_TieuDuong_Cleaned.csv", index=False, encoding='utf-8')
    print("[Lưu dữ liệu] Đã lưu dữ liệu làm sạch tại 'CSDL_TieuDuong_Cleaned.csv'")

    # 3. Phân chia & chuẩn hóa
    split_data = split_and_scale_data(df_cleaned)

    # Lưu metadata tiền xử lý phục vụ dự đoán
    meta = {
        'feature_cols': split_data['feature_cols'],
        'imputation_dict': imputation_dict,
        'clip_bounds': clip_bounds,
        'scaler': split_data['scaler']
    }
    joblib.dump(meta, "preprocessing_meta.joblib")
    print("[Lưu mô hình] Đã lưu 'preprocessing_meta.joblib' chứa Scaler và các giá trị điền khuyết.")

    return split_data

if __name__ == "__main__":
    run_preprocessing_pipeline()
