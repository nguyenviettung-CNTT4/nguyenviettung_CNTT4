"""
=============================================================================
BƯỚC 7 (ENGINE): DỰ ĐOÁN, SINH CẢNH BÁO Y TẾ VÀ GIẢI THÍCH KẾT QUẢ
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Đề tài: Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe
File này chứa bộ máy suy luận (Inference & Reasoning Engine) tích hợp:
  - Dự đoán phân loại nhị phân và xác suất (%)
  - Sinh cảnh báo y khoa tự động dựa trên hướng dẫn ADA & Bộ Y tế
  - Giải thích nguyên nhân và đóng góp của từng chỉ số đối với kết quả
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

class DiabetesRiskPredictor:
    """Bộ máy dự đoán nguy cơ tiểu đường, sinh cảnh báo và giải thích"""

    FEATURE_COLS = [
        'Tuoi', 'GioiTinh', 'Glucose', 'HbA1C', 'Cholesterol', 'HDL', 'LDL',
        'Triglycerid', 'Creatinine', 'Ure', 'AcidUric', 'AST', 'ALT'
    ]

    FEATURE_NAMES_VI = {
        'Tuoi': 'Tuổi',
        'GioiTinh': 'Giới tính',
        'Glucose': 'Đường huyết (Glucose)',
        'HbA1C': 'Chỉ số HbA1c',
        'Cholesterol': 'Cholesterol toàn phần',
        'HDL': 'Cholesterol tốt (HDL)',
        'LDL': 'Cholesterol xấu (LDL)',
        'Triglycerid': 'Chất béo trung tính (Triglycerid)',
        'Creatinine': 'Chức năng thận (Creatinine)',
        'Ure': 'Ure máu',
        'AcidUric': 'Acid Uric máu',
        'AST': 'Men gan AST (GOT)',
        'ALT': 'Men gan ALT (GPT)'
    }

    NORMAL_RANGES = {
        'Glucose': {'min': 3.9, 'max': 5.6, 'unit': 'mmol/L', 'desc': 'Đường huyết lúc đói chuẩn: 3.9 - 5.6 mmol/L'},
        'HbA1C': {'min': 4.0, 'max': 5.6, 'unit': '%', 'desc': 'HbA1c bình thường: < 5.7%'},
        'Cholesterol': {'min': 2.8, 'max': 5.2, 'unit': 'mmol/L', 'desc': 'Cholesterol toàn phần: < 5.2 mmol/L'},
        'HDL': {'min': 1.0, 'max': 2.5, 'unit': 'mmol/L', 'desc': 'HDL bảo vệ tim mạch: > 1.0 mmol/L (nam), > 1.2 mmol/L (nữ)'},
        'LDL': {'min': 0.0, 'max': 3.4, 'unit': 'mmol/L', 'desc': 'LDL tối ưu: < 3.4 mmol/L'},
        'Triglycerid': {'min': 0.0, 'max': 1.7, 'unit': 'mmol/L', 'desc': 'Triglycerid bình thường: < 1.7 mmol/L'},
        'Creatinine': {'min': 53.0, 'max': 106.0, 'unit': 'µmol/L', 'desc': 'Creatinine bình thường: 53 - 106 µmol/L'},
        'Ure': {'min': 2.5, 'max': 7.5, 'unit': 'mmol/L', 'desc': 'Ure bình thường: 2.5 - 7.5 mmol/L'},
        'AcidUric': {'min': 150.0, 'max': 420.0, 'unit': 'µmol/L', 'desc': 'Acid Uric bình thường: 150 - 420 µmol/L'},
        'AST': {'min': 0.0, 'max': 40.0, 'unit': 'U/L', 'desc': 'Men gan AST: < 40 U/L'},
        'ALT': {'min': 0.0, 'max': 40.0, 'unit': 'U/L', 'desc': 'Men gan ALT: < 40 U/L'}
    }

    def __init__(self, model_path="best_model.joblib", meta_path="preprocessing_meta.joblib"):
        """Tải mô hình và thông số tiền xử lý"""
        self.model_bundle = None
        self.meta = None
        self.model = None
        self.scaler = None
        self.use_scaled = False

        if os.path.exists(model_path):
            self.model_bundle = joblib.load(model_path)
            self.model = self.model_bundle['model']
            self.use_scaled = self.model_bundle.get('use_scaled', False)
        
        if os.path.exists(meta_path):
            self.meta = joblib.load(meta_path)
            self.scaler = self.meta.get('scaler', None)

    def predict_patient(self, input_data):
        """
        Dự đoán nguy cơ cho 1 người bệnh.
        input_data: dict chứa 13 trường hoặc list tương ứng
        """
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp hoặc chưa huấn luyện. Hãy chạy train_and_evaluate.py trước.")

        # Chuẩn bị DataFrame 1 dòng
        df_input = pd.DataFrame([input_data])

        # Đảm bảo có đủ 13 cột
        for col in self.FEATURE_COLS:
            if col not in df_input.columns:
                if self.meta and 'imputation_dict' in self.meta:
                    df_input[col] = self.meta['imputation_dict'].get(col, 0.0)
                else:
                    df_input[col] = 0.0

        # Thứ tự chuẩn
        X = df_input[self.FEATURE_COLS].copy()

        # Tiền xử lý: Gender số hóa nếu dạng chuỗi
        # (pandas >= 3.0 dùng kiểu 'str' thay cho 'object' nên kiểm tra theo kiểu số)
        if not pd.api.types.is_numeric_dtype(X['GioiTinh']):
            g_str = str(X['GioiTinh'].iloc[0]).strip().lower()
            X['GioiTinh'] = 1 if g_str in ('nam', 'male', '1', '1.0') else 0

        # Nếu Glucose > 50 và người dùng chưa đổi đơn vị, tự động quy đổi từ mg/dL sang mmol/L
        glucose_val = float(X['Glucose'].iloc[0])
        glucose_converted = False
        if glucose_val > 50.0:
            glucose_val = glucose_val / 18.0182
            X['Glucose'] = glucose_val
            glucose_converted = True

        # Dự đoán
        if self.use_scaled and self.scaler is not None:
            X_infer = self.scaler.transform(X)
        else:
            X_infer = X

        pred_class = int(self.model.predict(X_infer)[0])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_infer)[0]
            prob_risk = float(probs[1]) * 100.0
        else:
            prob_risk = 100.0 if pred_class == 1 else 0.0

        # Phân loại mức độ nguy cơ
        if prob_risk >= 80.0:
            risk_level = "RẤT CAO"
            risk_color = "red"
        elif prob_risk >= 50.0:
            risk_level = "CAO"
            risk_color = "darkorange"
        elif prob_risk >= 25.0:
            risk_level = "TRUNG BÌNH (Tiền đái tháo đường)"
            risk_color = "gold"
        else:
            risk_level = "THẤP (Bình thường)"
            risk_color = "green"

        # Sinh cảnh báo lâm sàng
        warnings_list = self._generate_clinical_warnings(X.iloc[0], glucose_converted)

        # Giải thích kết quả
        explanations_list = self._explain_prediction(X.iloc[0], prob_risk, pred_class)

        return {
            'prediction': pred_class,
            'prediction_label': 'CÓ NGUY CƠ MẮC TIỂU ĐƯỜNG' if pred_class == 1 else 'KHÔNG CÓ NGUY CƠ TIỂU ĐƯỜNG',
            'probability': prob_risk,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'warnings': warnings_list,
            'explanations': explanations_list,
            'processed_inputs': X.iloc[0].to_dict()
        }

    def _generate_clinical_warnings(self, row, glucose_converted=False):
        """Sinh các cảnh báo y khoa dựa trên các chỉ số thực tế"""
        warnings = []

        glucose = float(row['Glucose'])
        hba1c = float(row['HbA1C'])
        chol = float(row['Cholesterol'])
        trig = float(row['Triglycerid'])
        ldl = float(row['LDL'])
        hdl = float(row['HDL'])
        is_male = (int(row['GioiTinh']) == 1)
        creat = float(row['Creatinine'])
        ast = float(row['AST'])
        alt = float(row['ALT'])
        age = float(row['Tuoi'])

        # Cảnh báo Glucose
        if glucose_converted:
            warnings.append(f"ℹ️ Đã tự động quy đổi chỉ số Glucose từ đơn vị mg/dL sang {glucose:.2f} mmol/L chuẩn quốc tế.")

        if glucose >= 7.0:
            warnings.append(f"🚨 ĐƯỜNG HUYẾT NGUY HIỂM: Glucose = {glucose:.2f} mmol/L (ngưỡng chẩn đoán ĐTĐ ≥ 7.0 mmol/L lúc đói). Nguy cơ tổn thương mạch máu cấp tính.")
        elif glucose >= 5.6:
            warnings.append(f"⚠️ RỐI LOẠN DUNG NẠP GLUCOSE: Glucose = {glucose:.2f} mmol/L (nằm trong ngưỡng tiền đái tháo đường 5.6 - 6.9 mmol/L).")
        elif glucose < 3.9:
            warnings.append(f"⚠️ HẠ ĐƯỜNG HUYẾT: Glucose = {glucose:.2f} mmol/L (< 3.9 mmol/L). Cần chú ý nguy cơ choáng ngất do thiếu đường não.")

        # Cảnh báo HbA1C
        if hba1c >= 6.5:
            warnings.append(f"🚨 CHỈ SỐ HbA1c VƯỢT NGƯỠNG: HbA1c = {hba1c:.2f}% (ngưỡng xác định đái tháo đường theo ADA ≥ 6.5%). Thể hiện đường huyết trung bình 3 tháng qua tăng cao.")
        elif hba1c >= 5.7:
            warnings.append(f"⚠️ TIỀN ĐÁI THÁO ĐƯỜNG THEO HbA1c: HbA1c = {hba1c:.2f}% (vùng tiền bệnh 5.7% - 6.4%). Tốc độ chuyển thành tiểu đường thực thụ cao nếu không can thiệp.")

        # Cảnh báo Rối loạn mỡ máu (Lipid profile)
        lipid_issues = []
        if chol > 5.2:
            lipid_issues.append(f"Cholesterol toàn phần cao ({chol:.2f} mmol/L)")
        if trig > 1.7:
            lipid_issues.append(f"Triglycerid cao ({trig:.2f} mmol/L - dấu hiệu kháng insulin mạnh)")
        if ldl > 3.4:
            lipid_issues.append(f"LDL-C xấu cao ({ldl:.2f} mmol/L)")
        hdl_limit = 1.0 if is_male else 1.2
        if hdl < hdl_limit:
            lipid_issues.append(f"HDL-C tốt giảm thấp ({hdl:.2f} mmol/L)")

        if lipid_issues:
            warnings.append("⚠️ RỐI LOẠN CHUYỂN HÓA LIPID (MỠ MÁU): " + ", ".join(lipid_issues) + ". Nguy cơ xơ vữa động mạch và bệnh tim mạch đồng mắc.")

        # Cảnh báo Gan & Thận
        creat_max = 106.0 if is_male else 88.0
        if creat > creat_max:
            warnings.append(f"⚠️ SUY GIẢM CHỨC NĂNG THẬN: Creatinine = {creat:.1f} µmol/L (vượt ngưỡng {creat_max} µmol/L). Cảnh báo nguy cơ biến chứng thận do đái tháo đường.")

        if ast > 40.0 or alt > 40.0:
            warnings.append(f"⚠️ TỔN THƯƠNG MEN GAN: AST = {ast:.1f} U/L, ALT = {alt:.1f} U/L. Gợi ý nguy cơ gan nhiễm mỡ liên quan chuyển hóa (MASLD).")

        # Yếu tố Tuổi
        if age >= 45:
            warnings.append(f"ℹ️ YẾU TỐ NGUY CƠ TUỔI TÁC: Bệnh nhân {int(age)} tuổi (≥ 45 tuổi là nhóm nguy cơ cao cần sàng lọc định kỳ hàng năm).")

        if not warnings:
            warnings.append("✅ Tất cả các chỉ số xét nghiệm hiện tại đều nằm trong giới hạn tham chiếu an toàn lâm sàng.")

        return warnings

    def _explain_prediction(self, row, prob_risk, pred_class):
        """Giải thích chi tiết tại sao mô hình đưa ra kết luận này"""
        explanations = []

        glucose = float(row['Glucose'])
        hba1c = float(row['HbA1C'])
        trig = float(row['Triglycerid'])
        age = float(row['Tuoi'])

        # Phân tích nguyên nhân chính
        key_factors = []
        protective_factors = []

        if glucose >= 7.0:
            key_factors.append(f"Nồng độ Glucose ({glucose:.2f} mmol/L) là yếu tố quyết định mạnh nhất kéo nguy cơ tăng vọt.")
        elif glucose >= 5.6:
            key_factors.append(f"Nồng độ Glucose ({glucose:.2f} mmol/L) ở mức ranh giới làm tăng xác suất rủi ro.")
        else:
            protective_factors.append(f"Nồng độ Glucose tốt ({glucose:.2f} mmol/L) đóng vai trò kéo giảm mạnh nguy cơ.")

        if hba1c >= 6.5:
            key_factors.append(f"Chỉ số HbA1c ({hba1c:.2f}%) phản ánh tình trạng đường huyết tích lũy bất thường mạn tính.")
        elif hba1c < 5.7:
            protective_factors.append(f"Chỉ số HbA1c an toàn ({hba1c:.2f}%) khẳng định không có tăng đường huyết kéo dài.")

        if trig > 2.0:
            key_factors.append(f"Triglycerid cao ({trig:.2f} mmol/L) kích hoạt hội chứng kháng insulin tại tế bào.")

        if age >= 55:
            key_factors.append(f"Độ tuổi cao ({int(age)} tuổi) làm suy giảm chức năng tiết insulin của tế bào beta đảo tụy.")

        if pred_class == 1:
            explanations.append(f"Mô hình Trí Tuệ Nhân Tạo tính toán xác suất mắc bệnh là {prob_risk:.1f}%.")
            if key_factors:
                explanations.append("CÁC YẾU TỐ ĐÓNG GÓP CHÍNH DẪN ĐẾN KẾT LUẬN NGUY CƠ CAO:")
                for kf in key_factors:
                    explanations.append(f"  • {kf}")
            explanations.append("KHUYẾN NGHỊ XỬ TRÍ Y TẾ:")
            explanations.append("  1. Cần đến ngay cơ sở y tế chuyên khoa Nội tiết để làm nghiệm pháp dung nạp Glucose (OGTT).")
            explanations.append("  2. Cắt giảm tinh bột hấp thu nhanh (đường tinh luyện, nước ngọt, bánh mì trắng).")
            explanations.append("  3. Tăng cường vận động thể lực tối thiểu 30 phút/ngày, 5 ngày/tuần.")
        else:
            explanations.append(f"Mô hình Trí Tuệ Nhân Tạo ước tính xác suất mắc bệnh rất thấp (chỉ {prob_risk:.1f}%).")
            if protective_factors:
                explanations.append("CÁC CHỈ SỐ BẢO VỆ TÍCH CỰC:")
                for pf in protective_factors:
                    explanations.append(f"  • {pf}")
            explanations.append("KHUYẾN NGHỊ DUY TRÌ SỨC KHỎE:")
            explanations.append("  1. Tiếp tục duy trì chế độ dinh dưỡng lành mạnh giàu rau xanh và chất xơ.")
            explanations.append("  2. Khám sức khỏe tổng quát và xét nghiệm đường huyết định kỳ 6 - 12 tháng/lần.")

        return explanations

# Kiểm thử nhanh engine
if __name__ == "__main__":
    predictor = DiabetesRiskPredictor()
    sample_patient = {
        'Tuoi': 58, 'GioiTinh': 'Nam', 'Glucose': 9.5, 'HbA1C': 7.8,
        'Cholesterol': 6.2, 'HDL': 0.9, 'LDL': 3.9, 'Triglycerid': 2.8,
        'Creatinine': 92.0, 'Ure': 6.5, 'AcidUric': 410.0, 'AST': 35.0, 'ALT': 38.0
    }
    res = predictor.predict_patient(sample_patient)
    print("KẾT QUẢ DỰ ĐOÁN THỬ NGHIỆM:")
    print("Phân loại:", res['prediction_label'])
    print("Xác suất:", f"{res['probability']:.2f}%")
    print("Cảnh báo:\n", "\n".join(res['warnings']))
    print("Giải thích:\n", "\n".join(res['explanations']))
