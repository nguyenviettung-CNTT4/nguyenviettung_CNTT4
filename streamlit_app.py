"""
=============================================================================
BƯỚC 7 (GIAO DIỆN STREAMLIT): ỨNG DỤNG WEB DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Chạy ứng dụng: streamlit run streamlit_app.py
Triển khai: Streamlit Community Cloud (main file: streamlit_app.py hoặc main.py)
=============================================================================
"""

import os

import pandas as pd
import streamlit as st

from predict_engine import DiabetesRiskPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="AI Dự đoán nguy cơ tiểu đường",
    page_icon="🩺",
    layout="wide",
)

# Các trường nhập liệu: (khóa, nhãn, đơn vị, giá trị nhỏ nhất, lớn nhất, bước nhảy)
NUMERIC_FIELDS = [
    ("Tuoi", "Tuổi", "năm", 1.0, 120.0, 1.0),
    ("Glucose", "Đường huyết (Glucose)", "mmol/L", 0.0, 50.0, 0.1),
    ("HbA1C", "HbA1c", "%", 0.0, 20.0, 0.1),
    ("Cholesterol", "Cholesterol toàn phần", "mmol/L", 0.0, 25.0, 0.1),
    ("HDL", "HDL (Cholesterol tốt)", "mmol/L", 0.0, 10.0, 0.1),
    ("LDL", "LDL (Cholesterol xấu)", "mmol/L", 0.0, 15.0, 0.1),
    ("Triglycerid", "Triglycerid", "mmol/L", 0.0, 30.0, 0.1),
    ("Creatinine", "Creatinine", "µmol/L", 0.0, 1500.0, 1.0),
    ("Ure", "Ure máu", "mmol/L", 0.0, 60.0, 0.1),
    ("AcidUric", "Acid Uric", "µmol/L", 0.0, 1500.0, 1.0),
    ("AST", "Men gan AST (GOT)", "U/L", 0.0, 1000.0, 1.0),
    ("ALT", "Men gan ALT (GPT)", "U/L", 0.0, 1000.0, 1.0),
]

SAMPLES = {
    "Ca nguy cơ cao": {
        "Tuoi": 62, "GioiTinh": "Nữ", "Glucose": 11.9, "HbA1C": 8.0,
        "Cholesterol": 6.5, "HDL": 0.9, "LDL": 4.1, "Triglycerid": 3.2,
        "Creatinine": 95.0, "Ure": 8.2, "AcidUric": 440.0, "AST": 48.0, "ALT": 52.0,
    },
    "Ca tiền đái tháo đường": {
        "Tuoi": 52, "GioiTinh": "Nam", "Glucose": 6.2, "HbA1C": 6.0,
        "Cholesterol": 5.4, "HDL": 1.1, "LDL": 3.5, "Triglycerid": 2.2,
        "Creatinine": 88.0, "Ure": 5.8, "AcidUric": 390.0, "AST": 32.0, "ALT": 35.0,
    },
    "Ca bình thường": {
        "Tuoi": 28, "GioiTinh": "Nam", "Glucose": 4.8, "HbA1C": 5.1,
        "Cholesterol": 4.6, "HDL": 1.6, "LDL": 2.7, "Triglycerid": 1.1,
        "Creatinine": 75.0, "Ure": 4.5, "AcidUric": 310.0, "AST": 22.0, "ALT": 20.0,
    },
}

RISK_COLORS = {
    "red": "#d32f2f",
    "darkorange": "#ef6c00",
    "gold": "#f9a825",
    "green": "#2e7d32",
}


@st.cache_resource
def load_predictor():
    """Nạp mô hình một lần duy nhất (không huấn luyện lại mỗi lần tải trang)"""
    return DiabetesRiskPredictor(
        model_path=os.path.join(BASE_DIR, "best_model.joblib"),
        meta_path=os.path.join(BASE_DIR, "preprocessing_meta.joblib"),
    )


def load_sample(name):
    for key, value in SAMPLES[name].items():
        st.session_state[key] = value


def reset_form():
    for key, value in SAMPLES["Ca bình thường"].items():
        st.session_state[key] = value
    st.session_state.pop("result", None)


# Giá trị mặc định ban đầu cho form
for _key, _value in SAMPLES["Ca bình thường"].items():
    st.session_state.setdefault(_key, _value)

predictor = load_predictor()

st.title("🩺 Hệ thống AI dự đoán nguy cơ bệnh tiểu đường")
st.caption("Bài kiểm tra môn Trí Tuệ Nhân Tạo — dự đoán từ 13 thông số xét nghiệm sức khỏe")

if predictor.model is None:
    st.error("Không tìm thấy mô hình `best_model.joblib`. Hãy chạy `python main.py` để huấn luyện trước.")
    st.stop()

tab_predict, tab_eval = st.tabs(["🔍 Dự đoán", "📊 Đánh giá mô hình"])

with tab_predict:
    col_input, col_result = st.columns(2, gap="large")

    # ---------------- NỬA TRÁI: NHẬP THÔNG SỐ ----------------
    with col_input:
        with st.container(border=True):
            st.subheader("📝 Thông số xét nghiệm")
            st.caption("Nạp nhanh ca mẫu:")
            sample_cols = st.columns(len(SAMPLES) + 1)
            for col, name in zip(sample_cols, SAMPLES):
                col.button(name, on_click=load_sample, args=(name,), width="stretch")
            sample_cols[-1].button("↺ Nhập lại", on_click=reset_form, width="stretch")

            with st.form("patient_form", border=False):
                cols = st.columns(2)
                cols[0].selectbox("Giới tính", ["Nam", "Nữ"], key="GioiTinh")
                for i, (key, label, unit, vmin, vmax, step) in enumerate(NUMERIC_FIELDS, start=1):
                    cols[i % 2].number_input(
                        f"{label} ({unit})", min_value=vmin, max_value=vmax,
                        step=step, key=key, format="%.0f" if step >= 1 else "%.2f",
                    )
                submitted = st.form_submit_button("🚀 Dự đoán nguy cơ", type="primary", width="stretch")

        if submitted:
            inputs = {key: st.session_state[key] for key, *_ in NUMERIC_FIELDS}
            inputs["GioiTinh"] = st.session_state["GioiTinh"]
            try:
                st.session_state["result"] = predictor.predict_patient(inputs)
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"Lỗi khi dự đoán: {e}")

    # ---------------- NỬA PHẢI: KẾT QUẢ ----------------
    with col_result:
        with st.container(border=True):
            st.subheader("📋 Kết quả chẩn đoán")
            result = st.session_state.get("result")
            if not result:
                st.info("Nhập thông số ở bên trái (hoặc chọn ca mẫu) rồi bấm **🚀 Dự đoán nguy cơ** để xem kết quả.")
            else:
                color = RISK_COLORS.get(result["risk_color"], "#555")
                st.markdown(
                    f"<div style='padding:1rem 1.25rem;border-radius:0.5rem;border-left:6px solid {color};"
                    f"background:{color}1a'>"
                    f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{result['prediction_label']}</div>"
                    f"<div>Mức độ nguy cơ: <b>{result['risk_level']}</b></div></div>",
                    unsafe_allow_html=True,
                )
                st.metric("Xác suất mắc bệnh", f"{result['probability']:.2f}%")
                st.progress(min(max(result["probability"] / 100.0, 0.0), 1.0))

                st.markdown("##### ⚠️ Cảnh báo lâm sàng")
                for w in result["warnings"]:
                    if w.startswith("🚨"):
                        st.error(w)
                    elif w.startswith("⚠️"):
                        st.warning(w)
                    elif w.startswith("✅"):
                        st.success(w)
                    else:
                        st.info(w)

                st.markdown("##### 💡 Giải thích kết quả")
                st.markdown("\n".join(
                    f"- {line.strip().lstrip('•').strip()}" if line.startswith("  ") else f"\n**{line}**"
                    for line in result["explanations"]
                ))

                st.caption("⚕️ Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.")

with tab_eval:
    results_path = os.path.join(BASE_DIR, "ket_qua_danh_gia_mo_hinh.csv")
    if os.path.exists(results_path):
        st.subheader("Kết quả trên tập kiểm thử (Test set)")
        df_results = pd.read_csv(results_path).set_index("Model")
        st.dataframe(df_results.style.format("{:.4f}").highlight_max(axis=0), width="stretch")
        model_name = predictor.model_bundle.get("name") if isinstance(predictor.model_bundle, dict) else None
        if model_name:
            st.success(f"Mô hình đang dùng để dự đoán: **{model_name}**")

    charts = [
        ("01_so_sanh_hieu_nang_mo_hinh.png", "So sánh hiệu năng các mô hình"),
        ("02_duong_cong_roc.png", "Đường cong ROC"),
        ("03_ma_tran_nham_lan.png", "Ma trận nhầm lẫn"),
        ("04_do_quan_trong_dac_trung.png", "Độ quan trọng của đặc trưng"),
    ]
    chart_cols = st.columns(2)
    for i, (filename, caption) in enumerate(charts):
        path = os.path.join(BASE_DIR, "visualizations", filename)
        if os.path.exists(path):
            chart_cols[i % 2].image(path, caption=caption, width="stretch")
