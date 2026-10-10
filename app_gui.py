"""
=============================================================================
BƯỚC 7 (GIAO DIỆN DESKTOP GUI): PHẦN MỀM DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Chức năng:
  - Nhập thông tin và các chỉ số xét nghiệm sinh hóa của người bệnh
  - Chuyển đổi đơn vị linh hoạt (mmol/L, mg/dL)
  - Nút nạp nhanh ca bệnh mẫu (Ca nguy cơ cao, Ca an toàn, Tiền đái tháo đường)
  - Dự đoán nguy cơ thời gian thực bằng mô hình AI tối ưu nhất
  - Sinh cảnh báo lâm sàng y tế tự động (ADA & Bộ Y tế)
  - Giải thích nguyên nhân dự đoán và đưa ra khuyến nghị y khoa
  - Xuất phiếu khám bệnh nhân ra file text
=============================================================================
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox, filedialog

# Đảm bảo in tiếng Việt
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Sử dụng ttkbootstrap nếu có, nếu không thì dùng ttk chuẩn
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    from ttkbootstrap.scrolled import ScrolledText
    USE_TTKBOOTSTRAP = True
except ImportError:
    import tkinter.ttk as tb
    from tkinter import ttk
    USE_TTKBOOTSTRAP = False

from predict_engine import DiabetesRiskPredictor

class DiabetesAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG AI DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG - MÔN TRÍ TUỆ NHÂN TẠO")
        self.root.geometry("1100x820")
        self.root.minsize(980, 750)

        # Nạp mô hình dự đoán
        try:
            self.predictor = DiabetesRiskPredictor()
        except Exception as e:
            messagebox.showerror("Lỗi khởi tạo", f"Không thể nạp mô hình: {e}\nVui lòng chạy train_and_evaluate.py trước!")
            self.predictor = None

        self.last_result = None
        self._build_ui()

    def _build_ui(self):
        # 1. Header Banner
        if USE_TTKBOOTSTRAP:
            header_frame = tb.Frame(self.root, bootstyle="primary")
            header_frame.pack(fill="x", padx=0, pady=0)
            title_lbl = tb.Label(
                header_frame,
                text="HỆ THỐNG AI CHẨN ĐOÁN & DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG",
                font=("Segoe UI", 16, "bold"),
                bootstyle="inverse-primary"
            )
            title_lbl.pack(pady=(12, 2))
            subtitle_lbl = tb.Label(
                header_frame,
                text="Môn học: Trí Tuệ Nhân Tạo | Hỗ trợ lâm sàng từ các xét nghiệm sinh hóa máu",
                font=("Segoe UI", 10),
                bootstyle="inverse-primary"
            )
            subtitle_lbl.pack(pady=(0, 12))
        else:
            header_frame = tk.Frame(self.root, bg="#1a365d")
            header_frame.pack(fill="x")
            title_lbl = tk.Label(header_frame, text="HỆ THỐNG AI CHẨN ĐOÁN & DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG",
                                 font=("Segoe UI", 16, "bold"), fg="white", bg="#1a365d")
            title_lbl.pack(pady=(12, 2))
            subtitle_lbl = tk.Label(header_frame, text="Môn học: Trí Tuệ Nhân Tạo | Hỗ trợ lâm sàng",
                                    font=("Segoe UI", 10), fg="#e2e8f0", bg="#1a365d")
            subtitle_lbl.pack(pady=(0, 12))

        # Main Content Frame (Split 2 Cột: Trái = Form Nhập, Phải = Kết quả & Cảnh báo)
        content_frame = tb.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # ===================================================================
        # CỘT TRÁI: FORM NHẬP THÔNG TIN NGƯỜI BỆNH
        # ===================================================================
        left_frame = tb.Labelframe(content_frame, text=" 📝 THÔNG TIN NGƯỜI BỆNH & CHỈ SỐ XÉT NGHIỆM ", padding=12)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 8))

        # Khung chứa các input
        inputs_grid = tb.Frame(left_frame)
        inputs_grid.pack(fill="x", expand=False)

        self.entries = {}

        fields = [
            ("Tuổi (năm):", "Tuoi", "45", "Nhập tuổi (1 - 105)"),
            ("Giới tính:", "GioiTinh", "Nam", "Chọn Nam hoặc Nữ"),
            ("Glucose máu (mmol/L):", "Glucose", "5.6", "Bình thường 3.9 - 5.6 mmol/L"),
            ("Chỉ số HbA1c (%):", "HbA1C", "5.5", "Bình thường < 5.7%"),
            ("Cholesterol TP (mmol/L):", "Cholesterol", "5.0", "Mục tiêu < 5.2 mmol/L"),
            ("HDL-C (mmol/L):", "HDL", "1.4", "Tốt: > 1.0 (nam), > 1.2 (nữ)"),
            ("LDL-C (mmol/L):", "LDL", "3.0", "Mục tiêu: < 3.4 mmol/L"),
            ("Triglycerid (mmol/L):", "Triglycerid", "1.6", "Mục tiêu: < 1.7 mmol/L"),
            ("Creatinine (µmol/L):", "Creatinine", "80.0", "Nam 62-115, Nữ 44-88"),
            ("Ure máu (mmol/L):", "Ure", "5.2", "Bình thường: 2.5 - 7.5"),
            ("Acid Uric (µmol/L):", "AcidUric", "350.0", "Bình thường: 150 - 420"),
            ("Men gan AST (U/L):", "AST", "25.0", "Bình thường: < 40 U/L"),
            ("Men gan ALT (U/L):", "ALT", "22.0", "Bình thường: < 40 U/L")
        ]

        for i, (label_text, key, default_val, tooltip) in enumerate(fields):
            row_idx = i
            lbl = tb.Label(inputs_grid, text=label_text, font=("Segoe UI", 9, "bold"))
            lbl.grid(row=row_idx, column=0, sticky="w", pady=3, padx=(0, 8))

            if key == "GioiTinh":
                var = tk.StringVar(value=default_val)
                combo = tb.Combobox(inputs_grid, textvariable=var, values=["Nam", "Nữ"], state="readonly", width=18)
                combo.grid(row=row_idx, column=1, sticky="w", pady=3)
                self.entries[key] = var
            else:
                var = tk.StringVar(value=default_val)
                ent = tb.Entry(inputs_grid, textvariable=var, width=20)
                ent.grid(row=row_idx, column=1, sticky="w", pady=3)
                self.entries[key] = var

        # Khung Nút Nạp Mẫu Nhanh
        sample_frame = tb.Labelframe(left_frame, text="⚡ Nạp dữ liệu ca bệnh mẫu thử nghiệm", padding=8)
        sample_frame.pack(fill="x", pady=(10, 8))

        btn_case1 = tb.Button(
            sample_frame,
            text="Ca 1: Nguy cơ cao (Tiểu đường)",
            bootstyle="outline-danger",
            command=self._load_sample_diabetes
        )
        btn_case1.pack(fill="x", pady=2)

        btn_case2 = tb.Button(
            sample_frame,
            text="Ca 2: Người khỏe mạnh (Âm tính)",
            bootstyle="outline-success",
            command=self._load_sample_healthy
        )
        btn_case2.pack(fill="x", pady=2)

        btn_case3 = tb.Button(
            sample_frame,
            text="Ca 3: Tiền đái tháo đường (Ranh giới)",
            bootstyle="outline-warning",
            command=self._load_sample_prediabetes
        )
        btn_case3.pack(fill="x", pady=2)

        # Khung Nút Tác Vụ Chính
        action_frame = tb.Frame(left_frame)
        action_frame.pack(fill="x", pady=(5, 0))

        btn_predict = tb.Button(
            action_frame,
            text="🚀 DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG",
            bootstyle="success",
            command=self._handle_predict
        )
        btn_predict.pack(fill="x", pady=4, ipady=6)

        btn_reset = tb.Button(
            action_frame,
            text="🔄 Xóa trắng / Làm mới",
            bootstyle="secondary",
            command=self._reset_form
        )
        btn_reset.pack(fill="x", pady=2)

        # ===================================================================
        # CỘT PHẢI: KẾT QUẢ DỰ ĐOÁN, CẢNH BÁO VÀ GIẢI THÍCH
        # ===================================================================
        right_frame = tb.Labelframe(content_frame, text=" 📊 KẾT QUẢ DỰ ĐOÁN, CẢNH BÁO & GIẢI THÍCH ", padding=12)
        right_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # Khung Thẻ Kết Quả Nổi Bật (Card Result)
        self.result_card = tb.Frame(right_frame, padding=12)
        self.result_card.pack(fill="x", pady=(0, 10))

        self.lbl_card_title = tb.Label(
            self.result_card,
            text="CHƯA CÓ KẾT QUẢ DỰ ĐOÁN",
            font=("Segoe UI", 14, "bold"),
            bootstyle="secondary"
        )
        self.lbl_card_title.pack(anchor="center")

        self.lbl_prob = tb.Label(
            self.result_card,
            text="Vui lòng nhấn nút 'DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG' sau khi điền thông số",
            font=("Segoe UI", 11)
        )
        self.lbl_prob.pack(anchor="center", pady=4)

        if USE_TTKBOOTSTRAP:
            self.progress_risk = tb.Progressbar(self.result_card, bootstyle="secondary", maximum=100, value=0)
            self.progress_risk.pack(fill="x", padx=20, pady=6)
        else:
            self.progress_risk = None

        # Khung Tab hoặc Vùng hiển thị Cảnh báo & Giải thích
        notebook = tb.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Cảnh báo y tế
        tab_warnings = tb.Frame(notebook, padding=8)
        notebook.add(tab_warnings, text=" 🚨 Cảnh Báo Lâm Sàng Y Tế ")

        self.txt_warnings = ScrolledText(tab_warnings, wrap="word", font=("Segoe UI", 10)) if USE_TTKBOOTSTRAP else tk.Text(tab_warnings, wrap="word")
        self.txt_warnings.pack(fill="both", expand=True)

        # Tab 2: Giải thích kết quả
        tab_explain = tb.Frame(notebook, padding=8)
        notebook.add(tab_explain, text=" 🔍 Giải Thích & Khuyến Nghị AI ")

        self.txt_explain = ScrolledText(tab_explain, wrap="word", font=("Segoe UI", 10)) if USE_TTKBOOTSTRAP else tk.Text(tab_explain, wrap="word")
        self.txt_explain.pack(fill="both", expand=True)

        # Tab 3: Bảng giá trị chuẩn y khoa tham khảo
        tab_ref = tb.Frame(notebook, padding=8)
        notebook.add(tab_ref, text=" 📋 Chỉ Số Tham Chiếu Y Khoa ")

        self.txt_ref = ScrolledText(tab_ref, wrap="word", font=("Segoe UI", 9)) if USE_TTKBOOTSTRAP else tk.Text(tab_ref, wrap="word")
        self.txt_ref.pack(fill="both", expand=True)
        self._populate_reference_table()

        # Nút Xuất Phiếu Khám
        footer_frame = tb.Frame(right_frame)
        footer_frame.pack(fill="x", pady=(8, 0))

        btn_export = tb.Button(
            footer_frame,
            text="💾 Xuất Phiếu Kết Quả Khám Bệnh (.txt)",
            bootstyle="outline-primary",
            command=self._export_report
        )
        btn_export.pack(side="right")

    def _populate_reference_table(self):
        ref_text = (
            "BẢNG CHỈ SỐ THAM CHIẾU SINH HÓA MÁU THEO TIÊU CHUẨN ADA VÀ BỘ Y TẾ:\n"
            "=" * 70 + "\n\n"
            "1. Glucose huyết tương lúc đói:\n"
            "   - Bình thường           : 3.9 - 5.5 mmol/L (70 - 99 mg/dL)\n"
            "   - Tiền đái tháo đường    : 5.6 - 6.9 mmol/L (100 - 125 mg/dL)\n"
            "   - Đái tháo đường         : ≥ 7.0 mmol/L (≥ 126 mg/dL)\n\n"
            "2. Chỉ số HbA1c (Đường huyết trung bình 3 tháng):\n"
            "   - Bình thường           : < 5.7%\n"
            "   - Tiền đái tháo đường    : 5.7% - 6.4%\n"
            "   - Đái tháo đường         : ≥ 6.5%\n\n"
            "3. Bilan Lipid Máu:\n"
            "   - Cholesterol toàn phần  : < 5.2 mmol/L (< 200 mg/dL)\n"
            "   - Triglycerid           : < 1.7 mmol/L (< 150 mg/dL)\n"
            "   - LDL-Cholesterol (xấu)  : < 3.4 mmol/L (< 130 mg/dL)\n"
            "   - HDL-Cholesterol (tốt)  : > 1.0 mmol/L (Nam), > 1.2 mmol/L (Nữ)\n\n"
            "4. Chức năng Thận & Chuyển hóa:\n"
            "   - Creatinine             : Nam 62 - 115 µmol/L | Nữ 44 - 88 µmol/L\n"
            "   - Ure                    : 2.5 - 7.5 mmol/L\n"
            "   - Acid Uric              : Nam 150 - 420 µmol/L | Nữ 150 - 360 µmol/L\n\n"
            "5. Men gan (Tổn thương gan liên quan đái tháo đường):\n"
            "   - AST (GOT)              : < 40 U/L\n"
            "   - ALT (GPT)              : < 40 U/L\n"
        )
        self.txt_ref.insert("1.0", ref_text)
        self.txt_ref.configure(state="disabled")

    def _load_sample_diabetes(self):
        """Ca bệnh nguy cơ cao mắc bệnh tiểu đường"""
        sample = {
            "Tuoi": "62", "GioiTinh": "Nữ", "Glucose": "11.9", "HbA1C": "8.0",
            "Cholesterol": "6.5", "HDL": "0.9", "LDL": "4.1", "Triglycerid": "3.2",
            "Creatinine": "95.0", "Ure": "8.2", "AcidUric": "440.0", "AST": "48.0", "ALT": "52.0"
        }
        for k, v in sample.items():
            self.entries[k].set(v)

    def _load_sample_healthy(self):
        """Ca người bình thường, khỏe mạnh"""
        sample = {
            "Tuoi": "28", "GioiTinh": "Nam", "Glucose": "4.8", "HbA1C": "5.1",
            "Cholesterol": "4.6", "HDL": "1.6", "LDL": "2.7", "Triglycerid": "1.1",
            "Creatinine": "75.0", "Ure": "4.5", "AcidUric": "310.0", "AST": "22.0", "ALT": "20.0"
        }
        for k, v in sample.items():
            self.entries[k].set(v)

    def _load_sample_prediabetes(self):
        """Ca ranh giới, tiền đái tháo đường"""
        sample = {
            "Tuoi": "52", "GioiTinh": "Nam", "Glucose": "6.2", "HbA1C": "6.0",
            "Cholesterol": "5.4", "HDL": "1.1", "LDL": "3.5", "Triglycerid": "2.2",
            "Creatinine": "88.0", "Ure": "5.8", "AcidUric": "390.0", "AST": "32.0", "ALT": "35.0"
        }
        for k, v in sample.items():
            self.entries[k].set(v)

    def _reset_form(self):
        for k in self.entries:
            if k == "GioiTinh":
                self.entries[k].set("Nam")
            else:
                self.entries[k].set("")
        self.lbl_card_title.configure(text="CHƯA CÓ KẾT QUẢ DỰ ĐOÁN", bootstyle="secondary" if USE_TTKBOOTSTRAP else None)
        self.lbl_prob.configure(text="Vui lòng nhấn nút 'DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG' sau khi điền thông số")
        if self.progress_risk:
            self.progress_risk.configure(value=0, bootstyle="secondary")
        self.txt_warnings.delete("1.0", tk.END)
        self.txt_explain.delete("1.0", tk.END)
        self.last_result = None

    def _handle_predict(self):
        if not self.predictor:
            messagebox.showerror("Lỗi", "Mô hình AI chưa sẵn sàng.")
            return

        # Đọc dữ liệu từ form
        patient_data = {}
        try:
            for k, var in self.entries.items():
                val_str = var.get().strip()
                if not val_str:
                    messagebox.showwarning("Thiếu dữ liệu", f"Vui lòng nhập giá trị cho trường: {k}")
                    return
                if k == "GioiTinh":
                    patient_data[k] = val_str
                else:
                    patient_data[k] = float(val_str)
        except ValueError as ve:
            messagebox.showerror("Sai định dạng", f"Dữ liệu số không hợp lệ. Vui lòng kiểm tra lại các trường số.\nChi tiết: {ve}")
            return

        # Thực hiện dự đoán
        try:
            res = self.predictor.predict_patient(patient_data)
            self.last_result = res
        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", f"Quá trình dự đoán gặp lỗi: {e}")
            return

        # Cập nhật Card kết quả
        prob = res['probability']
        label = res['prediction_label']
        level = res['risk_level']

        if res['prediction'] == 1:
            card_style = "danger"
            card_title = f"🚨 {label.upper()}"
        elif prob >= 25.0:
            card_style = "warning"
            card_title = f"⚠️ CẢNH BÁO TIỀN ĐÁI THÁO ĐƯỜNG"
        else:
            card_style = "success"
            card_title = f"✅ NGUY CƠ THẤP - KẾT QUẢ AN TOÀN"

        if USE_TTKBOOTSTRAP:
            self.lbl_card_title.configure(text=card_title, bootstyle=card_style)
            self.lbl_prob.configure(text=f"Mức độ rủi ro: {level} | Xác suất nguy cơ: {prob:.1f}%")
            if self.progress_risk:
                self.progress_risk.configure(value=prob, bootstyle=card_style)
        else:
            self.lbl_card_title.configure(text=card_title)
            self.lbl_prob.configure(text=f"Mức độ rủi ro: {level} | Xác suất nguy cơ: {prob:.1f}%")

        # Cập nhật Tab Cảnh báo
        self.txt_warnings.delete("1.0", tk.END)
        for w in res['warnings']:
            self.txt_warnings.insert(tk.END, f"{w}\n\n")

        # Cập nhật Tab Giải thích
        self.txt_explain.delete("1.0", tk.END)
        for e in res['explanations']:
            self.txt_explain.insert(tk.END, f"{e}\n")

    def _export_report(self):
        if not self.last_result:
            messagebox.showinfo("Thông báo", "Chưa có kết quả dự đoán để xuất phiếu.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("Tất cả tập tin", "*.*")],
            title="Lưu Phiếu Kết Quả Chẩn Đoán",
            initialfile="Phieu_Ket_Qua_Tieu_Duong.txt"
        )
        if not file_path:
            return

        res = self.last_result
        inputs = res['processed_inputs']

        report_content = (
            "=======================================================================\n"
            "            PHIẾU KẾT QUẢ ĐÁNH GIÁ NGUY CƠ TIỂU ĐƯỜNG (AI)\n"
            "            Môn: Trí Tuệ Nhân Tạo - Hỗ trợ lâm sàng y khoa\n"
            "=======================================================================\n\n"
            "1. THÔNG TIN XÉT NGHIỆM ĐẦU VÀO:\n"
            f"   - Tuổi: {int(inputs['Tuoi'])} tuổi\n"
            f"   - Giới tính: {'Nam' if inputs['GioiTinh'] == 1 else 'Nữ'}\n"
            f"   - Glucose: {inputs['Glucose']:.2f} mmol/L\n"
            f"   - HbA1c: {inputs['HbA1C']:.2f} %\n"
            f"   - Cholesterol toàn phần: {inputs['Cholesterol']:.2f} mmol/L\n"
            f"   - HDL-C: {inputs['HDL']:.2f} mmol/L | LDL-C: {inputs['LDL']:.2f} mmol/L\n"
            f"   - Triglycerid: {inputs['Triglycerid']:.2f} mmol/L\n"
            f"   - Creatinine: {inputs['Creatinine']:.1f} µmol/L | Ure: {inputs['Ure']:.1f} mmol/L\n"
            f"   - Acid Uric: {inputs['AcidUric']:.1f} µmol/L\n"
            f"   - Men gan: AST={inputs['AST']:.1f} U/L | ALT={inputs['ALT']:.1f} U/L\n\n"
            "2. KẾT QUẢ DỰ ĐOÁN CỦA MÔ HÌNH HỌC MÁY:\n"
            f"   - Kết luận: {res['prediction_label']}\n"
            f"   - Phân cấp rủi ro: {res['risk_level']}\n"
            f"   - Xác suất dự đoán: {res['probability']:.2f} %\n\n"
            "3. CẢNH BÁO LÂM SÀNG Y TẾ:\n" +
            "\n".join([f"   * {w}" for w in res['warnings']]) + "\n\n"
            "4. GIẢI THÍCH NGUYÊN NHÂN VÀ KHUYẾN NGHỊ:\n" +
            "\n".join([f"   {e}" for e in res['explanations']]) + "\n\n"
            "=======================================================================\n"
            "Lưu ý: Kết quả trên là trợ lý AI phục vụ tham khảo lâm sàng, không thay thế\n"
            "hoàn toàn ý kiến chuyên môn của bác sĩ chuyên khoa Nội tiết.\n"
            "=======================================================================\n"
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            messagebox.showinfo("Thành công", f"Đã xuất phiếu khám thành công vào:\n{file_path}")
        except Exception as ex:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {ex}")

def main():
    if USE_TTKBOOTSTRAP:
        app = tb.Window(themename="cosmo")
    else:
        app = tk.Tk()
    DiabetesAppGUI(app)
    app.mainloop()

if __name__ == "__main__":
    main()
