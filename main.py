"""
=============================================================================
CHƯƠNG TRÌNH CHÍNH (MAIN PIPELINE)
BÀI KIỂM TRA MÔN: TRÍ TUỆ NHÂN TẠO
Đề tài: Xây dựng phần mềm AI dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe
=============================================================================
Cách sử dụng:
  1. Chạy toàn bộ quy trình AI (Từ Bước 1 đến Bước 6):
     py main.py

  2. Khởi chạy Giao diện Desktop (Bước 7):
     py app_gui.py

  3. Khởi chạy Giao diện Web (Bước 7):
     py app_web.py
=============================================================================
"""

import sys
import os

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pipeline_data import run_preprocessing_pipeline
from train_and_evaluate import run_full_training

def main():
    print("=" * 80)
    print("      ĐỀ KIỂM TRA MÔN HỌC: TRÍ TUỆ NHÂN TẠO")
    print("      DỰ ÁN: XÂY DỰNG HỆ THỐNG AI DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG")
    print("=" * 80)

    print("\n>>> BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH HỌC MÁY (END-TO-END PIPELINE)...\n")

    # Thực thi huấn luyện và đánh giá
    run_full_training()

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH XUẤT SẮC TOÀN BỘ CÁC BƯỚC TỪ 1 ĐẾN 6!")
    print("=" * 80)
    print("Các sản phẩm đã được tự động sinh ra và lưu trữ:")
    print("  1. Dữ liệu làm sạch   : CSDL_TieuDuong_Cleaned.csv")
    print("  2. Metadata tiền xử lý: preprocessing_meta.joblib")
    print("  3. Mô hình tốt nhất   : best_model.joblib")
    print("  4. Bảng kết quả test  : ket_qua_danh_gia_mo_hinh.csv")
    print("  5. Biểu đồ đánh giá   : Thư mục visualizations/ (4 biểu đồ khoa học)")
    print("  6. Thư mục mô hình    : Thư mục models/ (4 mô hình đã huấn luyện)")
    print("\n>>> HƯỚNG DẪN KHỞI CHẠY GIAO DIỆN (BƯỚC 7):")
    print("  • Cách 1 (Desktop GUI): Chạy lệnh: py app_gui.py")
    print("  • Cách 2 (Giao diện Web): Chạy lệnh: py app_web.py (Mở http://localhost:5000)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
