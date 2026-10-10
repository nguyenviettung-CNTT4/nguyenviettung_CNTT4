"""
=============================================================================
BƯỚC 7 (GIAO DIỆN WEB APP): ỨNG DỤNG WEB DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Chạy ứng dụng: py app_web.py
Truy cập: http://localhost:5000
=============================================================================
"""

import sys
import os
from flask import Flask, render_template, request, jsonify

# Đảm bảo in tiếng Việt
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from predict_engine import DiabetesRiskPredictor

app = Flask(__name__)
predictor = DiabetesRiskPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Dữ liệu trống'}), 400

        # Chuyển đổi kiểu dữ liệu
        clean_inputs = {}
        for k, v in data.items():
            if k == 'GioiTinh':
                clean_inputs[k] = v
            else:
                clean_inputs[k] = float(v)

        result = predictor.predict_patient(clean_inputs)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG AI DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG (WEB APP)")
    print("🌐 Truy cập tại: http://localhost:5000")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
