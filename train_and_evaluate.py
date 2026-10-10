"""
=============================================================================
BƯỚC 4, 5, 6: CHỌN MÔ HÌNH, HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING
=============================================================================
Môn học: Trí Tuệ Nhân Tạo (AI)
Đề tài: Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe y tế
Các mô hình:
  1. Logistic Regression (Hồi quy Logistic)
  2. Random Forest (Rừng ngẫu nhiên)
  3. Support Vector Machine - SVM (Máy vector hỗ trợ)
  4. Gradient Boosting Classifier (Tăng cường độ dốc)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from pipeline_data import load_data, clean_and_preprocess_data, split_and_scale_data

def train_and_tune_models(data_dict):
    """
    BƯỚC 4: CHỌN MÔ HÌNH & BƯỚC 5: HUẤN LUYỆN CÁC MÔ HÌNH
    - Huấn luyện trên tập Train (70% - 3,294 mẫu)
    - Kiểm tra trên tập Validation (15% - 706 mẫu)
    """
    print("\n" + "="*70)
    print("BƯỚC 4 & 5: KHỞI TẠO VÀ HUẤN LUYỆN CÁC MÔ HÌNH HỌC MÁY")
    print("="*70)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']

    X_train_scaled = data_dict['X_train_scaled']
    X_val_scaled = data_dict['X_val_scaled']

    # Danh sách các mô hình ứng viên
    # Lưu ý: Các mô hình tuyến tính (Logistic Regression, SVM) sử dụng dữ liệu đã chuẩn hóa (scaled)
    # Các mô hình dạng cây (Random Forest, Gradient Boosting) có thể dùng trực tiếp hoặc dữ liệu scaled
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                class_weight='balanced', 
                random_state=42
            ),
            'use_scaled': True,
            'desc': 'Mô hình tuyến tính cổ điển, có trọng số cân bằng lớp và khả năng giải thích tốt.'
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=150, 
                max_depth=12,
                min_samples_split=4,
                class_weight='balanced', 
                random_state=42
            ),
            'use_scaled': False,
            'desc': 'Học kết hợp (Bagging), chống quá khớp tốt, nắm bắt quan hệ phi tuyến mạnh mẽ.'
        },
        'SVM (RBF Kernel)': {
            'model': SVC(
                kernel='rbf', 
                C=2.0, 
                gamma='scale', 
                probability=True, 
                class_weight='balanced', 
                random_state=42
            ),
            'use_scaled': True,
            'desc': 'Máy vector hỗ trợ với nhân RBF phân tách phi tuyến trong không gian nhiều chiều.'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=120, 
                learning_rate=0.08, 
                max_depth=4, 
                random_state=42
            ),
            'use_scaled': False,
            'desc': 'Học tăng cường (Boosting), tối ưu hàm mất mát lặp lại từng cây quyết định.'
        }
    }

    trained_models = {}
    val_performances = {}

    print(f"{'Tên mô hình':<22} | {'Val Acc':<8} | {'Val Prec':<9} | {'Val Recall':<10} | {'Val F1':<8} | {'Val AUC':<8}")
    print("-" * 75)

    for name, config in models_config.items():
        model = config['model']
        use_scaled = config['use_scaled']

        # Chọn tập train phù hợp
        X_tr = X_train_scaled if use_scaled else X_train
        X_v = X_val_scaled if use_scaled else X_val

        # Huấn luyện mô hình
        model.fit(X_tr, y_train)
        trained_models[name] = {
            'model': model,
            'use_scaled': use_scaled,
            'desc': config['desc']
        }

        # Đánh giá sơ bộ trên tập Validation
        val_preds = model.predict(X_v)
        val_probs = model.predict_proba(X_v)[:, 1]

        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds)
        rec = recall_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)

        val_performances[name] = {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc
        }

        print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}     | {f1:.4f}   | {auc:.4f}")

    return trained_models, val_performances

def evaluate_models_on_test(trained_models, data_dict):
    """
    BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN TRÊN TẬP TEST ĐỘC LẬP (15% - 707 mẫu)
    - Đo lường: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - Lập ma trận nhầm lẫn (Confusion Matrix)
    - Vẽ và xuất 4 đồ thị trực quan hóa khoa học
    """
    print("\n" + "="*75)
    print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET - 707 MẪU)")
    print("="*75)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    X_test_scaled = data_dict['X_test_scaled']
    feature_cols = data_dict['feature_cols']

    results = []
    test_metrics = {}

    print(f"{'Mô hình':<22} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<9} | {'ROC-AUC':<8}")
    print("-" * 75)

    for name, item in trained_models.items():
        model = item['model']
        use_scaled = item['use_scaled']

        X_te = X_test_scaled if use_scaled else X_test

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, probs)

        test_metrics[name] = {
            'model': model,
            'use_scaled': use_scaled,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'ConfusionMatrix': cm,
            'FPR': fpr,
            'TPR': tpr,
            'Predictions': preds,
            'Probabilities': probs
        }

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

        print(f"{name:<22} | {acc:.4f}    | {prec:.4f}     | {rec:.4f}   | {f1:.4f}    | {auc:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("ket_qua_danh_gia_mo_hinh.csv", index=False, encoding='utf-8')
    print("\n[Xuất báo cáo] Đã lưu bảng kết quả chi tiết tại 'ket_qua_danh_gia_mo_hinh.csv'")

    # Xác định mô hình tốt nhất (dựa trên F1-Score và Recall)
    best_model_name = df_results.sort_values(by=['F1-Score', 'Recall', 'Accuracy'], ascending=False).iloc[0]['Model']
    print(f"\n=> MÔ HÌNH TỐI ƯU NHẤT ĐƯỢC CHỌN: [{best_model_name}]")
    print(f"   (Lý do: Đạt F1-score = {test_metrics[best_model_name]['F1-Score']:.4f} và Recall = {test_metrics[best_model_name]['Recall']:.4f}, kiểm soát rủi ro bỏ sót bệnh nhân tối ưu nhất)")

    # Lưu mô hình tốt nhất
    best_bundle = {
        'name': best_model_name,
        'model': test_metrics[best_model_name]['model'],
        'use_scaled': test_metrics[best_model_name]['use_scaled'],
        'feature_cols': feature_cols,
        'metrics': {k: v for k, v in test_metrics[best_model_name].items() if k not in ['model', 'ConfusionMatrix', 'FPR', 'TPR', 'Predictions', 'Probabilities']}
    }
    joblib.dump(best_bundle, "best_model.joblib")
    print(f"[Lưu mô hình] Đã lưu mô hình tốt nhất vào 'best_model.joblib'")

    # Lưu tất cả các mô hình
    os.makedirs("models", exist_ok=True)
    for name, item in trained_models.items():
        fname = f"models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(item, fname)
    print("[Lưu mô hình] Đã lưu đầy đủ 4 mô hình vào thư mục 'models/'")

    # -----------------------------------------------------------------------
    # VẼ VÀ XUẤT CÁC BIỂU ĐỒ ĐÁNH GIÁ (PLOTS)
    # -----------------------------------------------------------------------
    plot_evaluation_charts(df_results, test_metrics, best_model_name, feature_cols)

    return test_metrics, best_model_name

def plot_evaluation_charts(df_results, test_metrics, best_model_name, feature_cols):
    """Vẽ 4 biểu đồ trực quan hóa kết quả kiểm thử học máy"""
    os.makedirs("visualizations", exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Biểu đồ so sánh các chỉ số giữa các mô hình
    plt.figure(figsize=(12, 6))
    df_melt = df_results.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                              var_name='Metric', value_name='Score')
    ax = sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='tab10')
    plt.title('SO SÁNH CÁC CHỈ SỐ ĐÁNH GIÁ GIỮA CÁC MÔ HÌNH HỌC MÁY (TẬP TEST)', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.75, 1.02)
    plt.ylabel('Điểm số (Score)', fontsize=12)
    plt.xlabel('Chỉ số đánh giá', fontsize=12)
    plt.legend(title='Mô hình', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/01_so_sanh_hieu_nang_mo_hinh.png', dpi=300)
    plt.close()

    # 2. Đường cong ROC (Receiver Operating Characteristic)
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', label='Đường chuẩn ngẫu nhiên (AUC = 0.50)')
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    for idx, (name, metrics) in enumerate(test_metrics.items()):
        fpr = metrics['FPR']
        tpr = metrics['TPR']
        auc = metrics['ROC-AUC']
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.2,
                 label=f'{name} (AUC = {auc:.4f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate / Recall)', fontsize=12)
    plt.title('ĐƯỜNG CONG ROC (ROC CURVES) - ĐÁNH GIÁ NĂNG LỰC PHÂN TÁCH', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/02_duong_cong_roc.png', dpi=300)
    plt.close()

    # 3. Ma trận nhầm lẫn (Confusion Matrices) của cả 4 mô hình
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, (name, metrics) in enumerate(test_metrics.items()):
        cm = metrics['ConfusionMatrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Âm tính (0)', 'Dương tính (1)'],
                    yticklabels=['Âm tính (0)', 'Dương tính (1)'])
        axes[idx].set_title(f'{name}\n(Acc: {metrics["Accuracy"]:.3f} | F1: {metrics["F1-Score"]:.3f})', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Dự đoán của mô hình', fontsize=10)
        axes[idx].set_ylabel('Thực tế lâm sàng', fontsize=10)
    plt.suptitle('MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRÊN TẬP TEST ĐỘC LẬP', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('visualizations/03_ma_tran_nham_lan.png', dpi=300)
    plt.close()

    # 4. Độ quan trọng của đặc trưng (Feature Importance) từ Random Forest & Gradient Boosting
    rf_model = test_metrics['Random Forest']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('MỨC ĐỘ ĐÓNG GÓP CỦA CÁC CHỈ SỐ Y TẾ (FEATURE IMPORTANCE - RANDOM FOREST)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Tỷ trọng quan trọng tương đối', fontsize=11)
    plt.ylabel('Chỉ số sức khỏe / xét nghiệm', fontsize=11)
    for i, v in enumerate(sorted_importances):
        plt.text(v + 0.005, i, f'{v*100:.2f}%', va='center', fontsize=10, fontweight='bold')
    plt.xlim(0, max(sorted_importances) + 0.08)
    plt.tight_layout()
    plt.savefig('visualizations/04_do_quan_trong_dac_trung.png', dpi=300)
    plt.close()

    print("[Vẽ đồ thị] Đã xuất thành công 4 biểu đồ phân tích trực quan vào thư mục 'visualizations/'")

def run_full_training():
    """Hàm chạy toàn bộ quy trình tiền xử lý và huấn luyện"""
    df_raw = load_data()
    df_clean, imp, clips = clean_and_preprocess_data(df_raw)
    data_dict = split_and_scale_data(df_clean)

    trained_models, val_perf = train_and_tune_models(data_dict)
    test_metrics, best_name = evaluate_models_on_test(trained_models, data_dict)

    print("\n" + "="*70)
    print("HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ!")
    print(f"Mô hình được sử dụng trong Giao diện phần mềm: [{best_name}]")
    print("="*70)

if __name__ == "__main__":
    run_full_training()
