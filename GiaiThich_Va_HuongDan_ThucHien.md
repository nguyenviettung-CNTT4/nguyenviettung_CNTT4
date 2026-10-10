# BÁO CÁO GIẢI THÍCH VÀ HƯỚNG DẪN THỰC HIỆN BÀI KIỂM TRA
## MÔN HỌC: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE)
**Đề tài**: Xây dựng phần mềm AI (Machine Learning): Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe y tế  
**Tập dữ liệu**: `CSDL_TieuDuong.csv` (4,707 bản ghi bệnh nhân)  
**Tác giả**: [Họ và Tên Học Viên]  
**Lớp / Khóa**: K23.4  

---

## MỤC LỤC
1. [Bước 1: Xác Định Bài Toán](#bước-1-xác-định-bài-toán)
2. [Bước 2: Dữ Liệu Y Tế Thực Tế](#bước-2-dữ-liệu-y-tế-thực-tế)
3. [Bước 3: Tiền Xử Lý Dữ Liệu Toàn Diện](#bước-3-tiền-xử-lý-dữ-liệu-toàn-diện)
   - 3.1. Làm sạch dữ liệu (Thiếu, Đồng bộ, Lỗi logic, Ngoại lai)
   - 3.2. Biến đổi dữ liệu (Mã hóa và Chuẩn hóa)
   - 3.3. Phân chia tập dữ liệu (70% Train, 15% Val, 15% Test)
4. [Bước 4: Lựa Chọn Các Mô Hình Học Máy](#bước-4-lựa-chọn-các-mô-hình-học-máy)
5. [Bước 5: Huấn Luyện Và Tinh Chỉnh Mô Hình](#bước-5-huấn-luyện-và-tinh-chỉnh-mô-hình)
6. [Bước 6: Đánh Giá Mô Hình Toàn Diện](#bước-6-đánh-giá-mô-hình-toàn-diện)
7. [Bước 7: Xây Dựng Giao Diện, Sinh Cảnh Báo Và Giải Thích](#bước-7-xây-dựng-giao-diện-sinh-cảnh-báo-và-giải-thích)
8. [Hướng Dẫn Cài Đặt Và Khởi Chạy Phần Mềm](#hướng-dẫn-cài-đặt-và-khởi-chạy-phần-mềm)

---

## BƯỚC 1: XÁC ĐỊNH BÀI TOÁN

### 1.1. Bối cảnh thực tế và mục tiêu
Bệnh đái tháo đường (tiểu đường) là một trong những căn bệnh không lây nhiễm nguy hiểm hàng đầu trên thế giới và tại Việt Nam. Bệnh tiến triển âm thầm, nếu không được phát hiện sớm sẽ dẫn tới các biến chứng nghiêm trọng về tim mạch, suy thận, mù lòa, đột quỵ và hoại tử chi.

**Mục tiêu**: Xây dựng một hệ thống Trí tuệ Nhân tạo ứng dụng các thuật toán Học máy (Machine Learning) có khả năng tự động phân tích 13 chỉ số thông tin nhân khẩu học và xét nghiệm sinh hóa máu thông thường, từ đó đưa ra kết luận phân loại nguy cơ mắc bệnh tiểu đường, định lượng xác suất nguy cơ (%), tự động sinh cảnh báo lâm sàng và giải thích các nguyên nhân bệnh lý cho bác sĩ và người bệnh.

### 1.2. Phân loại bài toán trong Trí tuệ Nhân tạo
- **Dạng bài toán**: Học có giám sát (Supervised Learning) - Phân loại nhị phân (Binary Classification).
- **Không gian đầu vào (Features - $X \in \mathbb{R}^{13}$)**: Gồm 13 biến đặc trưng sinh học và xét nghiệm.
- **Không gian đầu ra (Target - $y \in \{0, 1\}$)**:
  - $y = 0$: Không có nguy cơ mắc bệnh tiểu đường (Âm tính / Bình thường).
  - $y = 1$: Có nguy cơ mắc bệnh tiểu đường (Dương tính).

### 1.3. Yêu cầu đặc thù trong bài toán Y tế
Trong bài toán chẩn đoán y khoa, việc dự đoán nhầm một người có bệnh thành không có bệnh (**Dương tính giả $\rightarrow$ False Negative**) là rủi ro nguy hiểm nhất vì làm mất cơ hội điều trị kịp thời của bệnh nhân. Do đó:
- Bên cạnh **Accuracy** (Độ chính xác tổng thể), các chỉ số **Recall (Độ nhạy)** và **F1-Score** đối với lớp nguy cơ ($y=1$) là tiêu chí ưu tiên cao nhất để lựa chọn mô hình.

---

## BƯỚC 2: DỮ LIỆU Y TẾ THỰC TẾ

Tập dữ liệu được cung cấp trong file `CSDL_TieuDuong.csv` gồm **4,707 dòng** và **14 cột**, bao gồm các chỉ số xét nghiệm lâm sàng tiêu chuẩn:

| STT | Tên Cột | Ý Nghĩa Y Khoa | Đơn Vị Đo | Khoảng Tham Chiếu Chuẩn |
|:---:|:---|:---|:---:|:---|
| 1 | `Tuoi` | Tuổi của người bệnh | Năm | 1 - 105 tuổi |
| 2 | `GioiTinh` | Giới tính sinh học | Phân loại | Nam / Nữ |
| 3 | `Glucose` | Đường huyết tương lúc đói / ngẫu nhiên | mmol/L | 3.9 - 5.6 mmol/L |
| 4 | `HbA1C` | Hemoglobin glycated (đường huyết 3 tháng) | % | < 5.7% (ĐTĐ khi ≥ 6.5%) |
| 5 | `Cholesterol` | Cholesterol toàn phần trong máu | mmol/L | 2.8 - 5.2 mmol/L |
| 6 | `HDL` | High-Density Lipoprotein (Cholesterol tốt) | mmol/L | > 1.0 (Nam), > 1.2 (Nữ) |
| 7 | `LDL` | Low-Density Lipoprotein (Cholesterol xấu) | mmol/L | < 3.4 mmol/L |
| 8 | `Triglycerid` | Chất béo trung tính trong máu | mmol/L | < 1.7 mmol/L |
| 9 | `Creatinine` | Chỉ số chức năng lọc cầu thận | µmol/L | Nam: 62-115, Nữ: 44-88 |
| 10 | `Ure` | Nồng độ Ure máu (đánh giá chuyển hóa/thận) | mmol/L | 2.5 - 7.5 mmol/L |
| 11 | `AcidUric` | Nồng độ Acid Uric máu (nguy cơ gút/chuyển hóa)| µmol/L | Nam: 150-420, Nữ: 150-360 |
| 12 | `AST` | Men gan AST / GOT (tổn thương tế bào gan) | U/L | < 40 U/L |
| 13 | `ALT` | Men gan ALT / GPT (tổn thương tế bào gan) | U/L | < 40 U/L |
| 14 | `NguyCoTieuDuong` | **Biến mục tiêu (Target)** | Nhãn | 0: Không / 1: Có nguy cơ |

---

## BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU TOÀN DIỆN

### 3.1. Làm sạch dữ liệu (Data Cleaning)

Qua quá trình phân tích khám phá dữ liệu (EDA), phát hiện dữ liệu thực tế có nhiều lỗi nhập liệu điển hình cần xử lý:

#### A. Đồng bộ dữ liệu (Data Synchronization):
1. **Cột mục tiêu `NguyCoTieuDuong`**:
   - Dữ liệu gốc tồn tại ở nhiều dạng biểu diễn hỗn hợp: `'FALSE'` (3,607 dòng), `'TRUE'` (1,094 dòng), `'1'` (2 dòng), `'0'` (2 dòng), `'Có nguy cơ'` (1 dòng), `'Không có nguy cơ'` (1 dòng).
   - **Xử lý**: Đồng bộ toàn bộ về chuẩn nhị phân $\{0, 1\}$:
     $$\{ \text{'TRUE'}, \text{'1'}, \text{'Có nguy cơ'}, 1 \} \rightarrow 1$$
     $$\{ \text{'FALSE'}, \text{'0'}, \text{'Không có nguy cơ'}, 0 \} \rightarrow 0$$
   - Kết quả: **3,610 mẫu nhãn 0** (76.7%) và **1,097 mẫu nhãn 1** (23.3%). Tỷ lệ mất cân bằng lớp xấp xỉ 3.3 : 1.
2. **Cột phân loại `GioiTinh`**:
   - Dữ liệu gốc gồm: `'Nam'` (2,854), `'Nữ'` (1,838), `'Male'` (6), `'Female'` (5), và 4 giá trị rỗng `NaN`.
   - **Xử lý**: Đồng bộ hóa:
     $$\{ \text{'Nam'}, \text{'Male'} \} \rightarrow 1$$
     $$\{ \text{'Nữ'}, \text{'Female'} \} \rightarrow 0$$

#### B. Kiểm tra và sửa lỗi logic y khoa (Medical Logic Error Correction):
1. **Lỗi chỉ số `HbA1C` = 8025.0**:
   - Phân tích: Ngưỡng sinh lý HbA1c chỉ từ 4.0% đến tối đa ~18.0%. Giá trị 8025.0 là lỗi người nhập liệu gõ thiếu dấu chấm thập phân của `8.025%`.
   - **Xử lý**: Lọc các giá trị $HbA1c > 20.0\%$ và chia cho $1000$ ($8025.0 / 1000 = 8.025\%$).
2. **Lỗi chỉ số `Cholesterol` = 55.0**:
   - Phân tích: Ngưỡng sinh lý Cholesterol toàn phần là 2.8 - 15.0 mmol/L. Giá trị 55.0 là do thiếu dấu phẩy thập phân của `5.5 mmol/L`.
   - **Xử lý**: Lọc giá trị $Cholesterol > 25.0$ và chia cho $10$ ($55.0 / 10 = 5.5$).
3. **Lỗi sai lệch đơn vị đo `Glucose` > 50 (59.0, 61.0, 92.0, 750.0)**:
   - Phân tích y khoa: Tại các bệnh viện, một số máy xét nghiệm xuất đơn vị `mg/dL` thay vì `mmol/L`.
     - Với bệnh nhân có Glucose = 59.0 mg/dL: Quy đổi $59 / 18.0182 = 3.27$ mmol/L (hạ đường huyết, nhãn gốc là FALSE).
     - Với Glucose = 61.0 mg/dL: Quy đổi $61 / 18.0182 = 3.38$ mmol/L (nhãn gốc là FALSE).
     - Với Glucose = 750.0 mg/dL: Quy đổi $750 / 18.0182 = 41.62$ mmol/L (hôn mê tăng đường huyết nặng, nhãn gốc là TRUE).
   - **Xử lý**: Tự động nhận diện giá trị $Glucose > 50$ và quy đổi: $Glucose_{mmol/L} = Glucose_{mg/dL} / 18.0182$.

#### C. Kiểm tra và xử lý dữ liệu thiếu (Missing Value Imputation):
- Trong toàn bộ 4,707 dòng, chỉ có **21 dòng** chứa ít nhất một giá trị khuyết thiếu (tỷ lệ rất nhỏ < 0.5%):
  - `Tuoi`: 1 khuyết; `GioiTinh`: 4 khuyết; `Glucose`: 1 khuyết; `HbA1C`: 2 khuyết; `Cholesterol`: 2 khuyết; `LDL`: 2 khuyết; `Triglycerid`: 3 khuyết; `Creatinine`: 1 khuyết; `AcidUric`: 5 khuyết; `AST`: 2 khuyết; `ALT`: 2 khuyết.
- **Phương pháp xử lý**:
  - Với biến định tính `GioiTinh`: Điền bằng giá trị xuất hiện nhiều nhất (**Mode** = `Nam` / 1).
  - Với các biến định lượng sinh hóa: Điền bằng trung vị (**Median**) của từng thuộc tính trên tập dữ liệu để tránh bị sai lệch bởi các giá trị cực trị.

#### D. Kiểm tra và xử lý dữ liệu ngoại lai (Outliers):
- Trong dữ liệu xét nghiệm lâm sàng, các giá trị cao bất thường (ví dụ: Creatinine > 1000 µmol/L ở bệnh nhân suy thận giai đoạn cuối, AST/ALT > 500 U/L ở bệnh nhân viêm gan cấp) là hiện tượng sinh lý bệnh thực tế, không được xóa vì sẽ làm mất đặc trưng của người bệnh nặng.
- **Phương pháp xử lý**: Áp dụng kỹ thuật **Winsorization (Clipping)** tại ngưỡng bách phân vị $0.1\%$ (cận dưới) và $99.9\%$ (cận trên). Điều này vừa giữ nguyên cấu trúc phân bố lâm sàng, vừa bảo vệ các mô hình tuyến tính (như Logistic Regression, SVM) khỏi bị phân tán gradient bởi các điểm ngoại lai cực đoan.

---

### 3.2. Biến đổi dữ liệu (Data Transformation)
1. **Mã hóa (Encoding)**:
   - Thuộc tính `GioiTinh` được mã hóa nhị phân: `Nam` = 1, `Nữ` = 0.
2. **Chuẩn hóa (Feature Scaling)**:
   - Các chỉ số xét nghiệm có đơn vị và thang đo rất khác nhau (Tuổi: 1-100; Glucose: 3-30; Creatinine: 50-1500; Acid Uric: 150-1000).
   - Sử dụng **StandardScaler** (Z-score normalization) để đưa dữ liệu về phân phối có kỳ vọng $\mu = 0$ và độ lệch chuẩn $\sigma = 1$:
     $$z = \frac{x - \mu}{\sigma}$$
   - **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage)**: Scaler được `fit` duy nhất trên tập Huấn luyện (Train), sau đó dùng để `transform` cho tập Validation và tập Test.

---

### 3.3. Phân chia tập dữ liệu (Data Splitting)
Theo yêu cầu đề bài: **Tỷ lệ 70% Train, 30% Test (trong đó gồm 15% Validation và 15% Test)**:
- **Tập Train (70%)**: **3,294 mẫu** (768 dương tính, tỷ lệ 23.3%).
- **Tập Validation (15%)**: **706 mẫu** (164 dương tính, tỷ lệ 23.2%).
- **Tập Test (15%)**: **707 mẫu** (165 dương tính, tỷ lệ 23.3%).

Áp dụng kỹ thuật **Stratified Sampling (Phân tầng)** theo nhãn mục tiêu `NguyCoTieuDuong` để đảm bảo tỷ lệ mẫu dương tính ở cả 3 tập đều đồng nhất chính xác 23.3%.

---

## BƯỚC 4: LỰA CHỌN CÁC MÔ HÌNH HỌC MÁY

Để có cơ sở so sánh khách quan và tìm ra giải pháp tối ưu nhất cho bài toán y tế, chúng tôi lựa chọn 4 mô hình đại diện cho 4 trường phái thuật toán khác nhau:

1. **Logistic Regression (Hồi quy Logistic)**:
   - *Nguyên lý*: Mô hình tuyến tính cổ điển, tính toán xác suất thông qua hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
   - *Ưu điểm*: Tốc độ huấn luyện cực nhanh, khả năng diễn giải cao thông qua trọng số trọng số $\beta_i$.
   - *Cấu hình*: Regularization L2, `class_weight='balanced'` để bù đắp mất cân bằng lớp.

2. **Support Vector Machine - SVM (Máy Vector Hỗ Trợ)**:
   - *Nguyên lý*: Tìm siêu phẳng tối ưu phân tách 2 lớp với lề cực đại (Maximum Margin). Sử dụng nhân phi tuyến **RBF (Radial Basis Function)**:
     $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
   - *Ưu điểm*: Hoạt động hiệu quả trong không gian nhiều chiều, chống quá khớp tốt nhờ cơ chế regularization $C$.
   - *Cấu hình*: `kernel='rbf'`, `probability=True`, `class_weight='balanced'`.

3. **Random Forest Classifier (Rừng Ngẫu Nhiên)**:
   - *Nguyên lý*: Thuật toán học kết hợp theo phương pháp **Bagging (Bootstrap Aggregating)**. Xây dựng một tập hợp nhiều Cây quyết định (Decision Trees) độc lập và lấy biểu quyết đa số.
   - *Ưu điểm*: Bắt trọn các tương tác phi tuyến phức tạp giữa các chỉ số xét nghiệm, kháng ngoại lai tốt, cung cấp trực quan độ quan trọng của đặc trưng (**Feature Importance**).
   - *Cấu hình*: `n_estimators=150`, `max_depth=12`, `min_samples_split=4`, `class_weight='balanced'`.

4. **Gradient Boosting Classifier (Tăng Cường Độ Dốc)**:
   - *Nguyên lý*: Thuật toán học kết hợp theo phương pháp **Boosting**. Xây dựng tuần tự các cây quyết định mới để sửa chữa phần dư (residuals) và tối ưu hàm mất mát Log-loss của các cây trước đó.
   - *Ưu điểm*: Khả năng khái quát hóa cực cao, thường đạt độ chính xác hàng đầu trên các bảng dữ liệu dạng bảng (Tabular Data).
   - *Cấu hình*: `n_estimators=120`, `learning_rate=0.08`, `max_depth=4`.

---

## BƯỚC 5: HUẤN LUYỆN VÀ TINH CHỈNH MÔ HÌNH

- Các mô hình được huấn luyện trên **Tập Train (3,294 mẫu)**.
- Đánh giá sơ bộ và kiểm chứng siêu tham số trên **Tập Validation (706 mẫu)** để tránh hiện tượng Overfitting.

### Kết quả trên tập Validation (706 mẫu):
| STT | Mô Hình | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Gradient Boosting** | **0.9972** | **1.0000** | **0.9878** | **0.9939** | **0.9939** |
| 2 | **Random Forest** | 0.9929 | 1.0000 | 0.9695 | 0.9845 | 1.0000 |
| 3 | **Logistic Regression** | 0.9660 | 0.8977 | 0.9634 | 0.9294 | 0.9801 |
| 4 | **SVM (RBF Kernel)** | 0.9561 | 0.8757 | 0.9451 | 0.9091 | 0.9885 |

---

## BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN (TRÊN TẬP TEST ĐỘC LẬP)

Tập Kiểm thử (Test Set) gồm **707 mẫu** hoàn toàn độc lập (chưa từng xuất hiện trong quá trình huấn luyện hay tinh chỉnh siêu tham số).

### 6.1. Bảng số liệu đánh giá đối sánh các mô hình trên tập Test:
| STT | Tên Mô Hình | Accuracy | Precision | Recall (Độ nhạy) | F1-Score | ROC-AUC |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | 🏆 **Gradient Boosting** | **99.86%** (0.9986) | **100.00%** (1.0000) | **99.39%** (0.9939) | **99.70%** (0.9970) | **0.9970** |
| 2 | 🥈 **Random Forest** | **99.58%** (0.9958) | **100.00%** (1.0000) | **98.18%** (0.9818) | **99.08%** (0.9908) | **0.9959** |
| 3 | 🥉 **Logistic Regression**| **97.45%** (0.9745) | **92.00%** (0.9200) | **97.58%** (0.9758) | **94.71%** (0.9471) | **0.9863** |
| 4 | 🏅 **SVM (RBF Kernel)** | **95.90%** (0.9590) | **87.36%** (0.8736) | **96.36%** (0.9636) | **91.64%** (0.9164) | **0.9885** |

### 6.2. Phân tích kết quả chuyên sâu
1. **Mô hình xuất sắc nhất**: **Gradient Boosting Classifier** đạt điểm số vượt trội trên mọi khía cạnh:
   - **Độ chính xác (Accuracy)**: 99.86% (chỉ sai 1 mẫu trên tổng số 707 bệnh nhân kiểm thử).
   - **Độ nhạy (Recall)**: 99.39% (phát hiện chính xác 164/165 người mắc bệnh, chỉ bỏ sót duy nhất 1 ca).
   - **Độ chính xác chẩn đoán (Precision)**: 100.00% (không có bất kỳ ca báo động giả nào).
   - **Điểm F1-Score**: 99.70% và diện tích dưới đường cong ROC (**AUC**): 0.9970 khẳng định khả năng phân tách ranh giới bệnh lý hoàn hảo.
2. **Mô hình Random Forest**: Đạt hiệu năng rất cao sát nút (F1: 99.08%, Recall: 98.18%), đồng thời cung cấp khả năng phân tích tầm quan trọng của các biến xét nghiệm.
3. **Mô hình Logistic Regression & SVM**: Mặc dù là các mô hình đơn giản hơn, nhưng nhờ quá trình làm sạch dữ liệu và chuẩn hóa kỹ lưỡng ở Bước 3, cả hai mô hình vẫn đạt Recall > 96% và AUC > 0.98.

### 6.3. Phân tích độ quan trọng của đặc trưng (Feature Importance)
Từ mô hình Random Forest, tỷ trọng đóng góp của các yếu tố nguy cơ dẫn đến bệnh tiểu đường được xếp hạng như sau:
1. **Glucose máu**: Đóng góp lớn nhất (**> 65%**), là chỉ số quyết định trực tiếp.
2. **Tuổi (Tuoi)**: Đóng góp **~10%**, phản ánh sự suy giảm tiết insulin theo thời gian.
3. **Triglycerid & Cholesterol**: Đóng góp **~8%**, phản ánh hội chứng rối loạn chuyển hóa và kháng insulin.
4. **Men gan (AST, ALT)**: Đóng góp **~6%**, liên quan đến tình trạng gan nhiễm mỡ chuyển hóa (MASLD).
5. **HbA1c & Chức năng thận (Creatinine, Ure, Acid Uric)**: Đóng góp phần còn lại.

*(Tất cả 4 biểu đồ trực quan hóa khoa học được lưu tại thư mục `visualizations/`: so sánh chỉ số, đường cong ROC, ma trận nhầm lẫn và độ quan trọng đặc trưng).*

---

## BƯỚC 7: XÂY DỰNG GIAO DIỆN, SINH CẢNH BÁO VÀ GIẢI THÍCH

Để đáp ứng trọn vẹn yêu cầu Bước 7 của đề bài, chúng tôi xây dựng **hai phiên bản giao diện người dùng tương tác**:
1. **Giao diện Desktop GUI (`app_gui.py`)**: Sử dụng thư viện `ttkbootstrap` / `tkinter` hiện đại.
2. **Giao diện Web Ứng dụng (`app_web.py`)**: Sử dụng `Flask` với giao diện HTML5/CSS3 tương tác thời gian thực.

### 7.1. Các chức năng chính của giao diện:
1. **Nhập thông tin người bệnh**: Form nhập liệu 13 chỉ số sức khỏe, hỗ trợ chuyển đổi đơn vị sinh hóa linh hoạt (mmol/L và mg/dL).
2. **Nạp dữ liệu ca bệnh mẫu thử nghiệm nhanh**:
   - *Ca 1: Nguy cơ cao* (Bệnh nhân 62 tuổi, Glucose 11.9, HbA1c 8.0%, rối loạn mỡ máu).
   - *Ca 2: Người khỏe mạnh* (Thanh niên 28 tuổi, Glucose 4.8, HbA1c 5.1%, các chỉ số đều trong chuẩn).
   - *Ca 3: Tiền đái tháo đường* (Bệnh nhân 52 tuổi, Glucose 6.2, HbA1c 6.0%).
3. **Dự đoán thời gian thực**:
   - Hiển thị nhãn kết luận rõ ràng: **CÓ NGUY CƠ TIỂU ĐƯỜNG** hoặc **NGUY CƠ THẤP - AN TOÀN**.
   - Thước đo định lượng phần trăm nguy cơ (Risk Probability %) kèm màu sắc cảnh báo trực quan.
4. **Hệ thống sinh cảnh báo lâm sàng tự động (Clinical Warning System)**:
   - Tự động đối chiếu từng chỉ số với các tiêu chuẩn của Hiệp hội Đái tháo đường Hoa Kỳ (ADA) và Bộ Y tế:
     - Cảnh báo đường huyết tăng cao ($Glucose \ge 7.0$ mmol/L) hoặc hạ đường huyết nguy hiểm ($< 3.9$ mmol/L).
     - Cảnh báo chỉ số $HbA1c \ge 6.5\%$ phản ánh kiểm soát đường huyết 3 tháng kém.
     - Cảnh báo rối loạn chuyển hóa Lipid (Triglycerid tăng, LDL tăng, HDL giảm).
     - Cảnh báo suy giảm chức năng lọc cầu thận ($Creatinine$ vượt chuẩn nam/nữ).
     - Cảnh báo tổn thương men gan ($AST, ALT > 40$ U/L) gợi ý gan nhiễm mỡ.
5. **Cơ chế giải thích kết quả AI (Explainable AI - XAI)**:
   - Liệt kê chi tiết các nguyên nhân bệnh lý trực tiếp khiến mô hình đưa ra kết luận (ví dụ: do Glucose tăng vọt kết hợp Triglycerid cao và độ tuổi > 50).
   - Cung cấp các khuyến nghị y tế hành động cụ thể: Chế độ dinh dưỡng giảm đường nhanh, tập thể dục 150 phút/tuần, xét nghiệm khẳng định OGTT và khám chuyên khoa Nội tiết.
6. **Xuất phiếu kết quả khám bệnh**: Cho phép bác sĩ/người dùng xuất phiếu kết quả ra file `.txt` để lưu trữ hồ sơ bệnh án.

---

## HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY PHẦN MỀM

### 1. Yêu cầu môi trường
- Hệ điều hành: Windows 10 / 11 (hoặc macOS / Linux).
- Python 3.9+ (Đã kiểm thử hoạt động hoàn hảo trên Python 3.10.11).
- Các thư viện cần thiết:
  ```bash
  py -m pip install pandas numpy scikit-learn matplotlib seaborn joblib ttkbootstrap flask
  ```

### 2. Cấu trúc thư mục mã nguồn
```
HoTenHV_KiemTra/
│
├── CSDL_TieuDuong.csv              # File dữ liệu y tế gốc đề bài cung cấp
├── CSDL_TieuDuong_Cleaned.csv      # File dữ liệu sau khi làm sạch hoàn chỉnh
│
├── pipeline_data.py                # Mã nguồn Bước 1, 2, 3: Tiền xử lý, làm sạch, phân chia dữ liệu
├── train_and_evaluate.py           # Mã nguồn Bước 4, 5, 6: Huấn luyện, tối ưu và đánh giá 4 mô hình
├── predict_engine.py               # Bộ máy dự đoán, sinh cảnh báo y tế & giải thích kết quả
├── app_gui.py                      # Bước 7: Ứng dụng Desktop GUI (Tkinter/ttkbootstrap)
├── app_web.py                      # Bước 7: Ứng dụng Web (Flask backend)
├── main.py                         # File thực thi toàn diện một lệnh duy nhất
│
├── best_model.joblib               # Mô hình tối ưu nhất được đóng gói (Gradient Boosting)
├── preprocessing_meta.joblib       # Bộ chuẩn hóa Scaler và thông số tiền xử lý
├── ket_qua_danh_gia_mo_hinh.csv   # Bảng tổng hợp các chỉ số đánh giá của 4 mô hình
│
├── models/                         # Thư mục lưu 4 mô hình học máy đã huấn luyện
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   ├── SVM_RBF_Kernel.joblib
│   └── Gradient_Boosting.joblib
│
├── visualizations/                 # Thư mục chứa 4 biểu đồ đánh giá khoa học
│   ├── 01_so_sanh_hieu_nang_mo_hinh.png
│   ├── 02_duong_cong_roc.png
│   ├── 03_ma_tran_nham_lan.png
│   └── 04_do_quan_trong_dac_trung.png
│
├── templates/                      # Giao diện web HTML5/CSS3
│   └── index.html
│
└── GiaiThich_Va_HuongDan_ThucHien.md # Báo cáo chi tiết và hướng dẫn sử dụng (File này)
```

### 3. Các bước thực hiện

#### Cách 1: Chạy toàn bộ pipeline huấn luyện và đánh giá
Mở cửa sổ dòng lệnh (Terminal / PowerShell) tại thư mục bài làm, gõ lệnh:
```bash
py main.py
```
*(Hệ thống sẽ tự động thực hiện từ Bước 1 đến Bước 6, xuất kết quả đánh giá ra màn hình console, tạo bảng CSV, lưu các mô hình và sinh 4 biểu đồ hình ảnh).*

#### Cách 2: Khởi chạy Giao diện Desktop (Bước 7)
Chạy lệnh:
```bash
py app_gui.py
```
*(Cửa sổ ứng dụng Desktop sẽ hiển thị. Người dùng có thể nhấn thử các nút ca bệnh mẫu hoặc nhập thông số và nhấn "DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG" để xem kết quả, cảnh báo và giải thích).*

#### Cách 3: Khởi chạy Giao diện Web (Bước 7)
Chạy lệnh:
```bash
py app_web.py
```
Mở trình duyệt Web (Chrome, Edge, Firefox) và truy cập vào địa chỉ:
```
http://localhost:5000
```

---
**KẾT LUẬN**: Toàn bộ yêu cầu của đề bài kiểm tra từ Bước 1 đến Bước 7 đã được thực hiện đầy đủ, chuẩn xác về mặt khoa học dữ liệu và logic y học, kèm theo mã nguồn Python trong sáng, module hóa chuyên nghiệp và tài liệu hướng dẫn cụ thể.
