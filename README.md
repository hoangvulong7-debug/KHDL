# Dự đoán xu hướng Bitcoin trong 4 giờ tới

## 1. Giới thiệu dự án

Đây là dự án môn Khoa học dữ liệu, tập trung vào bài toán dự đoán xu hướng giá Bitcoin trong 4 giờ tiếp theo dựa trên dữ liệu nến 15 phút của cặp BTCUSDT.

Bài toán được xây dựng dưới dạng phân loại nhị phân:

- `0`: GIẢM
- `1`: TĂNG

Thay vì dự đoán chính xác giá Bitcoin trong tương lai, dự án dự đoán xu hướng tăng hoặc giảm sau 4 giờ.

---

## 2. Pipeline tổng thể

Dự án hiện được tách thành 2 file chính:

```text
features_full.py  → lấy dữ liệu + tạo feature + lưu MySQL/CSV
train_model.py    → đọc feature từ MySQL + tạo target + train model
```

Pipeline đầy đủ:

```text
Binance API
→ lấy dữ liệu nến BTCUSDT 15 phút
→ tạo feature kỹ thuật
→ lưu dữ liệu feature vào MySQL
→ đọc feature từ MySQL
→ tạo nhãn tăng/giảm sau 4 giờ
→ train nhiều mô hình
→ đánh giá bằng TimeSeriesSplit
→ chọn model tốt nhất
→ lưu model và kết quả
```

---

## 3. Cấu trúc thư mục

```text
KHDL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── features_full.py
│   └── train_model.py
│
├── outputs/
│   ├── timeseries_cv_detail.csv
│   ├── timeseries_cv_summary.csv
│   └── latest_prediction.csv
│
├── models/
│   ├── best_btc_4h_model.pkl
│   ├── feature_cols.pkl
│   └── model_metadata.pkl
│
└── docs/
    └── report_notes.md
```

---

## 4. File `features_full.py`

File `features_full.py` chịu trách nhiệm lấy dữ liệu và tạo feature.

Các công việc chính:

- Lấy dữ liệu BTCUSDT khung 15 phút từ Binance API.
- Lấy dữ liệu 2 năm gần nhất để đủ dữ liệu tính các chỉ báo như MA99, RSI, Bollinger Band, volatility.
- Tạo các feature kỹ thuật.
- Giữ lại 1 năm dữ liệu gần nhất.
- Xóa các dòng bị thiếu giá trị.
- Xuất dữ liệu feature ra CSV.
- Lưu dữ liệu feature vào MySQL.

Thông tin cấu hình chính:

```python
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
FETCH_DAYS = 730
FINAL_DAYS = 365
OUTPUT_FILE = "BTCUSDT_XU_HUONG_1NAM_EXACT_FEATURES.csv"
```

Trong đó:

| Tham số | Ý nghĩa |
|---|---|
| `SYMBOL` | Cặp giao dịch cần lấy dữ liệu |
| `INTERVAL` | Khung thời gian của nến |
| `FETCH_DAYS` | Số ngày dữ liệu được lấy từ Binance |
| `FINAL_DAYS` | Số ngày dữ liệu cuối cùng được giữ lại |
| `OUTPUT_FILE` | File CSV feature đầu ra |

---

## 5. Lưu dữ liệu vào MySQL

Dữ liệu feature sau khi tạo sẽ được lưu vào MySQL.

Cấu hình MySQL mặc định:

```python
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DATABASE = "crypto_db"
MYSQL_TABLE = "btc_features"
```

Bảng MySQL được dùng:

```text
crypto_db.btc_features
```

Bảng này lưu:

- Thông tin cặp giao dịch.
- Khung thời gian.
- Timestamp.
- Dữ liệu OHLCV.
- Các feature kỹ thuật đã được tạo.

Cột `ts` được dùng làm khóa chống trùng để khi chạy lại chương trình, dữ liệu cũ sẽ được cập nhật thay vì insert trùng.

---

## 6. Danh sách feature

Các feature được chia thành nhiều nhóm.

### 6.1. Dữ liệu gốc

| Feature | Ý nghĩa |
|---|---|
| `iso` | Thời gian của nến |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất |
| `low` | Giá thấp nhất |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch |
| `trades` | Số lượng giao dịch |

---

### 6.2. Feature từng nến 15 phút

| Feature | Ý nghĩa |
|---|---|
| `candle_return` | Lợi suất của từng nến |
| `candle_range_pct` | Biên độ high-low của nến |
| `candle_body_pct` | Độ lớn thân nến |
| `up_candle` | Nến tăng hay giảm |
| `close_return_1` | Tỷ lệ thay đổi giá đóng cửa so với nến trước |

---

### 6.3. Feature rolling window 16 nến

Dự án sử dụng 16 nến 15 phút gần nhất, tương đương 4 giờ dữ liệu quá khứ.

| Feature | Ý nghĩa |
|---|---|
| `window_open` | Giá mở cửa của nến đầu tiên trong window |
| `window_high` | Giá cao nhất trong 16 nến |
| `window_low` | Giá thấp nhất trong 16 nến |
| `window_close` | Giá đóng cửa hiện tại |
| `window_return` | Lợi suất từ đầu đến cuối window |
| `window_range` | Biên độ dao động trong window |
| `window_body_size` | Độ lớn thay đổi giá từ đầu đến cuối window |

---

### 6.4. Feature volume

| Feature | Ý nghĩa |
|---|---|
| `window_volume_sum` | Tổng volume trong 16 nến |
| `window_volume_mean` | Volume trung bình trong 16 nến |
| `window_volume_std` | Độ lệch chuẩn volume |
| `volume_change` | Mức thay đổi volume so với nến trước |
| `volume_ratio_16` | Volume hiện tại so với volume trung bình 16 nến |

---

### 6.5. Feature thống kê return

| Feature | Ý nghĩa |
|---|---|
| `mean_15m_return` | Return trung bình trong 16 nến |
| `std_15m_return` | Độ lệch chuẩn return |
| `max_15m_return` | Return lớn nhất |
| `min_15m_return` | Return nhỏ nhất |
| `up_candle_ratio` | Tỷ lệ nến tăng trong 16 nến |
| `close_position` | Vị trí giá đóng cửa trong vùng high-low |
| `volatility_16` | Độ biến động 16 nến |
| `volatility_32` | Độ biến động 32 nến |

---

### 6.6. Chỉ báo kỹ thuật

| Feature | Ý nghĩa |
|---|---|
| `MA7` | Trung bình động 7 nến |
| `MA25` | Trung bình động 25 nến |
| `MA99` | Trung bình động 99 nến |
| `EMA12` | Trung bình động hàm mũ 12 nến |
| `EMA26` | Trung bình động hàm mũ 26 nến |
| `MACD` | Chỉ báo xu hướng |
| `MACD_signal` | Đường tín hiệu MACD |
| `RSI` | Chỉ báo sức mạnh tương đối |
| `BB_upper` | Dải trên Bollinger Band |
| `BB_lower` | Dải dưới Bollinger Band |
| `BB_width` | Độ rộng Bollinger Band |

---

### 6.7. Khoảng cách giá so với đường trung bình

| Feature | Ý nghĩa |
|---|---|
| `distance_ma7` | Khoảng cách giá hiện tại so với MA7 |
| `distance_ma25` | Khoảng cách giá hiện tại so với MA25 |
| `distance_ma99` | Khoảng cách giá hiện tại so với MA99 |

---

## 7. File `train_model.py`

File `train_model.py` không còn đọc dữ liệu trực tiếp từ CSV nữa.

Thay vào đó, file này đọc dữ liệu feature từ MySQL:

```text
crypto_db.btc_features
```

Luồng xử lý của `train_model.py`:

```text
Kết nối MySQL
→ đọc feature từ bảng btc_features
→ sắp xếp theo thời gian
→ tạo target 4 giờ sau
→ lọc nhiễu bằng threshold
→ chuẩn bị X, y
→ train và đánh giá model
→ lưu kết quả
→ dự đoán 4 giờ tới
```

---

## 8. Tạo target dự đoán 4 giờ sau

Các tham số chính:

```python
WINDOW_SIZE = 16
HORIZON = 16
INTERVAL_MINUTES = 15
THRESHOLD = 0.0015
```

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `WINDOW_SIZE = 16` | Dùng 16 nến gần nhất làm dữ liệu quá khứ |
| `HORIZON = 16` | Dự đoán 16 nến tiếp theo |
| `INTERVAL_MINUTES = 15` | Mỗi nến dài 15 phút |
| `THRESHOLD = 0.0015` | Ngưỡng lọc nhiễu 0.15% |

Vì:

```text
16 × 15 phút = 240 phút = 4 giờ
```

nên mô hình dự đoán xu hướng sau 4 giờ.

Công thức tạo biến động tương lai:

```python
future_return_4h = future_close_4h / close - 1
```

Cách gán nhãn:

```text
future_return_4h > +0.15%  →  TĂNG = 1
future_return_4h < -0.15%  →  GIẢM = 0
nằm giữa ±0.15%            →  bỏ khỏi tập train
```

Mục đích của threshold là loại bỏ các biến động quá nhỏ, vì các biến động nhỏ thường bị nhiễu và khó dự đoán.

---

## 9. Các mô hình sử dụng

Dự án so sánh 4 mô hình:

1. Random Walk baseline
2. Logistic Regression
3. Random Forest
4. XGBoost

---

## 10. Random Walk baseline

Random Walk là mô hình baseline đơn giản, không phải mô hình học máy.

Quy tắc:

```text
Nếu window_return > 0  → dự đoán TĂNG
Nếu window_return <= 0 → dự đoán GIẢM
```

Random Walk được dùng để kiểm tra xem các mô hình học máy có tốt hơn một quy tắc đơn giản hay không.

---

## 11. Logistic Regression

Logistic Regression là mô hình tuyến tính cho bài toán phân loại nhị phân.

Trong dự án, mô hình được đặt trong pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])
```

Ý nghĩa:

| Thành phần | Ý nghĩa |
|---|---|
| `StandardScaler()` | Chuẩn hóa feature |
| `max_iter=1000` | Tăng số vòng lặp tối đa |
| `class_weight="balanced"` | Cân bằng hai lớp TĂNG/GIẢM |
| `random_state=42` | Giúp kết quả ổn định |

---

## 12. Random Forest

Random Forest gồm nhiều cây quyết định độc lập.

Trong code:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    min_samples_split=60,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

Các tham số như `max_depth`, `min_samples_leaf`, `min_samples_split` giúp hạn chế overfitting vì dữ liệu Bitcoin có độ nhiễu cao.

---

## 13. XGBoost

XGBoost là mô hình boosting gồm nhiều cây quyết định được train tuần tự.

Khác với Random Forest, XGBoost train cây sau để sửa lỗi cho cây trước.

Trong code:

```python
XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.0,
    reg_alpha=0.01,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight
)
```

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `objective="binary:logistic"` | Phân loại nhị phân |
| `n_estimators=500` | Số cây boosting |
| `max_depth=3` | Độ sâu tối đa mỗi cây |
| `learning_rate=0.04` | Tốc độ học |
| `subsample=0.85` | Mỗi cây dùng 85% dữ liệu |
| `colsample_bytree=0.85` | Mỗi cây dùng 85% feature |
| `min_child_weight=3` | Hạn chế chia nhánh quá nhỏ |
| `reg_alpha=0.01` | L1 regularization |
| `reg_lambda=1.0` | L2 regularization |
| `scale_pos_weight` | Cân bằng lớp |

XGBoost có khả năng học quan hệ phi tuyến giữa các feature như RSI, MACD, volume, volatility và return.

---

## 14. Đánh giá bằng TimeSeriesSplit

Vì dữ liệu Bitcoin là chuỗi thời gian, dự án không chia train/test ngẫu nhiên.

Thay vào đó, dự án dùng:

```python
TimeSeriesSplit(n_splits=5, gap=HORIZON)
```

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `n_splits=5` | Chia thành 5 fold theo thời gian |
| `gap=HORIZON` | Tạo khoảng cách 16 nến giữa train và test |

Cách này giúp mô phỏng đúng hơn thực tế:

```text
Train bằng dữ liệu quá khứ
Test trên dữ liệu tương lai
```

---

## 15. Hàm đánh giá mô hình

Sau khi dự đoán xong, dự án dùng các chỉ số:

| Chỉ số | Ý nghĩa |
|---|---|
| `accuracy` | Tỷ lệ dự đoán đúng tổng thể |
| `balanced_accuracy` | Độ chính xác cân bằng giữa TĂNG và GIẢM |
| `macro_f1` | Trung bình F1-score của hai lớp |
| `roc_auc` | Khả năng phân biệt TĂNG/GIẢM bằng xác suất |
| `precision_giam` | Khi dự đoán GIẢM, có bao nhiêu lần đúng |
| `recall_giam` | Trong các lần thật sự GIẢM, mô hình bắt được bao nhiêu |
| `f1_giam` | F1-score của lớp GIẢM |
| `precision_tang` | Khi dự đoán TĂNG, có bao nhiêu lần đúng |
| `recall_tang` | Trong các lần thật sự TĂNG, mô hình bắt được bao nhiêu |
| `f1_tang` | F1-score của lớp TĂNG |

Dự án ưu tiên chọn model theo thứ tự:

```text
macro_f1 → balanced_accuracy → roc_auc
```

Lý do là không muốn model chỉ dự đoán tốt một chiều TĂNG hoặc GIẢM.

---

## 16. Kết quả đầu ra

Sau khi chạy `train_model.py`, chương trình tạo ra:

| File | Ý nghĩa |
|---|---|
| `outputs/timeseries_cv_detail.csv` | Kết quả chi tiết từng fold |
| `outputs/timeseries_cv_summary.csv` | Kết quả trung bình theo từng model |
| `outputs/latest_prediction.csv` | Dự đoán mới nhất |
| `models/best_btc_4h_model.pkl` | Model ML tốt nhất |
| `models/feature_cols.pkl` | Danh sách feature |
| `models/model_metadata.pkl` | Thông tin cấu hình model |

---

## 17. Cài đặt thư viện

Cài thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:

```txt
pandas
numpy
requests
mysql-connector-python
scikit-learn
xgboost
joblib
```

---

## 18. Cách chạy dự án

### Bước 1: Chạy file tạo feature

```bash
python src/features_full.py
```

File này sẽ:

```text
Lấy dữ liệu từ Binance
→ tạo feature
→ lưu CSV
→ lưu vào MySQL
```

### Bước 2: Chạy file train model

```bash
python src/train_model.py
```

File này sẽ:

```text
Đọc feature từ MySQL
→ tạo target
→ train model
→ đánh giá model
→ lưu kết quả
→ dự đoán 4 giờ tới
```

Nếu file không nằm trong thư mục `src`, chạy:

```bash
python features_full.py
python train_model.py
```

---

## 19. Lưu ý khi dùng MySQL

Cần cài MySQL và tạo kết nối đúng với thông tin trong code:

```python
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DATABASE = "crypto_db"
MYSQL_TABLE = "btc_features"
```

Nếu mật khẩu MySQL khác, cần sửa lại `MYSQL_PASSWORD`.

Khi đưa code lên GitHub public, không nên để mật khẩu thật trong code. Có thể dùng biến môi trường hoặc file `.env`.

---

## 20. Nhận xét kết quả

Kết quả thực nghiệm cho thấy bài toán dự đoán xu hướng Bitcoin trong 4 giờ tới là bài toán khó, vì thị trường có nhiều nhiễu.

Nếu Random Walk baseline đạt kết quả cao, điều này cho thấy trong dữ liệu có thể tồn tại momentum ngắn hạn, tức xu hướng gần đây có thể tiếp diễn trong ngắn hạn.

Nếu XGBoost hoặc Logistic Regression đạt kết quả tốt hơn baseline, điều này cho thấy các feature kỹ thuật như RSI, MACD, volume, volatility có thể cung cấp thêm thông tin cho mô hình.

Tuy nhiên, các chỉ số thường chỉ quanh mức gần 0.5, cho thấy việc dự đoán xu hướng Bitcoin ngắn hạn không dễ và cần thêm cải tiến.

---

## 21. Hạn chế

Một số hạn chế hiện tại:

- Chỉ sử dụng dữ liệu OHLCV.
- Chưa dùng dữ liệu order book.
- Chưa dùng funding rate, open interest.
- Chưa dùng tin tức hoặc sentiment thị trường.
- Chưa backtest lợi nhuận thực tế.
- Chưa tính phí giao dịch và trượt giá.
- Mô hình chỉ dự đoán TĂNG/GIẢM, chưa dự đoán mức biến động cụ thể.

---

## 22. Hướng phát triển

Các hướng cải tiến:

- Tự động cập nhật dữ liệu realtime từ Binance.
- Tối ưu hyperparameter bằng TimeSeriesSplit.
- Bổ sung order book, funding rate, open interest.
- Thêm dữ liệu tin tức hoặc sentiment.
- Thêm backtest chiến lược giao dịch.
- Tính phí giao dịch, slippage, stop-loss và take-profit.
- Xây dựng dashboard hiển thị xác suất TĂNG/GIẢM.
- Thử thêm LightGBM, CatBoost, LSTM hoặc Transformer.

---
