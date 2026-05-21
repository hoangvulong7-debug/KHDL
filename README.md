# Dự đoán xu hướng Bitcoin trong 4 giờ tới

## 1. Giới thiệu dự án

Đây là bài tập lớn môn Khoa học dữ liệu. Mục tiêu của dự án là xây dựng mô hình dự đoán xu hướng giá Bitcoin trong 4 giờ tiếp theo dựa trên dữ liệu nến 15 phút của cặp giao dịch BTCUSDT.

Bài toán được xây dựng dưới dạng phân loại nhị phân:

- `0`: GIẢM
- `1`: TĂNG

Thay vì dự đoán giá cụ thể của Bitcoin, dự án tập trung vào việc dự đoán xu hướng tăng hoặc giảm trong tương lai gần.

---

## 2. Mục tiêu bài toán

Mục tiêu chính của dự án là:

- Thu thập và xử lý dữ liệu giá Bitcoin theo khung thời gian 15 phút.
- Tạo các đặc trưng từ 16 nến 15 phút gần nhất, tương đương 4 giờ dữ liệu quá khứ.
- Dự đoán xu hướng giá Bitcoin trong 16 nến tiếp theo, tương đương 4 giờ tương lai.
- So sánh nhiều mô hình dự đoán khác nhau.
- Đánh giá mô hình bằng phương pháp phù hợp với dữ liệu chuỗi thời gian.
- Chọn ra mô hình học máy có kết quả tốt nhất.

---

## 3. Ý tưởng chính

Dự án sử dụng phương pháp rolling window.

Cụ thể:

```text
16 nến 15 phút gần nhất  →  dự đoán 16 nến 15 phút tiếp theo
```

Vì:

```text
16 × 15 phút = 240 phút = 4 giờ
```

Nên bài toán có thể hiểu là:

```text
Dữ liệu 4 giờ quá khứ  →  Dự đoán xu hướng 4 giờ tương lai
```

---

## 4. Dữ liệu đầu vào

Dữ liệu sử dụng là dữ liệu nến 15 phút của BTCUSDT.

Các cột dữ liệu cơ bản gồm:

| Cột | Ý nghĩa |
|---|---|
| `iso` | Thời gian của nến |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất |
| `low` | Giá thấp nhất |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch |
| `trades` | Số lượng giao dịch |

Trong code, file dữ liệu được đọc bằng:

```python
FILE_NAME = "BTCUSDT_XU_HUONG_1NAM_FULL_FEATURE (1).csv"
```

Nếu không tìm thấy file này, chương trình sẽ thử đọc file:

```python
BTCUSDT_XU_HUONG.csv
```

Người dùng cần đặt file CSV trong đúng thư mục chạy chương trình hoặc chỉnh lại biến `FILE_NAME` trong code.

---

## 5. Cấu hình bài toán

Trong code, các tham số chính được khai báo như sau:

```python
WINDOW_SIZE = 16
HORIZON = 16
INTERVAL_MINUTES = 15
THRESHOLD = 0.0015
```

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `WINDOW_SIZE = 16` | Sử dụng 16 nến 15 phút gần nhất làm dữ liệu đầu vào |
| `HORIZON = 16` | Dự đoán sau 16 nến tiếp theo |
| `INTERVAL_MINUTES = 15` | Mỗi nến có độ dài 15 phút |
| `THRESHOLD = 0.0015` | Ngưỡng lọc nhiễu 0.15% |

Ngưỡng `THRESHOLD = 0.0015` tương đương:

```text
0.0015 = 0.15%
```

Mục đích của threshold là loại bỏ các biến động quá nhỏ, vì nếu Bitcoin chỉ tăng hoặc giảm rất nhẹ thì tín hiệu đó có thể chỉ là nhiễu thị trường.

---

## 6. Cách tạo nhãn tăng/giảm

Nhãn được tạo bằng cách so sánh giá đóng cửa sau 16 nến tiếp theo với giá đóng cửa hiện tại.

Công thức:

```python
future_return_4h = future_close_4h / close_current - 1
```

Sau đó gán nhãn:

```text
future_return_4h > +0.15%  →  TĂNG = 1
future_return_4h < -0.15%  →  GIẢM = 0
ở giữa ±0.15%              →  bỏ khỏi tập train
```

Ví dụ:

| Biến động sau 4 giờ | Nhãn |
|---:|---|
| +0.30% | TĂNG |
| -0.25% | GIẢM |
| +0.05% | Bỏ qua |
| -0.08% | Bỏ qua |

Việc loại bỏ các mẫu biến động nhỏ giúp mô hình không phải học các tín hiệu quá nhiễu.

---

## 7. Tạo feature rolling 16 nến

Phần tạo feature là bước quan trọng nhất của dự án.

Với mỗi thời điểm, chương trình lấy 16 nến 15 phút gần nhất để tạo ra một dòng dữ liệu huấn luyện.

Ví dụ:

```text
Tại thời điểm 14:00
Lấy 16 nến từ 10:15 đến 14:00
Dùng các nến này để tạo feature
Sau đó dự đoán xu hướng từ 14:00 đến 18:00
```

Các nhóm feature chính gồm:

### 7.1. Feature của từng nến

| Feature | Ý nghĩa |
|---|---|
| `candle_return` | Lợi suất của từng nến 15 phút |
| `candle_range_pct` | Biên độ high-low của từng nến |
| `candle_body_pct` | Kích thước thân nến |
| `up_candle` | Nến tăng hay giảm |
| `close_return_1` | Tỷ lệ thay đổi close so với nến trước |

---

### 7.2. Feature tổng hợp trong 16 nến

| Feature | Ý nghĩa |
|---|---|
| `window_open` | Giá mở cửa của nến đầu tiên trong window |
| `window_high` | Giá cao nhất trong 16 nến |
| `window_low` | Giá thấp nhất trong 16 nến |
| `window_close` | Giá đóng cửa hiện tại |
| `window_return` | Lợi suất của toàn bộ window |
| `window_range` | Biên độ dao động trong 16 nến |
| `window_body_size` | Độ lớn thay đổi giá từ đầu đến cuối window |

---

### 7.3. Feature về khối lượng giao dịch

| Feature | Ý nghĩa |
|---|---|
| `window_volume_sum` | Tổng volume trong 16 nến |
| `window_volume_mean` | Volume trung bình |
| `window_volume_std` | Độ lệch chuẩn của volume |
| `volume_change` | Mức thay đổi volume so với nến trước |
| `volume_ratio_16` | Volume hiện tại so với trung bình 16 nến |

---

### 7.4. Feature về xu hướng và biến động

| Feature | Ý nghĩa |
|---|---|
| `mean_15m_return` | Return trung bình của các nến 15 phút |
| `std_15m_return` | Độ biến động của return |
| `max_15m_return` | Nến tăng mạnh nhất trong window |
| `min_15m_return` | Nến giảm mạnh nhất trong window |
| `up_candle_ratio` | Tỷ lệ nến tăng trong 16 nến |
| `close_position` | Vị trí giá đóng cửa so với vùng high-low |
| `volatility_16` | Độ biến động trong 16 nến |
| `volatility_32` | Độ biến động trong 32 nến |

---

### 7.5. Chỉ báo kỹ thuật

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

### 7.6. Khoảng cách giá so với đường trung bình

| Feature | Ý nghĩa |
|---|---|
| `distance_ma7` | Khoảng cách giá hiện tại so với MA7 |
| `distance_ma25` | Khoảng cách giá hiện tại so với MA25 |
| `distance_ma99` | Khoảng cách giá hiện tại so với MA99 |

Các feature này giúp mô hình biết giá hiện tại đang nằm trên hay dưới xu hướng trung bình.

---

## 8. Kiểm tra window hợp lệ

Code kiểm tra xem 16 nến quá khứ có liên tục hay không:

```python
valid_past_window
```

Điều kiện:

```text
Có đủ 16 nến
Và thời gian từ nến đầu đến nến cuối đúng bằng 3 giờ 45 phút
```

Lý do là 16 nến có 15 khoảng cách giữa các nến:

```text
15 khoảng × 15 phút = 225 phút = 3 giờ 45 phút
```

Ngoài ra, code cũng kiểm tra dữ liệu tương lai có đúng cách 4 giờ không:

```python
valid_future_window
```

Việc kiểm tra này giúp tránh trường hợp dữ liệu bị thiếu nến làm sai lệch quá trình train.

---

## 9. Các mô hình sử dụng

Dự án so sánh các mô hình sau:

1. Random Walk baseline
2. Logistic Regression
3. Random Forest
4. XGBoost

---

## 10. Random Walk baseline

Random Walk là mô hình baseline đơn giản, không phải mô hình học máy.

Quy tắc:

```text
Nếu 16 nến gần nhất đang tăng  →  dự đoán TĂNG
Nếu 16 nến gần nhất đang giảm  →  dự đoán GIẢM
```

Random Walk được dùng để kiểm tra xem các mô hình học máy có thật sự tốt hơn một quy tắc đơn giản hay không.

Nếu mô hình học máy không vượt được Random Walk, điều đó cho thấy dữ liệu còn nhiễu hoặc feature chưa đủ mạnh.

---

## 11. Logistic Regression

Logistic Regression là mô hình tuyến tính dùng để phân loại nhị phân.

Mô hình học một công thức dạng:

```text
score = w1*x1 + w2*x2 + ... + b
```

Sau đó đưa qua hàm sigmoid để tạo xác suất tăng.

Trong code, Logistic Regression được đặt trong pipeline:

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
| `StandardScaler()` | Chuẩn hóa dữ liệu đầu vào |
| `max_iter=1000` | Số vòng lặp tối đa khi train |
| `class_weight="balanced"` | Cân bằng trọng số hai lớp |
| `random_state=42` | Giúp kết quả ổn định hơn |

Logistic Regression thường đơn giản, dễ giải thích và ít bị overfitting hơn các mô hình phức tạp.

---

## 12. Random Forest

Random Forest là mô hình ensemble gồm nhiều cây quyết định độc lập.

Mỗi cây đưa ra một dự đoán, sau đó mô hình lấy kết quả theo đa số phiếu.

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

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `n_estimators=300` | Số lượng cây |
| `max_depth=5` | Giới hạn độ sâu của cây |
| `min_samples_leaf=30` | Số mẫu tối thiểu ở mỗi lá |
| `min_samples_split=60` | Số mẫu tối thiểu để tiếp tục tách |
| `max_features="sqrt"` | Mỗi cây chỉ dùng một phần feature |
| `class_weight="balanced_subsample"` | Cân bằng lớp trong từng cây |
| `n_jobs=-1` | Dùng tối đa CPU để train |

Các tham số này giúp giảm overfitting vì dữ liệu Bitcoin có nhiều nhiễu.

---

## 13. XGBoost

XGBoost là viết tắt của Extreme Gradient Boosting.

Đây là mô hình ensemble gồm nhiều cây quyết định, nhưng khác Random Forest ở cách train.

Random Forest train nhiều cây độc lập rồi bỏ phiếu.

XGBoost train cây theo thứ tự:

```text
Cây 1 dự đoán
→ tính lỗi
→ cây 2 sửa lỗi cây 1
→ cây 3 sửa lỗi tiếp
→ ...
```

Nói cách khác, cây sau tập trung sửa lỗi của cây trước.

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

Ý nghĩa các tham số:

| Tham số | Ý nghĩa |
|---|---|
| `objective="binary:logistic"` | Bài toán phân loại nhị phân |
| `n_estimators=500` | Số cây boosting |
| `max_depth=3` | Độ sâu tối đa của mỗi cây |
| `learning_rate=0.04` | Tốc độ học |
| `subsample=0.85` | Mỗi cây dùng 85% dữ liệu |
| `colsample_bytree=0.85` | Mỗi cây dùng 85% feature |
| `min_child_weight=3` | Hạn chế chia nhánh quá nhỏ |
| `gamma=0.0` | Mức phạt khi tách nhánh |
| `reg_alpha=0.01` | L1 regularization |
| `reg_lambda=1.0` | L2 regularization |
| `eval_metric="logloss"` | Hàm đánh giá trong lúc train |
| `tree_method="hist"` | Phương pháp train cây nhanh hơn |
| `scale_pos_weight` | Cân bằng lớp TĂNG/GIẢM |

XGBoost mạnh vì có thể học được các quan hệ phi tuyến giữa các feature như RSI, MACD, volume, volatility và return.

Tuy nhiên, do dữ liệu tài chính rất nhiễu, XGBoost cũng có nguy cơ overfitting nếu mô hình quá phức tạp.

---

## 14. Đánh giá bằng TimeSeriesSplit

Vì dữ liệu Bitcoin là dữ liệu chuỗi thời gian, dự án không chia train/test ngẫu nhiên.

Nếu shuffle dữ liệu, thông tin tương lai có thể bị lẫn vào tập train, gây sai lệch kết quả.

Do đó, dự án sử dụng:

```python
TimeSeriesSplit(n_splits=5, gap=HORIZON)
```

Ý nghĩa:

| Tham số | Ý nghĩa |
|---|---|
| `n_splits=5` | Chia dữ liệu thành 5 fold theo thời gian |
| `gap=HORIZON` | Tạo khoảng cách giữa train và test |

Cách hoạt động:

```text
Fold 1: train quá khứ ngắn      → test đoạn tương lai
Fold 2: train quá khứ dài hơn   → test đoạn tương lai tiếp theo
Fold 3: train dài hơn nữa       → test đoạn tiếp theo
...
```

Cách này phù hợp với bài toán dự đoán thời gian vì mô hình luôn được train bằng dữ liệu quá khứ và test trên dữ liệu tương lai.

---

## 15. Hàm đánh giá mô hình

Sau khi mỗi mô hình dự đoán xong, dự án sử dụng hàm `evaluate_model()` để tính các chỉ số đánh giá.

Các chỉ số gồm:

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

Dự án không chỉ dùng Accuracy vì Accuracy có thể gây hiểu nhầm nếu mô hình bị lệch về một lớp.

Ví dụ, nếu mô hình thường xuyên dự đoán TĂNG, nó có thể đúng nhiều khi thị trường tăng nhiều, nhưng lại dự đoán GIẢM rất kém.

Vì vậy, dự án ưu tiên:

```text
macro_f1 → balanced_accuracy → roc_auc
```

---

## 16. Kết quả đầu ra

Sau khi chạy chương trình, các file kết quả được tạo ra:

| File | Ý nghĩa |
|---|---|
| `timeseries_cv_detail.csv` | Kết quả chi tiết từng fold |
| `timeseries_cv_summary.csv` | Kết quả trung bình theo từng model |
| `best_btc_4h_model.pkl` | Model tốt nhất đã được lưu |
| `feature_cols.pkl` | Danh sách feature dùng khi train |
| `model_metadata.pkl` | Thông tin cấu hình của mô hình |

---

## 17. Cách chọn mô hình tốt nhất

Sau khi đánh giá bằng TimeSeriesSplit, chương trình tổng hợp kết quả trung bình của từng model.

Mô hình được sắp xếp theo:

```text
macro_f1
balanced_accuracy
roc_auc
```

Mô hình có điểm cao nhất theo thứ tự này được chọn làm mô hình tốt nhất trong nhóm mô hình học máy.

Nếu Random Walk baseline có kết quả cao, nó vẫn được xem là baseline để tham khảo, nhưng mô hình cuối cùng nên được chọn trong nhóm mô hình học máy như Logistic Regression, Random Forest hoặc XGBoost.

---

## 18. Dự đoán bằng 16 nến gần nhất

Sau khi train xong, chương trình tiếp tục lấy 16 nến 15 phút gần nhất để dự đoán xu hướng 4 giờ tới.

Kết quả in ra gồm:

```text
Dự đoán: TĂNG hoặc GIẢM
Xác suất GIẢM
Xác suất TĂNG
```

Ví dụ:

```text
XGBoost:
Dự đoán: TĂNG
Xác suất GIẢM: 0.430
Xác suất TĂNG: 0.570
```

Điều này giúp người dùng không chỉ biết mô hình dự đoán tăng hay giảm, mà còn biết mức độ tự tin của mô hình.

---

## 19. Cách cài đặt

Cài các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nội dung file `requirements.txt`:

```txt
pandas
numpy
scikit-learn
xgboost
joblib
```

---

## 20. Cách chạy

Chạy file train:

```bash
python src/train_model.py
```

Nếu file code không nằm trong thư mục `src`, có thể chạy:

```bash
python train_model.py
```

Lưu ý: cần đặt file CSV dữ liệu đúng tên hoặc sửa lại biến `FILE_NAME` trong code.

---

## 21. Cấu trúc thư mục đề xuất

```text
KHDL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── train_model.py
│
├── outputs/
│   ├── timeseries_cv_detail.csv
│   └── timeseries_cv_summary.csv
│
├── models/
│   ├── best_btc_4h_model.pkl
│   ├── feature_cols.pkl
│   └── model_metadata.pkl
│
└── data/
    └── BTCUSDT_XU_HUONG_1NAM_FULL_FEATURE.csv
```

---

## 22. Nhận xét kết quả

Kết quả thực nghiệm cho thấy bài toán dự đoán xu hướng Bitcoin trong 4 giờ tới là bài toán khó do thị trường có nhiều nhiễu.

Một mô hình phức tạp hơn như XGBoost không phải lúc nào cũng chắc chắn vượt trội tuyệt đối so với các mô hình đơn giản hơn.

Vì vậy, dự án so sánh nhiều mô hình và sử dụng TimeSeriesSplit để đánh giá công bằng hơn.

Nếu Random Walk baseline đạt kết quả cao, điều đó cho thấy thị trường trong giai đoạn kiểm thử có thể tồn tại yếu tố momentum ngắn hạn, tức xu hướng gần nhất có thể tiếp diễn trong ngắn hạn.

Nếu XGBoost hoặc Logistic Regression đạt kết quả tốt hơn, điều đó cho thấy các feature kỹ thuật như RSI, MACD, volume, volatility có thể cung cấp thêm thông tin cho mô hình.

---

## 23. Hạn chế của dự án

Một số hạn chế hiện tại:

- Chỉ sử dụng dữ liệu OHLCV, chưa dùng dữ liệu order book.
- Chưa sử dụng tin tức, tâm lý thị trường hoặc dữ liệu vĩ mô.
- Chưa backtest lợi nhuận thực tế.
- Chưa tính phí giao dịch, trượt giá và quản trị rủi ro.
- Mô hình chỉ dự đoán xu hướng tăng/giảm, chưa dự đoán mức tăng/giảm cụ thể.
- Dữ liệu Bitcoin có độ nhiễu cao nên kết quả có thể thay đổi theo từng giai đoạn thị trường.

---

## 24. Hướng phát triển

Trong tương lai, dự án có thể được cải tiến theo các hướng sau:

- Kết nối Binance API để lấy dữ liệu realtime.
- Tự động cập nhật dữ liệu mỗi 15 phút.
- Tối ưu hyperparameter bằng TimeSeriesSplit.
- Bổ sung feature từ order book, funding rate, open interest.
- Bổ sung dữ liệu tin tức hoặc sentiment thị trường.
- Xây dựng dashboard hiển thị xác suất tăng/giảm.
- Thêm backtest chiến lược giao dịch dựa trên tín hiệu mô hình.
- Tính thêm phí giao dịch, slippage, stop-loss và take-profit.
- Thử thêm các mô hình khác như LightGBM, CatBoost, LSTM hoặc Transformer.

---

## 25. Kết luận

Dự án đã xây dựng một pipeline hoàn chỉnh để dự đoán xu hướng Bitcoin trong 4 giờ tới.

Pipeline gồm các bước:

```text
Đọc dữ liệu nến 15 phút
→ Tạo feature rolling 16 nến
→ Tạo nhãn tăng/giảm sau 16 nến
→ Lọc nhiễu bằng threshold
→ Train nhiều mô hình
→ Đánh giá bằng TimeSeriesSplit
→ So sánh bằng nhiều chỉ số
→ Lưu model tốt nhất
→ Dự đoán xu hướng 4 giờ tới
```

Dự án cho thấy việc dự đoán xu hướng Bitcoin ngắn hạn là một bài toán khó, nhưng thông qua việc tạo feature kỹ thuật, dùng threshold giảm nhiễu và đánh giá bằng TimeSeriesSplit, mô hình có thể được kiểm tra một cách nghiêm túc và phù hợp hơn với dữ liệu chuỗi thời gian.
