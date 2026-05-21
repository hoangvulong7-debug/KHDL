import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Chưa cài xgboost. Nếu muốn dùng XGBoost, chạy: pip install xgboost")


# =========================
# 1. Đọc dữ liệu 15 phút
# =========================

FILE_NAME = "BTCUSDT_XU_HUONG_1NAM_FULL_FEATURE (1).csv"

try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    FILE_NAME = "BTCUSDT_XU_HUONG.csv"
    df = pd.read_csv(FILE_NAME)

print("Đang đọc file:", FILE_NAME)

df["iso"] = pd.to_datetime(df["iso"], utc=True)
df = df.sort_values("iso").drop_duplicates(subset=["iso"]).reset_index(drop=True)

required_cols = ["open", "high", "low", "close", "volume"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"File thiếu cột bắt buộc: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "trades" in df.columns:
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
else:
    df["trades"] = 0


# =========================
# 2. Cấu hình bài toán
# =========================

WINDOW_SIZE = 16
HORIZON = 16
INTERVAL_MINUTES = 15
THRESHOLD = 0.0015


# =========================
# 3. Tạo feature rolling 16 nến
# =========================

def safe_div_series(a, b):
    result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_rsi(close_series, window=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_rolling_16_features(raw_df):
    d = raw_df.copy()

    d["candle_return"] = safe_div_series(d["close"] - d["open"], d["open"])
    d["candle_range_pct"] = safe_div_series(d["high"] - d["low"], d["open"])
    d["candle_body_pct"] = safe_div_series((d["close"] - d["open"]).abs(), d["open"])
    d["up_candle"] = (d["close"] > d["open"]).astype(int)
    d["close_return_1"] = d["close"].pct_change(1)

    f = pd.DataFrame(index=d.index)
    f["iso"] = d["iso"]

    f["hour"] = d["iso"].dt.hour
    f["day"] = d["iso"].dt.day
    f["month"] = d["iso"].dt.month
    f["day_of_week"] = d["iso"].dt.dayofweek
    f["weekend"] = (f["day_of_week"] >= 5).astype(int)

    f["open_current"] = d["open"]
    f["high_current"] = d["high"]
    f["low_current"] = d["low"]
    f["close_current"] = d["close"]
    f["volume_current"] = d["volume"]
    f["trades_current"] = d["trades"]

    f["window_start_time"] = d["iso"].shift(WINDOW_SIZE - 1)
    f["window_open"] = d["open"].shift(WINDOW_SIZE - 1)
    f["window_high"] = d["high"].rolling(WINDOW_SIZE).max()
    f["window_low"] = d["low"].rolling(WINDOW_SIZE).min()
    f["window_close"] = d["close"]
    f["candle_count"] = d["close"].rolling(WINDOW_SIZE).count()

    f["window_return"] = safe_div_series(f["window_close"] - f["window_open"], f["window_open"])
    f["window_range"] = safe_div_series(f["window_high"] - f["window_low"], f["window_open"])
    f["window_body_size"] = safe_div_series((f["window_close"] - f["window_open"]).abs(), f["window_open"])

    f["window_volume_sum"] = d["volume"].rolling(WINDOW_SIZE).sum()
    f["window_volume_mean"] = d["volume"].rolling(WINDOW_SIZE).mean()
    f["window_volume_std"] = d["volume"].rolling(WINDOW_SIZE).std(ddof=0)

    f["window_trades_sum"] = d["trades"].rolling(WINDOW_SIZE).sum()
    f["window_trades_mean"] = d["trades"].rolling(WINDOW_SIZE).mean()

    f["mean_15m_return"] = d["candle_return"].rolling(WINDOW_SIZE).mean()
    f["std_15m_return"] = d["candle_return"].rolling(WINDOW_SIZE).std(ddof=0)
    f["max_15m_return"] = d["candle_return"].rolling(WINDOW_SIZE).max()
    f["min_15m_return"] = d["candle_return"].rolling(WINDOW_SIZE).min()

    f["mean_15m_range"] = d["candle_range_pct"].rolling(WINDOW_SIZE).mean()
    f["std_15m_range"] = d["candle_range_pct"].rolling(WINDOW_SIZE).std(ddof=0)

    f["mean_body_pct"] = d["candle_body_pct"].rolling(WINDOW_SIZE).mean()
    f["up_candle_ratio"] = d["up_candle"].rolling(WINDOW_SIZE).mean()

    f["close_position"] = safe_div_series(
        f["window_close"] - f["window_low"],
        f["window_high"] - f["window_low"]
    ).fillna(0.5)

    f["return_1"] = d["close"].pct_change(1)
    f["return_2"] = d["close"].pct_change(2)
    f["return_4"] = d["close"].pct_change(4)
    f["return_8"] = d["close"].pct_change(8)
    f["return_16"] = d["close"].pct_change(16)

    f["close_lag1"] = d["close"].shift(1)
    f["close_lag4"] = d["close"].shift(4)
    f["close_lag8"] = d["close"].shift(8)
    f["close_lag16"] = d["close"].shift(16)

    f["MA7"] = d["close"].rolling(7).mean()
    f["MA25"] = d["close"].rolling(25).mean()
    f["MA99"] = d["close"].rolling(99).mean()

    f["EMA12"] = d["close"].ewm(span=12, adjust=False).mean()
    f["EMA26"] = d["close"].ewm(span=26, adjust=False).mean()
    f["MACD"] = f["EMA12"] - f["EMA26"]
    f["MACD_signal"] = f["MACD"].ewm(span=9, adjust=False).mean()

    f["RSI"] = add_rsi(d["close"], window=14)

    ma20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std(ddof=0)

    f["BB_upper"] = ma20 + 2 * std20
    f["BB_lower"] = ma20 - 2 * std20
    f["BB_width"] = safe_div_series(f["BB_upper"] - f["BB_lower"], ma20)

    f["volatility_16"] = d["close_return_1"].rolling(WINDOW_SIZE).std(ddof=0)
    f["volatility_32"] = d["close_return_1"].rolling(32).std(ddof=0)

    f["volume_ma16"] = d["volume"].rolling(16).mean()
    f["volume_ma64"] = d["volume"].rolling(64).mean()
    f["volume_change"] = d["volume"].pct_change(1)
    f["volume_ratio_16"] = safe_div_series(d["volume"], f["volume_ma16"])

    f["distance_ma7"] = safe_div_series(d["close"] - f["MA7"], f["MA7"])
    f["distance_ma25"] = safe_div_series(d["close"] - f["MA25"], f["MA25"])
    f["distance_ma99"] = safe_div_series(d["close"] - f["MA99"], f["MA99"])

    expected_past_delta = pd.Timedelta(minutes=INTERVAL_MINUTES * (WINDOW_SIZE - 1))

    f["valid_past_window"] = (
        (f["candle_count"] == WINDOW_SIZE)
        & ((f["iso"] - f["window_start_time"]) == expected_past_delta)
    )

    f["future_time"] = d["iso"].shift(-HORIZON)
    f["future_close_4h"] = d["close"].shift(-HORIZON)

    expected_future_delta = pd.Timedelta(minutes=INTERVAL_MINUTES * HORIZON)
    f["valid_future_window"] = (f["future_time"] - f["iso"]) == expected_future_delta

    f["future_return_4h"] = f["future_close_4h"] / f["close_current"] - 1

    f["target_up_4h"] = np.nan

    f.loc[
        f["valid_past_window"]
        & f["valid_future_window"]
        & (f["future_return_4h"] > THRESHOLD),
        "target_up_4h"
    ] = 1

    f.loc[
        f["valid_past_window"]
        & f["valid_future_window"]
        & (f["future_return_4h"] < -THRESHOLD),
        "target_up_4h"
    ] = 0

    f = f.set_index("iso")
    return f


feature_df = make_rolling_16_features(df)


# =========================
# 4. Chuẩn bị dữ liệu
# =========================

drop_cols = [
    "window_start_time",
    "valid_past_window",
    "future_time",
    "valid_future_window",
    "future_close_4h",
    "future_return_4h",
    "target_up_4h"
]

feature_cols = [c for c in feature_df.columns if c not in drop_cols]

model_data_all = feature_df.replace([np.inf, -np.inf], np.nan)

valid_train_candidates = model_data_all[
    model_data_all["valid_past_window"] & model_data_all["valid_future_window"]
].copy()

noisy_samples = valid_train_candidates["target_up_4h"].isna().sum()
clear_samples = valid_train_candidates["target_up_4h"].notna().sum()

print("\n===== THỐNG KÊ ROLLING 16 NẾN =====")
print("WINDOW_SIZE:", WINDOW_SIZE, "nến 15 phút")
print("HORIZON:", HORIZON, "nến 15 phút = 4 giờ")
print("Threshold:", THRESHOLD, "=", THRESHOLD * 100, "%")
print("Tổng số dòng 15m:", len(df))
print("Số mẫu có đủ 16 nến quá khứ và đủ 16 nến tương lai:", len(valid_train_candidates))
print("Số mẫu biến động rõ dùng để train:", clear_samples)
print("Số mẫu biến động nhỏ bị loại khỏi train:", noisy_samples)

if len(valid_train_candidates) > 0:
    print("Tỷ lệ bị loại vì nhiễu:", round(noisy_samples / len(valid_train_candidates) * 100, 2), "%")

model_data = model_data_all.dropna(subset=feature_cols + ["target_up_4h"]).copy()
model_data = model_data[
    model_data["valid_past_window"] & model_data["valid_future_window"]
].copy()

model_data["target_up_4h"] = model_data["target_up_4h"].astype(int)

X = model_data[feature_cols]
y = model_data["target_up_4h"]

if len(X) < 50:
    raise ValueError("Dữ liệu sau khi lọc threshold còn quá ít. Hãy giảm THRESHOLD.")

if y.nunique() < 2:
    raise ValueError("Sau khi lọc threshold chỉ còn 1 lớp nhãn. Hãy giảm THRESHOLD.")

print("\n===== THỐNG KÊ DATA SAU KHI LỌC =====")
print("Tổng số mẫu dùng được:", len(X))
print("Số feature:", len(feature_cols))
print("Tỷ lệ nhãn trong toàn bộ dữ liệu dùng được:")
print(y.value_counts(normalize=True).rename({0: "GIẢM", 1: "TĂNG"}))


# =========================
# 5. Khai báo model
# =========================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        min_samples_split=60,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

if HAS_XGB:
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1

    models["XGBoost"] = XGBClassifier(
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


# =========================
# 6. Hàm đánh giá
# =========================

def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def evaluate_model(name, y_true, y_pred, y_prob=None):
    if y_prob is None:
        y_prob = y_pred

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_prob),

        "precision_giam": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_giam": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_giam": f1_score(y_true, y_pred, pos_label=0, zero_division=0),

        "precision_tang": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_tang": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_tang": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


# =========================
# 7. TimeSeriesSplit
# =========================

tscv = TimeSeriesSplit(n_splits=5, gap=HORIZON)

cv_results = []


# =========================
# 7.1. Baseline Random Walk
# =========================
# SỬA: thêm lại baseline Random Walk.
# Random Walk không cần train. Quy tắc:
# Nếu 16 nến gần nhất đang tăng -> dự đoán 4 giờ tới TĂNG
# Nếu 16 nến gần nhất đang giảm hoặc bằng 0 -> dự đoán 4 giờ tới GIẢM

print(f"\n{'='*60}")
print("ĐÁNH GIÁ MODEL: Random Walk")
print(f"{'='*60}")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    y_pred = (X_test["window_return"] > 0).astype(int)
    y_prob = y_pred.astype(float)

    result = evaluate_model(
        name=f"Random Walk - Fold {fold}",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob
    )

    result["model"] = "Random Walk"
    result["fold"] = fold
    result["train_from"] = X_train.index.min()
    result["train_to"] = X_train.index.max()
    result["test_from"] = X_test.index.min()
    result["test_to"] = X_test.index.max()
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)

    cv_results.append(result)

    print(
        f"Fold {fold}: "
        f"Train {X_train.index.min()} -> {X_train.index.max()} | "
        f"Test {X_test.index.min()} -> {X_test.index.max()} | "
        f"Macro F1 = {result['macro_f1']:.4f}, "
        f"Balanced Acc = {result['balanced_accuracy']:.4f}, "
        f"ROC-AUC = {result['roc_auc']:.4f}"
    )


for name, model in models.items():

    print(f"\n{'='*60}")
    print(f"ĐÁNH GIÁ MODEL: {name}")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        current_model = clone(model)
        current_model.fit(X_train, y_train)

        y_pred = current_model.predict(X_test)

        if hasattr(current_model, "predict_proba"):
            y_prob = current_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        result = evaluate_model(
            name=f"{name} - Fold {fold}",
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob
        )

        result["model"] = name
        result["fold"] = fold
        result["train_from"] = X_train.index.min()
        result["train_to"] = X_train.index.max()
        result["test_from"] = X_test.index.min()
        result["test_to"] = X_test.index.max()
        result["train_size"] = len(X_train)
        result["test_size"] = len(X_test)

        cv_results.append(result)

        print(
            f"Fold {fold}: "
            f"Train {X_train.index.min()} -> {X_train.index.max()} | "
            f"Test {X_test.index.min()} -> {X_test.index.max()} | "
            f"Macro F1 = {result['macro_f1']:.4f}, "
            f"Balanced Acc = {result['balanced_accuracy']:.4f}, "
            f"ROC-AUC = {result['roc_auc']:.4f}"
        )

cv_results_df = pd.DataFrame(cv_results)


# =========================
# 8. Bảng so sánh
# =========================

summary_df = (
    cv_results_df
    .groupby("model")
    .agg({
        "accuracy": "mean",
        "balanced_accuracy": "mean",
        "macro_f1": "mean",
        "roc_auc": "mean",
        "precision_giam": "mean",
        "recall_giam": "mean",
        "f1_giam": "mean",
        "precision_tang": "mean",
        "recall_tang": "mean",
        "f1_tang": "mean"
    })
    .reset_index()
    .sort_values(
        by=["macro_f1", "balanced_accuracy", "roc_auc"],
        ascending=False
    )
)

print("\n===== KẾT QUẢ TRUNG BÌNH TIME SERIES SPLIT =====")
print(summary_df)

print("\n===== KẾT QUẢ CHI TIẾT TỪNG FOLD =====")
print(cv_results_df)


# =========================
# 9. Lưu kết quả
# =========================

cv_results_df.to_csv("timeseries_cv_detail.csv", index=False)
summary_df.to_csv("timeseries_cv_summary.csv", index=False)

print("\nĐã lưu kết quả chi tiết từng fold vào: timeseries_cv_detail.csv")
print("Đã lưu bảng tổng hợp trung bình vào: timeseries_cv_summary.csv")


# =========================
# 10. Train lại model machine learning tốt nhất trên toàn bộ dữ liệu
# =========================

# SỬA: vì đã thêm Random Walk vào bảng so sánh, model tốt nhất tổng thể có thể là Random Walk.
# Nhưng Random Walk là baseline, không có model sklearn để fit/save.
# Vì vậy:
# - best_overall_name: model tốt nhất tổng thể, có thể là Random Walk
# - best_ml_name: model machine learning tốt nhất, dùng để train lại và lưu .pkl

ml_summary_df = summary_df[summary_df["model"] != "Random Walk"].copy()

if ml_summary_df.empty:
    raise ValueError("Không có mô hình ML nào để chọn.")

best_ml_name = ml_summary_df.iloc[0]["model"]
best_model = models[best_ml_name]

print("\n===== MODEL MACHINE LEARNING TỐT NHẤT =====")
print("ML model tốt nhất:", best_ml_name)
print("Tiêu chí chọn: macro_f1 → balanced_accuracy → roc_auc")
print("Lưu ý: Random Walk chỉ được dùng làm baseline so sánh, không dùng để lưu model.")

final_model = clone(best_model)
final_model.fit(X, y)

joblib.dump(final_model, "best_btc_4h_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

joblib.dump(
    {
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "interval_minutes": INTERVAL_MINUTES,
        "threshold": THRESHOLD,
        "label_map": {0: "GIẢM", 1: "TĂNG"},
        "method": "rolling_16_candles_predict_next_16_candles_timeseries_split",
        "best_ml_model": best_ml_name,
        "baseline": "Random Walk",
        "cv_n_splits": 5,
        "cv_gap": HORIZON,
        "selection_metric_order": ["macro_f1", "balanced_accuracy", "roc_auc"]
    },
    "model_metadata.pkl"
)

print("\nĐã lưu model machine learning tốt nhất vào: best_btc_4h_model.pkl")
print("Đã lưu danh sách feature vào: feature_cols.pkl")
print("Đã lưu metadata vào: model_metadata.pkl")


# =========================
# 11. Train full từng model để dự đoán cuối
# =========================

trained_full_models = {}

for name, model in models.items():
    model_full = clone(model)
    model_full.fit(X, y)
    trained_full_models[name] = model_full


# =========================
# 12. Dự đoán bằng 16 nến gần nhất
# =========================

all_feature_data = feature_df.replace([np.inf, -np.inf], np.nan)

latest_data = all_feature_data.dropna(subset=feature_cols).copy()
latest_data = latest_data[latest_data["valid_past_window"]].copy()

if latest_data.empty:
    raise ValueError("Không tìm thấy mẫu nào có đủ 16 nến 15 phút liên tục để dự đoán.")

latest_X = latest_data[feature_cols].iloc[[-1]]
latest_time = latest_X.index[-1]

window_start = latest_data.loc[latest_time, "window_start_time"]
predict_start = latest_time
predict_end = latest_time + pd.Timedelta(minutes=INTERVAL_MINUTES * HORIZON)

print("\n===== DỰ ĐOÁN BẰNG 16 NẾN 15 PHÚT GẦN NHẤT =====")
print("Window dùng để dự đoán:", window_start, "→", latest_time)
print("Số nến 15 phút trong window:", int(latest_data.loc[latest_time, "candle_count"]))
print("Dự đoán xu hướng 4 giờ tới:", predict_start, "→", predict_end)

# SỬA: thêm lại dự đoán baseline Random Walk ở phần predict cuối.
# Quy tắc giống lúc cross-validation: 16 nến gần nhất tăng thì dự đoán TĂNG, ngược lại GIẢM.
rw_pred_latest = int(latest_X["window_return"].iloc[0] > 0)
rw_trend = "TĂNG" if rw_pred_latest == 1 else "GIẢM"
rw_prob_down = 1.0 if rw_pred_latest == 0 else 0.0
rw_prob_up = 1.0 if rw_pred_latest == 1 else 0.0

print("\nRandom Walk:")
print(f"Dự đoán: {rw_trend}")
print(f"Xác suất GIẢM: {rw_prob_down:.3f}")
print(f"Xác suất TĂNG: {rw_prob_up:.3f}")

for name, model_full in trained_full_models.items():
    pred = model_full.predict(latest_X)[0]

    if hasattr(model_full, "predict_proba"):
        prob = model_full.predict_proba(latest_X)[0]
        classes = model_full.classes_

        prob_down = prob[list(classes).index(0)]
        prob_up = prob[list(classes).index(1)]
    else:
        prob_down = np.nan
        prob_up = np.nan

    trend = "TĂNG" if pred == 1 else "GIẢM"

    print(f"\n{name}:")
    print(f"Dự đoán: {trend}")
    print(f"Xác suất GIẢM: {prob_down:.3f}")
    print(f"Xác suất TĂNG: {prob_up:.3f}")
