import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mysql.connector
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Chưa cài xgboost. Nếu muốn dùng XGBoost, chạy: pip install xgboost")


warnings.filterwarnings("ignore")


# =========================
# 1. CONFIG
# =========================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

WINDOW_SIZE = 16
HORIZON = 16
INTERVAL_MINUTES = 15
THRESHOLD = 0.0015  # 0.15%

N_SPLITS = 5

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# =========================
# 2. MYSQL CONFIG
# =========================
# Có thể dùng biến môi trường để đỡ lộ mật khẩu khi đưa lên GitHub.
# Nếu không có biến môi trường thì dùng giá trị mặc định bên dưới.

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "crypto_db")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "btc_features")


# =========================
# 3. DANH SÁCH CỘT FEATURE TRONG MYSQL
# =========================
# Danh sách này phải khớp với FINAL_COLUMNS trong features_full.py

FINAL_COLUMNS = [
    "iso",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",

    "candle_return",
    "candle_range_pct",
    "candle_body_pct",
    "up_candle",
    "close_return_1",

    "window_open",
    "window_high",
    "window_low",
    "window_close",
    "window_return",
    "window_range",
    "window_body_size",
    "window_volume_sum",
    "window_volume_mean",
    "window_volume_std",
    "volume_change",
    "volume_ratio_16",

    "mean_15m_return",
    "std_15m_return",
    "max_15m_return",
    "min_15m_return",
    "up_candle_ratio",
    "close_position",
    "volatility_16",
    "volatility_32",

    "MA7",
    "MA25",
    "MA99",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_signal",
    "RSI",
    "BB_upper",
    "BB_lower",
    "BB_width",

    "distance_ma7",
    "distance_ma25",
    "distance_ma99",
]


SQL_COLUMNS = [
    "ts",
    "symbol",
    "interval_name",
] + FINAL_COLUMNS


# =========================
# 4. ĐỌC DỮ LIỆU TỪ MYSQL
# =========================

def load_features_from_mysql():
    print("\n===== ĐỌC FEATURE TỪ MYSQL =====")
    print("Database:", MYSQL_DATABASE)
    print("Table   :", MYSQL_TABLE)
    print("Symbol  :", SYMBOL)
    print("Interval:", INTERVAL)

    db = None
    cursor = None

    try:
        db = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )

        cursor = db.cursor()

        select_cols_sql = ", ".join([f"`{col}`" for col in SQL_COLUMNS])

        sql = f"""
        SELECT
            {select_cols_sql}
        FROM `{MYSQL_TABLE}`
        WHERE `symbol` = %s
          AND `interval_name` = %s
        ORDER BY `ts` ASC
        """

        cursor.execute(sql, (SYMBOL, INTERVAL))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=columns)

    except mysql.connector.Error as err:
        raise RuntimeError(f"Lỗi MySQL: {err}")

    finally:
        if cursor is not None:
            cursor.close()

        if db is not None:
            db.close()

    if df.empty:
        raise ValueError(
            "Không đọc được dữ liệu từ MySQL. "
            "Hãy chạy file features_full.py trước để insert dữ liệu vào bảng btc_features."
        )

    print("Số dòng đọc được từ MySQL:", len(df))

    return df


# =========================
# 5. KIỂM TRA VÀ LÀM SẠCH DỮ LIỆU
# =========================

def validate_and_clean_sql_data(df):
    df = df.copy()

    missing_cols = [col for col in SQL_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Thiếu cột trong MySQL: {missing_cols}")

    df["iso"] = pd.to_datetime(df["iso"], utc=True)

    for col in df.columns:
        if col not in ["iso", "symbol", "interval_name"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("iso").drop_duplicates(subset=["ts"], keep="last")
    df = df.reset_index(drop=True)

    print("\n===== THÔNG TIN DỮ LIỆU SAU KHI CLEAN =====")
    print("Số dòng:", len(df))
    print("Từ:", df["iso"].min())
    print("Đến:", df["iso"].max())

    return df


# =========================
# 6. TẠO TARGET 4H SAU
# =========================

def create_target(df):
    df = df.copy()
    df = df.sort_values("iso").reset_index(drop=True)

    expected_future_delta = pd.Timedelta(
        minutes=INTERVAL_MINUTES * HORIZON
    )

    df["future_time"] = df["iso"].shift(-HORIZON)
    df["future_close_4h"] = df["close"].shift(-HORIZON)

    df["valid_future_window"] = (
        df["future_time"] - df["iso"]
    ) == expected_future_delta

    df["future_return_4h"] = df["future_close_4h"] / df["close"] - 1

    df["target_up_4h"] = np.nan

    df.loc[
        df["valid_future_window"]
        & (df["future_return_4h"] > THRESHOLD),
        "target_up_4h"
    ] = 1

    df.loc[
        df["valid_future_window"]
        & (df["future_return_4h"] < -THRESHOLD),
        "target_up_4h"
    ] = 0

    print("\n===== TẠO TARGET =====")
    print("Horizon:", HORIZON, "nến")
    print("Dự đoán sau:", HORIZON * INTERVAL_MINUTES, "phút")
    print("Threshold:", THRESHOLD, "=", THRESHOLD * 100, "%")

    print("\nSố lượng target:")
    print(df["target_up_4h"].value_counts(dropna=False))

    return df


# =========================
# 7. CHUẨN BỊ X, y
# =========================

def prepare_training_data(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    drop_cols = [
        "ts",
        "symbol",
        "interval_name",
        "iso",

        "future_time",
        "future_close_4h",
        "future_return_4h",
        "valid_future_window",
        "target_up_4h",
    ]

    feature_cols = [
        col for col in df.columns
        if col not in drop_cols
    ]

    model_data = df.dropna(
        subset=feature_cols + ["target_up_4h"]
    ).copy()

    model_data["target_up_4h"] = model_data["target_up_4h"].astype(int)

    model_data = model_data.set_index("iso")

    X = model_data[feature_cols]
    y = model_data["target_up_4h"]

    print("\n===== DATA TRAIN =====")
    print("Số mẫu train được:", len(X))
    print("Số feature:", len(feature_cols))

    print("\nPhân bố nhãn:")
    print(y.value_counts())
    print("\nTỷ lệ nhãn:")
    print(y.value_counts(normalize=True))

    print("\nDanh sách feature:")
    for col in feature_cols:
        print("-", col)

    if len(X) == 0:
        raise ValueError("Không còn mẫu nào để train sau khi drop NaN.")

    if y.nunique() < 2:
        raise ValueError("Target chỉ có 1 lớp, không thể train model phân loại.")

    return X, y, feature_cols, model_data


# =========================
# 8. RANDOM WALK BASELINE
# =========================

def predict_random_walk(X_test):
    if "window_return" not in X_test.columns:
        raise ValueError("Thiếu feature window_return để chạy Random Walk baseline.")

    return (X_test["window_return"] > 0).astype(int)


# =========================
# 9. BUILD MODEL
# =========================

def build_models(y_train):
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
        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())

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

    return models


# =========================
# 10. HÀM ĐÁNH GIÁ MODEL
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
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "roc_auc": safe_roc_auc(y_true, y_prob),

        "precision_giam": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "recall_giam": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "f1_giam": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),

        "precision_tang": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "recall_tang": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "f1_tang": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
    }


# =========================
# 11. LẤY XÁC SUẤT CLASS 1
# =========================

def get_prob_up(model, X_test, y_pred):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)
        classes = list(model.classes_)

        if 1 in classes:
            return prob[:, classes.index(1)]

    return y_pred


def get_prob_down_up(model, X_one_row):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_one_row)[0]
        classes = list(model.classes_)

        prob_down = prob[classes.index(0)] if 0 in classes else np.nan
        prob_up = prob[classes.index(1)] if 1 in classes else np.nan

        return prob_down, prob_up

    pred = model.predict(X_one_row)[0]

    if pred == 1:
        return 0.0, 1.0

    return 1.0, 0.0


# =========================
# 12. TIME SERIES CROSS VALIDATION
# =========================

def run_time_series_cv(X, y):
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=HORIZON
    )

    cv_results = []

    # -------------------------
    # 12.1. Random Walk baseline
    # -------------------------

    print(f"\n{'=' * 70}")
    print("ĐÁNH GIÁ BASELINE: Random Walk")
    print(f"{'=' * 70}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        y_pred = predict_random_walk(X_test)
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

    # -------------------------
    # 12.2. ML models
    # -------------------------

    model_names = list(build_models(y).keys())

    for name in model_names:
        print(f"\n{'=' * 70}")
        print(f"ĐÁNH GIÁ MODEL: {name}")
        print(f"{'=' * 70}")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            current_model = build_models(y_train)[name]
            current_model.fit(X_train, y_train)

            y_pred = current_model.predict(X_test)
            y_prob = get_prob_up(current_model, X_test, y_pred)

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

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "precision_giam",
        "recall_giam",
        "f1_giam",
        "precision_tang",
        "recall_tang",
        "f1_tang"
    ]

    summary_df = (
        cv_results_df
        .groupby("model")[metric_cols]
        .mean()
        .reset_index()
        .sort_values(
            by=["macro_f1", "balanced_accuracy", "roc_auc"],
            ascending=False
        )
    )

    detail_path = OUTPUT_DIR / "timeseries_cv_detail.csv"
    summary_path = OUTPUT_DIR / "timeseries_cv_summary.csv"

    cv_results_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n===== BẢNG KẾT QUẢ TRUNG BÌNH THEO MODEL =====")
    print(summary_df.to_string(index=False))

    print("\nĐã lưu kết quả chi tiết từng fold vào:", detail_path)
    print("Đã lưu bảng tổng hợp trung bình vào:", summary_path)

    return cv_results_df, summary_df


# =========================
# 13. TRAIN FINAL MODEL
# =========================

def train_final_model(X, y, feature_cols, summary_df):
    best_overall_name = summary_df.iloc[0]["model"]

    print("\n===== MODEL TỐT NHẤT TỔNG THỂ =====")
    print("Best overall:", best_overall_name)
    print("Lưu ý: Random Walk là baseline, không phải model ML.")

    ml_summary_df = summary_df[
        summary_df["model"] != "Random Walk"
    ].copy()

    if ml_summary_df.empty:
        raise ValueError("Không có model ML nào để chọn.")

    best_ml_name = ml_summary_df.iloc[0]["model"]

    print("\n===== MODEL MACHINE LEARNING TỐT NHẤT =====")
    print("Best ML model:", best_ml_name)
    print("Tiêu chí chọn: macro_f1 → balanced_accuracy → roc_auc")

    best_model = build_models(y)[best_ml_name]
    final_model = clone(best_model)
    final_model.fit(X, y)

    model_path = MODEL_DIR / "best_btc_4h_model.pkl"
    feature_path = MODEL_DIR / "feature_cols.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"

    joblib.dump(final_model, model_path)
    joblib.dump(feature_cols, feature_path)

    metadata = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "source": "mysql",
        "mysql_database": MYSQL_DATABASE,
        "mysql_table": MYSQL_TABLE,
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "interval_minutes": INTERVAL_MINUTES,
        "threshold": THRESHOLD,
        "label_map": {0: "GIẢM", 1: "TĂNG"},
        "method": "features_from_mysql_predict_next_16_candles_timeseries_split",
        "best_overall_model": best_overall_name,
        "best_ml_model": best_ml_name,
        "cv_n_splits": N_SPLITS,
        "cv_gap": HORIZON,
        "selection_metric_order": [
            "macro_f1",
            "balanced_accuracy",
            "roc_auc"
        ]
    }

    joblib.dump(metadata, metadata_path)

    print("\nĐã lưu model tốt nhất vào:", model_path)
    print("Đã lưu danh sách feature vào:", feature_path)
    print("Đã lưu metadata vào:", metadata_path)

    return final_model, best_ml_name


# =========================
# 14. CLASSIFICATION REPORT CHO FOLD CUỐI
# =========================

def print_last_fold_report(X, y, best_ml_name):
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=HORIZON
    )

    folds = list(tscv.split(X))
    train_idx, test_idx = folds[-1]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = build_models(y_train)[best_ml_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== CLASSIFICATION REPORT FOLD CUỐI CỦA BEST ML MODEL =====")
    print("Model:", best_ml_name)

    print(classification_report(
        y_test,
        y_pred,
        target_names=["GIẢM", "TĂNG"],
        zero_division=0
    ))

    print("\n===== CONFUSION MATRIX FOLD CUỐI =====")
    print(confusion_matrix(y_test, y_pred))


# =========================
# 15. DỰ ĐOÁN 4 GIỜ TỚI
# =========================

def predict_latest(df, X, y, feature_cols):
    trained_full_models = {}

    for name, model in build_models(y).items():
        model_full = clone(model)
        model_full.fit(X, y)
        trained_full_models[name] = model_full

    latest_data = df.replace([np.inf, -np.inf], np.nan)
    latest_data = latest_data.dropna(subset=feature_cols).copy()

    if latest_data.empty:
        raise ValueError("Không tìm thấy dòng mới nhất đủ feature để dự đoán.")

    latest_data = latest_data.set_index("iso")

    latest_X = latest_data[feature_cols].iloc[[-1]]

    latest_time = latest_X.index[-1]

    window_start = latest_time - pd.Timedelta(
        minutes=INTERVAL_MINUTES * (WINDOW_SIZE - 1)
    )

    predict_start = latest_time
    predict_end = latest_time + pd.Timedelta(
        minutes=INTERVAL_MINUTES * HORIZON
    )

    print("\n===== DỰ ĐOÁN BẰNG 16 NẾN 15 PHÚT GẦN NHẤT =====")
    print("Window dùng để dự đoán:", window_start, "→", latest_time)
    print("Dự đoán xu hướng 4 giờ tới:", predict_start, "→", predict_end)

    predictions = []

    # Random Walk baseline
    rw_pred = int(latest_X["window_return"].iloc[0] > 0)
    rw_trend = "TĂNG" if rw_pred == 1 else "GIẢM"

    predictions.append({
        "model": "Random Walk",
        "prediction": rw_trend,
        "prob_down": 1.0 - rw_pred,
        "prob_up": float(rw_pred),
        "note": "Baseline hard prediction"
    })

    print("\nRandom Walk:")
    print("Dự đoán:", rw_trend)
    print(f"Xác suất GIẢM: {1.0 - rw_pred:.3f}")
    print(f"Xác suất TĂNG: {float(rw_pred):.3f}")

    # ML models
    for name, model_full in trained_full_models.items():
        pred = model_full.predict(latest_X)[0]
        prob_down, prob_up = get_prob_down_up(model_full, latest_X)

        trend = "TĂNG" if pred == 1 else "GIẢM"

        predictions.append({
            "model": name,
            "prediction": trend,
            "prob_down": prob_down,
            "prob_up": prob_up,
            "note": "ML model"
        })

        print(f"\n{name}:")
        print("Dự đoán:", trend)
        print(f"Xác suất GIẢM: {prob_down:.3f}")
        print(f"Xác suất TĂNG: {prob_up:.3f}")

    prediction_df = pd.DataFrame(predictions)

    prediction_path = OUTPUT_DIR / "latest_prediction.csv"
    prediction_df.to_csv(prediction_path, index=False)

    print("\nĐã lưu dự đoán mới nhất vào:", prediction_path)

    return prediction_df


# =========================
# 16. MAIN
# =========================

def main():
    df = load_features_from_mysql()

    df = validate_and_clean_sql_data(df)

    df = create_target(df)

    X, y, feature_cols, model_data = prepare_training_data(df)

    cv_results_df, summary_df = run_time_series_cv(X, y)

    final_model, best_ml_name = train_final_model(
        X=X,
        y=y,
        feature_cols=feature_cols,
        summary_df=summary_df
    )

    print_last_fold_report(
        X=X,
        y=y,
        best_ml_name=best_ml_name
    )

    predict_latest(
        df=df,
        X=X,
        y=y,
        feature_cols=feature_cols
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
