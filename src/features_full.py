import time
import requests
import pandas as pd
import numpy as np
import mysql.connector

from datetime import datetime, timedelta, timezone


# =====================================
# CONFIG
# =====================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

# Lấy 2 năm để đủ dữ liệu tính MA99, RSI, Bollinger, volatility_32...
FETCH_DAYS = 730

# File cuối chỉ lấy 1 năm gần nhất
FINAL_DAYS = 365

# 1 ngày có 24h, mỗi giờ có 4 nến 15m
EXPECTED_FINAL_ROWS = FINAL_DAYS * 24 * 4

BINANCE_API = "https://api.binance.com/api/v3/klines"
REQ_TIMEOUT = 15

OUTPUT_FILE = "BTCUSDT_XU_HUONG_1NAM_EXACT_FEATURES.csv"


# =====================================
# MYSQL CONFIG
# =====================================

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DATABASE = "crypto_db"

# Bảng lưu dữ liệu
MYSQL_TABLE = "btc_features"


# =====================================
# DANH SÁCH CỘT XUẤT CSV
# Chỉ đúng các feature yêu cầu
# Không có ts trong CSV
# =====================================

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


# =====================================
# TIME HELPERS
# =====================================

def get_aligned_time_range():
    now = datetime.now(timezone.utc)

    # Lấy mốc 15 phút gần nhất
    minute = (now.minute // 15) * 15

    end_dt = now.replace(
        minute=minute,
        second=0,
        microsecond=0
    )

    # Không lấy cây nến đang chạy
    end_dt = end_dt - timedelta(minutes=15)

    # Lùi lại 2 năm để đủ dữ liệu tính rolling/indicator
    start_dt = end_dt - timedelta(days=FETCH_DAYS)

    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(end_dt.timestamp() * 1000)

    return start_time, end_time, end_dt


# =====================================
# MYSQL FUNCTIONS
# =====================================

def connect_mysql():
    db = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )

    cursor = db.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
    cursor.execute(f"USE `{MYSQL_DATABASE}`")

    return db, cursor


def mysql_column_type(col):
    if col == "ts":
        return "BIGINT NOT NULL"

    if col == "iso":
        return "DATETIME NULL"

    if col in ["trades", "up_candle"]:
        return "INT NULL"

    if col in ["symbol"]:
        return "VARCHAR(20)"

    if col in ["interval_name"]:
        return "VARCHAR(10)"

    return "DOUBLE NULL"


def get_existing_columns_lower(cursor, table_name):
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    rows = cursor.fetchall()

    # MySQL không phân biệt hoa/thường ở tên cột,
    # nên cần convert lower để tránh lỗi duplicate column.
    return {row[0].lower(): row[0] for row in rows}


def add_column_if_not_exists(cursor, table_name, col_name, col_type):
    existing_columns = get_existing_columns_lower(cursor, table_name)

    if col_name.lower() in existing_columns:
        return

    try:
        alter_sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_type}"
        cursor.execute(alter_sql)

    except mysql.connector.Error as err:
        # 1060 = Duplicate column name
        # Bỏ qua vì cột thực tế đã tồn tại.
        if err.errno == 1060:
            return

        raise


def get_existing_indexes(cursor, table_name):
    cursor.execute(f"SHOW INDEX FROM `{table_name}`")
    rows = cursor.fetchall()

    indexes = set()

    for row in rows:
        # row[2] là Key_name
        indexes.add(row[2])

    return indexes


def create_table(cursor):
    # Tạo bảng cơ bản trước
    # Không nhét hết cột vào CREATE TABLE để tránh lỗi bảng cũ lệch schema.
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{MYSQL_TABLE}` (
        id INT AUTO_INCREMENT PRIMARY KEY
    )
    """

    cursor.execute(sql)

    # Danh sách cột bắt buộc trong MySQL
    # MySQL cần ts để làm khóa chống trùng, nhưng CSV không xuất ts.
    required_columns = ["symbol", "interval_name", "ts"] + FINAL_COLUMNS

    # Loại trùng phòng trường hợp có cột lặp
    seen = set()
    unique_required_columns = []

    for col in required_columns:
        key = col.lower()
        if key not in seen:
            seen.add(key)
            unique_required_columns.append(col)

    # Thêm các cột còn thiếu
    for col in unique_required_columns:
        col_type = mysql_column_type(col)
        add_column_if_not_exists(
            cursor=cursor,
            table_name=MYSQL_TABLE,
            col_name=col,
            col_type=col_type
        )

    # Đảm bảo ts có UNIQUE index để ON DUPLICATE KEY UPDATE hoạt động
    try:
        indexes = get_existing_indexes(cursor, MYSQL_TABLE)

        if "unique_ts" not in indexes:
            cursor.execute(
                f"ALTER TABLE `{MYSQL_TABLE}` ADD UNIQUE KEY `unique_ts` (`ts`)"
            )

    except mysql.connector.Error as err:
        # 1061 = Duplicate key name
        # 1062 = Duplicate entry, xảy ra nếu bảng cũ đã có nhiều dòng trùng ts
        # Với 1062 thì không dừng chương trình, nhưng ON DUPLICATE có thể không hoạt động đúng.
        if err.errno in [1061, 1062]:
            print("Warning: Không thêm được UNIQUE index cho ts:", err)
        else:
            raise


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert(None)

        return value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.replace(tzinfo=None)

        return value

    return value


def insert_to_mysql(db, cursor, df):
    mysql_columns = ["symbol", "interval_name", "ts"] + FINAL_COLUMNS

    insert_cols_sql = ", ".join([f"`{col}`" for col in mysql_columns])
    placeholders = ", ".join(["%s"] * len(mysql_columns))

    update_columns = []

    for col in mysql_columns:
        if col == "ts":
            continue

        update_columns.append(f"`{col}` = VALUES(`{col}`)")

    update_sql = ",\n        ".join(update_columns)

    sql = f"""
    INSERT INTO `{MYSQL_TABLE}` (
        {insert_cols_sql}
    )
    VALUES (
        {placeholders}
    )
    ON DUPLICATE KEY UPDATE
        {update_sql}
    """

    data = []

    for _, row in df.iterrows():
        values = [
            SYMBOL,
            INTERVAL,
            clean_value(row["ts"])
        ]

        for col in FINAL_COLUMNS:
            values.append(clean_value(row[col]))

        data.append(tuple(values))

    if len(data) == 0:
        print("Không có dữ liệu để insert vào MySQL.")
        return

    cursor.executemany(sql, data)
    db.commit()

    print(f"Inserted/Updated {len(data)} rows into MySQL table `{MYSQL_TABLE}`")


# =====================================
# FETCH DATA
# =====================================

def fetch_klines(symbol, interval, start_time, end_time):
    all_bars = []

    while start_time <= end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            response = requests.get(
                BINANCE_API,
                params=params,
                timeout=REQ_TIMEOUT
            )

        except requests.exceptions.RequestException as err:
            print("Request Error:", err)
            break

        if response.status_code != 200:
            print("API Error:", response.status_code, response.text)
            break

        data = response.json()

        if not data:
            break

        all_bars.extend(data)

        # Binance trả open time theo millisecond.
        # Cộng 1 để không lấy trùng cây nến cuối batch trước.
        start_time = data[-1][0] + 1

        time.sleep(0.2)

    return all_bars


# =====================================
# DATAFRAME
# =====================================

def create_dataframe(bars):
    rows = []

    for b in bars:
        ts = b[0] // 1000

        iso = datetime.fromtimestamp(
            ts,
            tz=timezone.utc
        )

        rows.append({
            "ts": ts,
            "iso": iso,
            "open": float(b[1]),
            "high": float(b[2]),
            "low": float(b[3]),
            "close": float(b[4]),
            "volume": float(b[5]),
            "trades": int(b[8])
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("iso").reset_index(drop=True)

    # Xóa trùng theo ts nếu API trả trùng
    df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    return df


# =====================================
# FEATURE ENGINEERING
# =====================================

def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_features_inference(df):
    df = df.copy()

    df["iso"] = pd.to_datetime(df["iso"], utc=True)

    # -----------------------------
    # 1. Feature của từng nến 15 phút
    # -----------------------------

    df["candle_return"] = safe_divide(
        df["close"] - df["open"],
        df["open"]
    )

    df["candle_range_pct"] = safe_divide(
        df["high"] - df["low"],
        df["open"]
    )

    df["candle_body_pct"] = safe_divide(
        (df["close"] - df["open"]).abs(),
        df["open"]
    )

    df["up_candle"] = (df["close"] > df["open"]).astype(int)

    df["close_return_1"] = df["close"].pct_change(1)

    # -----------------------------
    # 2. Feature rolling window 16 nến
    # Window gồm nến hiện tại + 15 nến trước đó
    # -----------------------------

    window = 16

    df["window_open"] = df["open"].shift(window - 1)

    df["window_high"] = df["high"].rolling(window).max()

    df["window_low"] = df["low"].rolling(window).min()

    df["window_close"] = df["close"]

    df["window_return"] = safe_divide(
        df["window_close"] - df["window_open"],
        df["window_open"]
    )

    df["window_range"] = safe_divide(
        df["window_high"] - df["window_low"],
        df["window_open"]
    )

    df["window_body_size"] = safe_divide(
        (df["window_close"] - df["window_open"]).abs(),
        df["window_open"]
    )

    df["window_volume_sum"] = df["volume"].rolling(window).sum()

    df["window_volume_mean"] = df["volume"].rolling(window).mean()

    df["window_volume_std"] = df["volume"].rolling(window).std()

    df["volume_change"] = df["volume"].pct_change(1)

    df["volume_ratio_16"] = safe_divide(
        df["volume"],
        df["window_volume_mean"]
    )

    # -----------------------------
    # 3. Thống kê return trong 16 nến
    # -----------------------------

    df["mean_15m_return"] = df["close_return_1"].rolling(window).mean()

    df["std_15m_return"] = df["close_return_1"].rolling(window).std()

    df["max_15m_return"] = df["close_return_1"].rolling(window).max()

    df["min_15m_return"] = df["close_return_1"].rolling(window).min()

    df["up_candle_ratio"] = df["up_candle"].rolling(window).mean()

    df["close_position"] = safe_divide(
        df["close"] - df["window_low"],
        df["window_high"] - df["window_low"]
    )

    df["volatility_16"] = df["close_return_1"].rolling(16).std()

    df["volatility_32"] = df["close_return_1"].rolling(32).std()

    # -----------------------------
    # 4. Moving Average
    # -----------------------------

    df["MA7"] = df["close"].rolling(7).mean()

    df["MA25"] = df["close"].rolling(25).mean()

    df["MA99"] = df["close"].rolling(99).mean()

    # -----------------------------
    # 5. EMA
    # -----------------------------

    df["EMA12"] = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA26"] = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    # -----------------------------
    # 6. MACD
    # -----------------------------

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACD_signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # -----------------------------
    # 7. RSI 14 nến
    # -----------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # Nếu avg_loss = 0 và avg_gain > 0 thì RSI = 100
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    # Nếu avg_loss = 0 và avg_gain = 0 thì RSI = 50
    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # -----------------------------
    # 8. Bollinger Band 20 nến
    # -----------------------------

    ma20 = df["close"].rolling(20).mean()

    std20 = df["close"].rolling(20).std()

    df["BB_upper"] = ma20 + 2 * std20

    df["BB_lower"] = ma20 - 2 * std20

    df["BB_width"] = safe_divide(
        df["BB_upper"] - df["BB_lower"],
        ma20
    )

    # -----------------------------
    # 9. Khoảng cách giá hiện tại so với MA
    # -----------------------------

    df["distance_ma7"] = safe_divide(
        df["close"] - df["MA7"],
        df["MA7"]
    )

    df["distance_ma25"] = safe_divide(
        df["close"] - df["MA25"],
        df["MA25"]
    )

    df["distance_ma99"] = safe_divide(
        df["close"] - df["MA99"],
        df["MA99"]
    )

    return df


# =====================================
# KIỂM TRA ĐÚNG FEATURE
# =====================================

def validate_exact_features(df):
    missing_columns = [
        col for col in FINAL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Thiếu feature: {missing_columns}")

    duplicated_columns = [
        col for col in FINAL_COLUMNS
        if FINAL_COLUMNS.count(col) > 1
    ]

    if duplicated_columns:
        raise ValueError(f"Có feature bị lặp: {duplicated_columns}")

    print("Feature check OK")
    print("Total output columns:", len(FINAL_COLUMNS))

    print("Output columns:")
    for col in FINAL_COLUMNS:
        print("-", col)


# =====================================
# LỌC ĐÚNG 1 NĂM GẦN NHẤT VÀ XÓA NaN
# =====================================

def keep_last_1_year_no_nan(df, end_dt):
    expected_rows = EXPECTED_FINAL_ROWS

    start_final_dt = end_dt - timedelta(days=FINAL_DAYS)

    start_final_ts = pd.Timestamp(start_final_dt)

    end_final_ts = pd.Timestamp(end_dt)

    if start_final_ts.tzinfo is None:
        start_final_ts = start_final_ts.tz_localize("UTC")
    else:
        start_final_ts = start_final_ts.tz_convert("UTC")

    if end_final_ts.tzinfo is None:
        end_final_ts = end_final_ts.tz_localize("UTC")
    else:
        end_final_ts = end_final_ts.tz_convert("UTC")

    df = df[
        (df["iso"] >= start_final_ts) &
        (df["iso"] <= end_final_ts)
    ].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    before_drop = len(df)

    # Xóa NaN chỉ dựa trên các cột cần xuất + ts để lưu MySQL
    df = df.dropna(
        subset=["ts"] + FINAL_COLUMNS
    ).reset_index(drop=True)

    after_drop = len(df)

    # Giữ đúng 35040 dòng gần nhất
    if len(df) > expected_rows:
        df = df.tail(expected_rows).reset_index(drop=True)

    print("Expected final rows:", expected_rows)
    print("Rows before drop NaN:", before_drop)
    print("Rows after drop NaN :", after_drop)
    print("Final rows after cut:", len(df))
    print("Dropped rows        :", before_drop - after_drop)

    return df


# =====================================
# MAIN
# =====================================

def main():
    start_time, end_time, end_dt = get_aligned_time_range()

    bars = fetch_klines(
        SYMBOL,
        INTERVAL,
        start_time,
        end_time
    )

    print("Fetched 2 years bars:", len(bars))

    if len(bars) == 0:
        print("Không fetch được dữ liệu từ Binance.")
        return

    df = create_dataframe(bars)

    print("Raw DataFrame rows:", len(df))

    if df.empty:
        print("DataFrame rỗng, dừng chương trình.")
        return

    df = add_features_inference(df)

    print("Rows after feature engineering:", len(df))

    validate_exact_features(df)

    df = keep_last_1_year_no_nan(df, end_dt)

    if df.empty:
        print("Sau khi lọc và xóa NaN, DataFrame rỗng.")
        return

    # DataFrame để lưu CSV: chỉ đúng các feature yêu cầu
    df_csv = df[FINAL_COLUMNS].copy()

    print("Final CSV rows:", len(df_csv))
    print("CSV total lines will be:", len(df_csv) + 1)

    print("NaN count after final filter:")
    print(df_csv.isna().sum())

    df_csv.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Saved file:", OUTPUT_FILE)

    # DataFrame để lưu MySQL: thêm ts để tránh lỗi Field 'ts' doesn't have a default value
    df_mysql = df[["ts"] + FINAL_COLUMNS].copy()

    db = None
    cursor = None

    try:
        db, cursor = connect_mysql()

        create_table(cursor)

        insert_to_mysql(
            db,
            cursor,
            df_mysql
        )

    except mysql.connector.Error as err:
        print("MySQL Error:", err)

    finally:
        if cursor is not None:
            cursor.close()

        if db is not None:
            db.close()

    print("Done!")


if __name__ == "__main__":
    main()