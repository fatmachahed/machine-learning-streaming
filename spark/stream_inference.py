import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, DoubleType

if os.name == 'nt':
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

BOOTSTRAP_SERVERS = "localhost:9092"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

def train_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    os.makedirs(os.path.join(BASE_DIR, "..", "model"), exist_ok=True)
    csv_path = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")

    print("Training model on creditcard.csv...")
    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(int)

    feature_names = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_names]
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    train_model()

# ─────────────────────────────────────────
# Schema - tout en DoubleType ✅
# Class sera 0.0 ou 1.0 (JSON encode les int en float)
# ─────────────────────────────────────────
csv_path = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
df_sample = pd.read_csv(csv_path, nrows=1)

schema = StructType()
for col_name in df_sample.columns:
    schema = schema.add(col_name, DoubleType())  # ✅ tout en Double, plus de NaN

feature_cols = [c for c in df_sample.columns if c not in ["Time", "Class"]]
print(f"Feature cols ({len(feature_cols)}): {feature_cols[:5]}...")

# ─────────────────────────────────────────
# Spark session
# ─────────────────────────────────────────
CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints", "stream_inference")
os.makedirs(CKPT_DIR, exist_ok=True)

spark = SparkSession.builder \
    .appName("Phase1-StreamInference") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ─────────────────────────────────────────
# Read stream
# ─────────────────────────────────────────
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS) \
    .option("subscribe", "credit_card_transactions") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# ─────────────────────────────────────────
# Process batch
# ─────────────────────────────────────────
def process_batch(batch_df, batch_id):
    import joblib
    import pandas as pd

    if batch_df.isEmpty():
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    pdf = batch_df.toPandas()
    pdf["Class"] = pdf["Class"].astype("Int64")  # ✅ 0.0 → 0

    X = pdf[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    pdf["fraud_probability"] = model.predict_proba(X_scaled)[:, 1]
    pdf["prediction"] = (pdf["fraud_probability"] > 0.5).astype(int)

    print(f"\n=== Batch {batch_id} — {len(pdf)} events ===")
    print(pdf[["Time", "Amount", "fraud_probability", "prediction", "Class"]].to_string())

    fraud_count = int(pdf["prediction"].sum())
    if fraud_count > 0:
        print(f"🚨 {fraud_count} FRAUD ALERT(S) in batch {batch_id}!")
    else:
        print(f"✅ No fraud detected in batch {batch_id}")

# ─────────────────────────────────────────
# Start stream
# ─────────────────────────────────────────
query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", CKPT_DIR) \
    .start()

query.awaitTermination()