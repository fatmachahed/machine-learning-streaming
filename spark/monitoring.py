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

csv_path = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
df_sample = pd.read_csv(csv_path, nrows=1)

schema = StructType()
for col_name in df_sample.columns:
    schema = schema.add(col_name, DoubleType())  

feature_cols = [c for c in df_sample.columns if c not in ["Time", "Class"]]

CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints", "monitoring")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "monitoring")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = SparkSession.builder \
    .appName("Phase3-DriftMonitoring") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS) \
    .option("subscribe", "credit_card_transactions") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

batch_stats_history = []

def process_batch(batch_df, batch_id):
    import joblib

    if batch_df.isEmpty():
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    pdf = batch_df.toPandas()
    pdf["Class"] = pdf["Class"].astype("Int64")

    X = pdf[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    pdf["fraud_probability"] = model.predict_proba(X_scaled)[:, 1]
    pdf["prediction"] = (pdf["fraud_probability"] > 0.5).astype(int)

    avg_proba = pdf["fraud_probability"].mean()
    fraud_rate = pdf["prediction"].mean()
    avg_amount = pdf["Amount"].mean()

    batch_stats_history.append({
        "batch_id": batch_id,
        "avg_fraud_probability": avg_proba,
        "fraud_rate": fraud_rate,
        "avg_amount": avg_amount,
        "n_events": len(pdf)
    })

    print(f"\n=== Batch {batch_id} — {len(pdf)} events ===")
    print(pdf[["Time", "Amount", "fraud_probability", "prediction", "Class"]].to_string())
    print(f"\n📊 Drift Stats:")
    print(f"   avg fraud probability : {avg_proba:.4f}")
    print(f"   fraud rate            : {fraud_rate:.4f}")
    print(f"   avg amount            : {avg_amount:.2f}")

    # Drift alert after 5 batches baseline
    if len(batch_stats_history) > 5:
        baseline = pd.DataFrame(batch_stats_history[:5])["avg_fraud_probability"].mean()
        if avg_proba > baseline * 2:
            print(f"⚠️  DRIFT ALERT! {avg_proba:.4f} >> baseline {baseline:.4f}")

    # Save stats for offline analysis
    pd.DataFrame(batch_stats_history).to_csv(
        os.path.join(OUTPUT_DIR, "drift_stats.csv"), index=False
    )

query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", CKPT_DIR) \
    .start()

query.awaitTermination()
