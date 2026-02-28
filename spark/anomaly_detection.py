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

csv_path = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
df_sample = pd.read_csv(csv_path, nrows=1)

schema = StructType()
for col_name in df_sample.columns:
    schema = schema.add(col_name, DoubleType())  

feature_cols = [c for c in df_sample.columns if c not in ["Time", "Class"]]

# Compute stats on normal transactions
df_train = pd.read_csv(csv_path)
df_train["Class"] = df_train["Class"].astype(int)
normal = df_train[df_train["Class"] == 0]

stats = {}
for f in feature_cols:
    stats[f] = {
        "mean": float(normal[f].mean()),
        "std": float(normal[f].std())
    }
print(f"✅ Stats computed on {len(normal)} normal transactions")

CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints", "anomaly_detection")
os.makedirs(CKPT_DIR, exist_ok=True)

spark = SparkSession.builder \
    .appName("Phase2-AnomalyDetection") \
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

def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    pdf = batch_df.toPandas()
    pdf["Class"] = pdf["Class"].astype("Int64")

    # Compute anomaly score (mean z-score across all features)
    scores = []
    for _, row in pdf.iterrows():
        score = 0.0
        for f in feature_cols:
            mean = stats[f]["mean"]
            std = stats[f]["std"]
            if std > 0:
                score += abs((row[f] - mean) / std)
        scores.append(score / len(feature_cols))

    pdf["anomaly_score"] = scores
    pdf["is_anomaly"] = (pdf["anomaly_score"] > 3.0).astype(int)

    anomalies = pdf[pdf["is_anomaly"] == 1]

    print(f"\n=== Batch {batch_id} — {len(pdf)} events, {len(anomalies)} anomalies ===")
    if not anomalies.empty:
        print("🚨 ANOMALIES DETECTED:")
        print(anomalies[["Time", "Amount", "anomaly_score", "Class"]].to_string())
    else:
        print("✅ No anomalies in this batch")

query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", CKPT_DIR) \
    .start()

query.awaitTermination()
