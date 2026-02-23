import os
import json
import time
import pandas as pd
from confluent_kafka import Producer

conf = {"bootstrap.servers": "localhost:9092"}
producer = Producer(conf)

def delivery_report(err, msg):
    if err:
        print("Delivery failed:", err)
    else:
        print(f"✅ Delivered to {msg.topic()} partition {msg.partition()}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")

print(f"Reading from: {os.path.abspath(CSV_FILE)}")
df = pd.read_csv(CSV_FILE)
df["Class"] = df["Class"].astype(int)

print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {df['Class'].sum()}")

for _, row in df.iterrows():
    event = row.to_dict()
    event["Class"] = int(event["Class"])  # ✅ force int natif Python (pas float)
    producer.produce(
        "credit_card_transactions",
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_report
    )
    producer.poll(0)
    time.sleep(0.05)

producer.flush()
print("✅ All transactions sent.")