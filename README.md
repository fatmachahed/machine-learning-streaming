# Lab — Machine Learning on Streams

You will build a **real-time Machine Learning pipeline** using **Kafka** and **Spark Structured Streaming**, and apply ML concepts on **continuous data streams**.

---


## Concepts Covered

This lab integrates all major concepts seen in the course:

### Streaming & Systems
- Kafka topics, partitions, offsets
- Kafka producers and consumers
- Kafka → Spark integration
- Continuous data ingestion

### Stream Processing
- Spark Structured Streaming
- Streaming DataFrames
- Stateless vs stateful computations (conceptual)
- Micro-batch execution model

### Machine Learning on Streams
- Streaming inference
- Anomaly detection
- Drift monitoring
- Model behavior over time

---

## Data
The data used in this lab is **Credit Card Fraud Detection** on Kaggle : 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=

---

## Lab Structure

The lab is divided into **three progressive phases**, each increasing in difficulty.

You are strongly encouraged to complete the phases **in order**.

---

## 🔹 Phase 1 — Streaming Inference

### Objective
Apply a pre-trained ML model to **each incoming event** in real time.

### What you will do
- Stream events from Kafka into Spark
- Load a pre-trained ML model
- Apply the model to streaming events
- Output predictions continuously

### Key ideas
- One event → one prediction
- Low-latency inference
- No retraining in this phase

---

## 🔹 Phase 2 — Anomaly Detection on Streams

### Objective
Detect **unusual events** in streaming data.

### What you will do
- Define what “normal” behavior looks like
- Compute simple statistics on the stream
- Flag events that deviate significantly
- Output anomalies in real time

### Key ideas
- Anomalies are detected at the **event level**
- Not all ML requires labeled data
- Streaming systems must react immediately

---

## 🔹 Phase 3 — Drift Monitoring

### Objective
Monitor how data and model behavior change **over time**.

### What you will do
- Track prediction statistics over the stream
- Observe long-term changes
- Detect potential concept drift
- Discuss mitigation strategies

### Key ideas
- Drift happens gradually
- Models degrade silently
- Monitoring is essential in production ML

---

## End-to-End Pipeline

Events
↓
Kafka
↓
Spark Structured Streaming
↓
ML Model
↓
Predictions
↓
Alerts / Decisions


Kafka transports events.  
Spark processes streams.  
ML makes decisions.

---

## Requirements

- Docker
- Python 3.9+
- Kafka running via Docker
- Spark available (local or Docker)

---
