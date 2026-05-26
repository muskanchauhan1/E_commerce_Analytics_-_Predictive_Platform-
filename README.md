<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=00C8FF&center=true&vCenter=true&width=700&lines=E-Commerce+Analytics+%26+Predictive+Platform;Olist+Brazilian+Dataset+%7C+1%2C19%2C143+Orders;ML+%7C+NLP+%7C+Big+Data+%7C+AWS+%7C+Flask" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

<br/>

> **An end-to-end data science platform on 1,19,143 Olist e-commerce orders —**  
> **combining Big Data, Machine Learning, NLP, and Cloud to generate real business intelligence.**

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/muskanchauhan1/E_commerce_Analytics_-_Predictive_Platform?style=social)](https://github.com/muskanchauhan1/E_commerce_Analytics_-_Predictive_Platform)
[![GitHub Forks](https://img.shields.io/github/forks/muskanchauhan1/E_commerce_Analytics_-_Predictive_Platform?style=social)](https://github.com/muskanchauhan1/E_commerce_Analytics_-_Predictive_Platform)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Institution](https://img.shields.io/badge/CDAC-Mumbai-blue)

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏆 Key Business Insights](#-key-business-insights)
- [🗂️ Dataset](#️-dataset)
- [🔄 Project Pipeline](#-project-pipeline)
- [📊 EDA Highlights](#-eda-highlights)
- [⚡ Spark SQL — Delay Analysis](#-spark-sql--delay-analysis)
- [📈 Time Series Forecasting](#-time-series-forecasting)
- [🎯 Clustering & Segmentation](#-clustering--segmentation)
- [🧠 Sentiment Analysis (NLP)](#-sentiment-analysis-nlp)
- [🌐 Flask Web Application](#-flask-web-application)
- [☁️ AWS Architecture](#️-aws-architecture)
- [📊 Power BI Dashboard](#-power-bi-dashboard)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Installation & Setup](#️-installation--setup)
- [👥 Team](#-team)
- [📁 Project Structure](#-project-structure)

---

## 🎯 Project Overview

This project implements a **complete, production-grade data science pipeline** on the **Olist Brazilian E-Commerce dataset** — Brazil's largest online retail platform. It solves four real business problems:

| Business Problem | Solution | Result |
|---|---|---|
| Why are deliveries delayed? | Spark SQL analysis on 1.19L orders | AL state = 24.49% delay rate identified |
| Who are valuable vs churning customers? | K-Means clustering on RFM features | 22 VIP customers spending avg R$26,932 each |
| What will sales look like next quarter? | Facebook Prophet time series | 12-month forecast with 31.7% MAPE |
| Are customers satisfied with products? | DistilBERT transformer | 92.06% confidence on 40,809 reviews |

---

## 🏆 Key Business Insights

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║  🔥 MOST POWERFUL FINDING FROM SPARK SQL                        ║
║                                                                  ║
║  1-star reviews → 31.69% delivery delay rate                    ║
║  5-star reviews →  2.97% delivery delay rate                    ║
║                                                                  ║
║  → Delivery speed is the #1 driver of customer satisfaction     ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

| Metric | Value |
|---|---|
| 📦 Total Orders | 1,19,143 |
| 💰 Total Revenue | R$20,579,664 |
| 🚚 Delivery Rate | **97.13%** |
| ⚠️ Delay Rate | **7.32%** (8,469 orders) |
| ⭐ Avg Review Score | 4.02 / 5.0 |
| 📈 Prophet MAPE | **31.7%** |
| 🤖 DistilBERT Confidence | **92.06%** avg |
| 👑 VIP Customers Found | **22** (avg R$26,932 spend) |

---

## 🗂️ Dataset

**Source:** [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Coverage:** December 2016 → August 2018 | 9 CSV files | 1,19,143 real orders

| File | Contents | Rows |
|---|---|---|
| `olist_orders_dataset.csv` | Order status, timestamps, delivery dates | 99,441 |
| `olist_customers_dataset.csv` | Customer ID, city, state | 99,441 |
| `olist_order_items_dataset.csv` | Products, sellers, prices, freight | 1,12,650 |
| `olist_products_dataset.csv` | Category, weight, dimensions | 32,951 |
| `olist_sellers_dataset.csv` | Seller location and state | 3,095 |
| `olist_order_reviews_dataset.csv` | Star ratings (1–5), review text | 99,224 |
| `olist_order_payments_dataset.csv` | Payment type, value | 1,03,886 |
| `olist_geolocation_dataset.csv` | Zip, latitude, longitude | 10,00,163 |
| `product_category_name_translation.csv` | Portuguese → English categories | 71 |

---

## 🔄 Project Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    9 RAW CSV FILES (Kaggle)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Pandas .merge() on order_id
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           DATA CLEANING & FEATURE ENGINEERING               │
│  • Fixed nulls in 5 columns                                 │
│  • Created: delay_days, is_delayed, order_month             │
│  • Output: master_df.csv (1,19,143 rows × 39 cols)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           EDA — 7 CHARTS (Matplotlib + Seaborn)             │
│  • Order status, monthly trend, top categories              │
│  • Delay distribution, revenue by state, review scores      │
└──────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼                ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ SPARK SQL│    │  PROPHET │    │ K-MEANS  │    │ DistilBERT│
       │10 queries│    │  MAPE    │    │ 3 types  │    │ NLP      │
       │ Google   │    │  31.7%   │    │ of seg.  │    │ 92.06%   │
       │  Colab   │    │          │    │          │    │ conf.    │
       └──────────┘    └──────────┘    └──────────┘    └──────────┘
              │                ▼                 ▼
              ▼         ┌──────────────────────────────┐
       ┌──────────┐     │   SAVED ARTIFACTS (AWS S3)   │
       │Delay CSVs│     │ • 3 PKL clustering models    │
       │for Power │     │ • Prophet forecast results   │
       │    BI    │     │ • DistilBERT on S3 bucket    │
       └──────────┘     └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │    FLASK WEB APP         │
                        │  6 routes · Jinja2       │
                        │  7 HTML templates        │
                        │  localhost:5000          │
                        └──────────────────────────┘
```

---

## 📊 EDA Highlights

<table>
<tr>
<td width="50%">

**7 Charts Built**
- 📊 Order status distribution → **97.13% delivered**
- 📈 Monthly trend → **Black Friday +80% November spike**
- 🛍️ Top 15 categories → **Bed & bath most ordered**
- 🚚 Delay distribution → **7.32% delayed, most arrive early**
- 🗺️ Revenue by state → **SP (São Paulo) top revenue**
- ⭐ Review scores → **57.8% give 5 stars**
- 💰 Price vs freight → **Correlation = 0.29**

</td>
<td width="50%">

**Key EDA Numbers**
```
Total Revenue    : R$20,579,664
Avg Order Value  : R$172.74
Avg Early Deliver: 11 days early
Top Payment      : Credit Card 73%
Top State (Rev)  : São Paulo (SP)
Avg Review Score : 4.02 / 5.0
Price-Freight Cor: 0.29 (weak +ve)
```

</td>
</tr>
</table>

---

## ⚡ Spark SQL — Delay Analysis

**Why Spark over Pandas?** → Distributed computation across partitions for 1.19L rows with 9-table joins. Standard SQL syntax readable by stakeholders.

**Setup on Google Colab:**
```python
# Install Java + Spark
!apt-get install openjdk-8-jdk-headless -qq
!wget -q https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
!tar xf spark-3.4.1-bin-hadoop3.tgz
!pip install -q findspark

import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DelayAnalysis").getOrCreate()
spark_df = spark.createDataFrame(pandas_df)
spark_df.createOrReplaceTempView("orders")
```

**10 Queries & Results:**

| # | Query | Key Result |
|---|---|---|
| Q1 | Overall delay summary | **7.84% delay rate**, avg 12 days early |
| Q2 | Delay by seller state | SP most delayed; **MA = 23.59% rate** |
| Q3 | Top 5 worst sellers | 1 seller caused **2,266 total delay days** |
| Q4 | Delay by category | Audio **12.4%**, Christmas items **11.84%** |
| Q5 | Delay by customer state | **AL (Alagoas) = 24.49%** worst |
| Q6 | Monthly delay trend | **March 2018 = 20.36%** worst month |
| Q7 | Payment vs delay | **Boleto = 8.59%** (manual bank payment) |
| Q8 ⭐ | Review score vs delay | **1-star = 31.69%** delayed vs **5-star = 2.97%** |
| Q9 | Seller + category combo | Furniture seller **167-day avg delay** |
| Q10 | Save results | `delay_by_state.csv` + `review_vs_delay.csv` |

> **⭐ Q8 is the strongest business insight:** Proves delivery speed directly drives customer star ratings.

---

## 📈 Time Series Forecasting

**Goal:** Predict weekly order volumes for inventory and logistics planning.

**Stationarity Check (ADF Test):**
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(weekly['order_count'])
# p-value = 0.13 → NOT stationary (upward trend present)
# Fix: Apply first-order differencing → p-value drops to 0.0001 ✅
weekly['order_count_diff'] = weekly['order_count'].diff()
```

**Model Comparison:**

| Model | MAPE | Reason |
|---|---|---|
| SARIMA (1,1,1)(1,1,1,52) | 44,268% ❌ | Failed — incomplete weeks at data end |
| **Facebook Prophet** | **31.7% ✅** | **Auto-handles seasonality + changepoints** |

**Prophet Configuration:**
```python
model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(train_df)
future = model.make_future_dataframe(periods=52, freq='W')
forecast = model.predict(future)
```

**Seasonal Patterns Discovered:**
- 🛍️ **November: +80%** — Black Friday shopping surge
- ❄️ **June/July: −40%** — Winter slowdown in Brazil
- 📉 **January: −40%** — Post-holiday spending drop
- 📈 **Overall trend:** 0 → 3,000 orders/week over 2 years

---

## 🎯 Clustering & Segmentation

### Customer Segmentation — K-Means K=4

**Features (RFM Framework):**
- **R**ecency — days since last purchase
- **F**requency — number of unique orders
- **M**onetary — total spend amount

| Segment | Customers | Avg Spend | Business Action |
|---|---|---|---|
| 🔴 VIP Spenders | 22 | **R$26,932** | Personal account managers, exclusive deals |
| 🔵 High Spenders | 2,993 | R$457 | Upsell premium products |
| 🟢 Medium Spenders | 39,774 | R$201 | Push notifications, vouchers |
| 🟡 Low / Churned | 53,307 | R$199 | Win-back campaigns, big discounts |

> **Silhouette Score: 0.53** | K chosen via Elbow Method + business context

---

### Review-Based Clustering — K-Means K=3

**Features:** `review_score` · `comment_length` · `sentiment_score`

| Segment | Customers | Avg Score | Insight |
|---|---|---|---|
| 😊 Happy | 71,984 (74.9%) | 4.75 ⭐ | Ask for referrals, loyalty program |
| 😞 Unclear / Low Score | 16,198 (16.8%) | 1.97 ⭐ | Churn risk — immediate intervention needed |
| 😐 Mixed Feedback | 7,914 (8.2%) | 2.14 ⭐ | Wrote longest comments (avg 155 chars) |

> **Silhouette Score: 0.6875** — Very good cluster separation

---

### Seller Segmentation — K-Means K=4

**Features:** `total_revenue` · `total_orders` · `avg_order_value` · `avg_delay_days`

| Segment | Sellers | Avg Revenue | Avg Orders |
|---|---|---|---|
| 🟢 Top Performers | 22 | **R$1,98,610** | 996 orders |
| 🔴 Mid-Tier | 76 | R$12,903 | 5.74 orders |
| 🟠 Regular Sellers | 1,656 | R$7,556 | 38 orders |
| 🔵 Dormant Sellers | 1,341 | R$1,904 | 10 orders |

---

## 🧠 Sentiment Analysis (NLP)

**Model:** `distilbert-base-uncased-finetuned-sst-2-english` from 🤗 HuggingFace

**Why DistilBERT over traditional models?**

| Traditional (TF-IDF) | DistilBERT |
|---|---|
| Treats words independently | Understands full sentence context |
| Fails on "not bad" (reads "bad") | Correctly classifies "not bad" as positive |
| No semantic understanding | 768-dim embeddings per token |
| Train from scratch | Pre-trained on millions of texts |

**Inference Pipeline:**
```python
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Live demo result from Flask app:
result = sentiment_model("The product I received was really of not good quality")
# → {'label': 'NEGATIVE', 'score': 0.9997}
```

**Results on 40,809 Reviews:**

```
Total analyzed  : 40,809 reviews
Negative        : 31,152 (76.3%)
Positive        :  9,657 (23.7%)
Avg confidence  : 92.06%
Max confidence  : 99.97%
```

> ⚠️ **Limitation:** High negative rate because Olist reviews are in **Portuguese** but model is English-trained.  
> **Production fix:** Replace with `bert-base-multilingual-uncased` (supports 104 languages including Portuguese)

---

## 🌐 Flask Web Application

**7 pages, 6 routes, 4 Python modules** — all ML models served as live interactive pages.

```
localhost:5000/             → Homepage (hero, stats, feature cards)
localhost:5000/sentiment    → Customer Emotion Analysis (DistilBERT)
localhost:5000/delivery     → Delivery Delay Prevention
localhost:5000/clustering   → Buying Pattern Recognition (3 tabs)
localhost:5000/forecasting  → Sales Forecasting (Prophet)
localhost:5000/team         → Our Team
localhost:5000/support      → FAQ & Contact
```

**App Architecture:**
```python
# app.py — models load once on startup
def load_all_models():
    global sentiment_analyzer_obj, clustering_models_obj, delivery_analyzer_obj
    sentiment_analyzer_obj = SentimentAnalyzer()          # DistilBERT
    clustering_models_obj  = ClusteringModels(...)         # 3 PKL files
    delivery_analyzer_obj  = DeliveryAnalyzer()            # Rule-based

# Jinja2 template rendering
@app.route('/sentiment', methods=['GET','POST'])
def sentiment_route():
    result = sentiment_analyzer_obj.analyze_sentiment(text)
    return render_template('customer_emotion_analysis.html', result=result)
```

```html
<!-- HTML template — server-side rendering -->
<div>{{ result.sentiment }}</div>        <!-- Renders: "Negative" -->
<div>{{ result.confidence }}</div>       <!-- Renders: "99.97%"   -->
```

---

## ☁️ AWS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS PIPELINE                          │
│                                                             │
│  Local CSVs ──→ S3 (raw/)                                  │
│                    │                                        │
│                    ▼                                        │
│             SageMaker Notebook                              │
│             (reads from S3)                                 │
│                    │                                        │
│         ┌──────────┴──────────┐                            │
│         ▼                     ▼                             │
│    S3 (processed/)      S3 (models/)                       │
│    cleaned CSVs         .pkl files                          │
│                                                             │
│  IAM Role: SageMaker execution role                        │
│  → Read/write ecom-models-007 bucket only                  │
│  → Principle of least privilege                            │
└─────────────────────────────────────────────────────────────┘
```

| Service | Role |
|---|---|
| **S3** (`ecom-models-007`) | Data lake — raw CSVs, processed data, PKL models |
| **SageMaker** | Scalable training on `ml.m5.large` instance |
| **IAM** | Role-based access control for S3 bucket |

---

## 📊 Power BI Dashboard

Connected to processed CSVs to visualize:
- 📊 **Sales Overview** — R$16.01M total sales, 0.10M customers, 0.11M orders
- 📈 **Monthly Goals** — customer count vs order count vs actual sales
- 🗺️ **Sales per State** — treemap showing geographic distribution
- 🥧 **Payment Methods** — credit card 73%, boleto 18%
- 🏙️ **Top 5 Cities** — São Paulo R$1.89M, Rio de Janeiro R$0.98M
- 🛍️ **Top 10 Categories** — health_beauty 15%, watches_gifts 14%

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|---|---|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas, NumPy, PySpark 3.4.1 |
| **Visualization** | Matplotlib, Seaborn, Power BI |
| **Machine Learning** | Scikit-learn (K-Means, StandardScaler, t-SNE) |
| **Time Series** | Facebook Prophet, SARIMA (statsmodels) |
| **Deep Learning / NLP** | DistilBERT, HuggingFace Transformers, PyTorch |
| **Web Framework** | Flask, Jinja2 |
| **Cloud** | AWS S3, AWS SageMaker, AWS IAM |
| **Big Data** | Apache Spark 3.4.1 (PySpark), Spark SQL |
| **Environment** | Google Colab, Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |

</div>

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/muskanchauhan1/E_commerce_Analytics_-_Predictive_Platform.git
cd E_commerce_Analytics_-_Predictive_Platform
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Olist dataset
```
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
```
Place all 9 CSV files into the `data/` folder.

### 5. Run notebooks in order
```
notebooks/01_Cleaning.ipynb              → Local Jupyter
notebooks/02_Spark_SQL.ipynb             → Google Colab
notebooks/03_Time_Series.ipynb           → Google Colab
notebooks/04_Clustering_Customer.ipynb   → Google Colab
notebooks/05_Clustering_Review.ipynb     → Google Colab
notebooks/06_Clustering_Seller.ipynb     → Google Colab
notebooks/07_Sentiment_Analysis.ipynb    → Google Colab
```

### 6. Download PKL models from Google Drive → `flask_app/models/`
```
customer_clustering_model.pkl
review_clustering_model.pkl
seller_clustering_model.pkl
```

### 7. Run Flask app
```bash
cd flask_app
python app.py
```

### 8. Open in browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
ecommerce-analytics/
│
├── 📂 data/
│   ├── olist_orders_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── ... (all 9 CSV files)
│   ├── master_df.csv                   ← merged & cleaned
│   └── df_delivered.csv                ← delivered orders only
│
├── 📂 notebooks/
│   ├── 01_Cleaning.ipynb
│   ├── 02_Spark_SQL.ipynb
│   ├── 03_Time_Series.ipynb
│   ├── 04_Clustering_Customer.ipynb
│   ├── 05_Clustering_Review.ipynb
│   ├── 06_Clustering_Seller.ipynb
│   └── 07_Sentiment_Analysis.ipynb
│
├── 📂 flask_app/
│   ├── app.py                          ← Flask routes & startup
│   ├── sentiment_analyzer.py           ← DistilBERT wrapper
│   ├── clustering_models.py            ← PKL loader & predictor
│   ├── delivery_analyzer.py            ← Risk rule engine
│   ├── 📂 models/
│   │   ├── customer_clustering_model.pkl
│   │   ├── review_clustering_model.pkl
│   │   └── seller_clustering_model.pkl
│   ├── 📂 templates/
│   │   ├── index.html
│   │   ├── customer_emotion_analysis.html
│   │   ├── delivery_delay_prevention.html
│   │   ├── buying_pattern_recognition.html
│   │   ├── accurate_sales_forecasting.html
│   │   ├── team.html
│   │   └── support.html
│   └── 📂 static/
│       └── 📂 images/
│
├── 📂 aws/
│   └── train.py                        ← SageMaker training script
│
├── 📂 powerbi/
│   └── dashboard.pbix
│
├── requirements.txt
└── README.md
```

---

## 👥 Team

<div align="center">

**CDAC Mumbai · PG Diploma Big Data Analytics · Batch 2025**

| Name | Role |
|---|---|
| **Mr. Sumit Bansod** | Project Guide |
| Siddhant Sharma | Project Lead |
| Udhav Kardile | ML Engineer |
| Rahul Behra | Frontend Developer |
| Naresh Khanderay | Cloud Engineer |
| Advait Patil | Data Engineer |
| **Muskan Chauhan** | **Data Scientist** |
| Anjali Manohar | Data Analyst |
| Shubham Patil | Business Analyst |

</div>

---

<div align="center">

### ⭐ If this project helped you, please give it a star!

**Built with ❤️ by Team 008 — CDAC Mumbai**

![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

</div>
