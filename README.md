# 📊 Token-Efficient Data Science Agent

An adaptive data analysis system that compresses dataset schemas and significantly reduces Large Language Model (LLM) token costs through intelligent information prioritization.

---

## 🚀 Overview

When using LLMs for dataset analysis, entire datasets are often sent as raw CSV or JSON input.  
For medium to large datasets, this results in:

- High token consumption
- Increased API cost
- Redundant data processing
- Slower inference

This project builds a **Token-Efficient Data Science Agent** that:

- Automatically performs exploratory data analysis (EDA)
- Scores column importance
- Applies adaptive schema compression
- Benchmarks token reduction
- Simulates LLM cost savings (in INR)

Instead of sending full datasets to an LLM, the system sends a compact, structured representation.

---

## 🔍 Key Features

### ✅ Automated EDA
- Schema summarization
- Missing value analysis
- IQR-based outlier detection
- Strong correlation detection

### ✅ Adaptive Column Importance Ranking
Each column is scored using:
- Log-normalized variance (numeric columns)
- Maximum correlation magnitude
- Uniqueness ratio (categorical columns)
- Missing value penalty

Top-ranked columns retain detailed statistics.  
Lower-ranked columns are aggressively compressed.

### ✅ Intelligent Schema Compression
- Full detail for top 3 important columns
- Lightweight summaries for remaining columns
- Compact, structured representation

### ✅ Token Optimization Benchmark
- Estimates raw dataset token usage
- Estimates compressed schema token usage
- Computes:
  - Token reduction percentage
  - Compression ratio

### ✅ LLM Cost Simulation (INR)
Simulates GPT-style pricing to estimate:
- Raw cost
- Compressed cost
- Savings per dataset
- Scaled monthly savings potential

---

## 📊 Example Performance

For a dataset with:

- 958 rows
- 10 columns

Results:

- Raw Tokens: 20,027
- Compressed Tokens: 160
- Token Reduction: **99.2%**
- Compression Ratio: **125x smaller**

This demonstrates significant cost reduction potential when integrating with LLM-based workflows.

---

## 🏗️ System Architecture

CSV Upload
↓
Data Cleaning
↓
Automated EDA
↓
Column Importance Scoring
↓
Adaptive Schema Compression
↓
Token Estimation
↓
Cost Simulation
↓
Dashboard Visualization


---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib

---

## ▶️ How to Run Locally

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/token-efficient-data-science-agent.git

cd token-efficient-data-science-agent


2. Install dependencies:

pip install -r requirements.txt


3. Run the app:

streamlit run app.py


4. Open in browser:

http://localhost:8501


---

## 📌 Design Decisions

- Used log-normalized variance to avoid scale distortion.
- Correlation magnitude used to detect structural relationships.
- Missing-value penalty reduces importance score.
- Token estimation uses 1 token ≈ 4 characters approximation.
- Cost simulation based on example GPT-style pricing.

---

## ⚠️ Assumptions

- Token estimation is approximate.
- Pricing is simulated for demonstration.
- Designed for structured CSV datasets.

---

## 🔮 Future Improvements

- Entropy-based compression
- Multi-dataset comparison
- LLM API integration
- Streamlit Cloud deployment
- Dynamic real-time pricing integration
- Prompt generation module

---

## 🧠 What This Project Demonstrates

- Systems thinking
- Cost-aware AI design
- Intelligent information prioritization
- Practical LLM workflow optimization
- Adaptive compression logic

---

Built as part of an AI optimization challenge to demonstrate real-world reduction in LLM operational costs.
