# Adaptive Portfolio Optimization Across Risk Profiles

A research-focused implementation of my Master's Thesis at **IIT Kharagpur** on building and evaluating a **Hybrid LSTM–Transformer** framework for portfolio allocation under different investor risk appetites.

This repository contains the complete end-to-end pipeline: data collection, feature engineering, stock selection, model training, backtesting, and performance evaluation.

---

## Author
**Kartik Vij**  
Dual Degree (B.Tech Chemical Engineering + M.Tech Financial Engineering), IIT Kharagpur (2020–2025)

- Email: **vijkartik2002@gmail.com**
- LinkedIn: **[Kartik Vij](https://www.linkedin.com/in/kartik-vij-cqf-87b6b3207/)**

---

## Project Overview

### Objective
Design a practical deep-learning portfolio framework that dynamically allocates capital across equities for three risk profiles:
- **Risk Averse**
- **Risk Neutral**
- **Risk Seeking**

### Highlights
- Uses **18 years of NSE equity data** across multiple sectors.
- Combines sequential modeling (**LSTM**) with cross-asset representation learning (**Transformer**).
- Generates profile-specific portfolio weights and evaluates out-of-sample performance.
- Compares strategy behavior against benchmark index data (`^NSEI.csv`, `^NSEI_test.csv`).

---

## Repository Structure

```text
.
├── data/                         # Training-period stock price data
├── test_data/                    # Test-period stock price data
├── technical_data/               # Technical indicators per stock
├── fundamental_data/             # Fundamental features per stock
├── correlation_data/             # Inter-stock correlation artifacts
├── performance_data/             # Performance summaries and analysis tables
├── categorized_results/          # Intermediate stock category outputs
├── results/                      # Final portfolio weights and test allocations
├── models/
│   ├── risk_averse/              # Saved model checkpoints (risk-averse)
│   ├── risk_neutral/             # Saved model checkpoints (risk-neutral)
│   └── risk_seeking/             # Saved model checkpoints (risk-seeking)
├── notebooks/                    # Visualizations and exploratory analysis
├── fetch_data.py                 # Download training data
├── fetch_data_test.py            # Download test-period data
├── fetch_technical_data.py       # Build technical indicator dataset
├── fetch_fundamental_data.py     # Build fundamental feature dataset
├── compute_correlation.py        # Correlation computation across assets
├── categorise_stocks.py          # Stock categorization by selected logic
├── rank_and_select_stocks.py     # Ranking and final stock universe selection
├── preprocess.py                 # Data preprocessing for model inputs
├── model.py                      # Hybrid LSTM–Transformer architecture
├── train_model.py                # Training pipeline
├── test_model.py                 # Evaluation/testing pipeline
├── compute_performance_metrics.py# Risk/return performance metrics
├── ^NSEI.csv                     # Benchmark data (train period)
└── ^NSEI_test.csv                # Benchmark data (test period)
```

---

## Pipeline

1. **Data Ingestion**  
   Fetch market, technical, and fundamental data.

2. **Feature Construction & Selection**  
   Compute correlations, categorize stocks, and rank/select the investment universe.

3. **Preprocessing**  
   Transform raw inputs into model-ready tensors.

4. **Model Training**  
   Train the Hybrid LSTM–Transformer model separately for each risk profile.

5. **Backtesting & Evaluation**  
   Generate portfolio weights and compute performance metrics against benchmark behavior.

---

## How to Run

> Run from the repository root.

```bash
# 1) Fetch and prepare datasets
python fetch_data.py
python fetch_data_test.py
python fetch_technical_data.py
python fetch_fundamental_data.py

# 2) Build selection inputs
python compute_correlation.py
python categorise_stocks.py
python rank_and_select_stocks.py

# 3) Preprocess + train + test
python preprocess.py
python train_model.py
python test_model.py

# 4) Compute final metrics
python compute_performance_metrics.py
```

---

## Key Outputs

- **Portfolio weights** for each risk profile in `results/`
- **Saved model artifacts** in `models/`
- **Performance summaries** in `performance_data/`
- **Visual analysis notebooks** in `notebooks/`

---

## Resume Context

This project demonstrates:
- Time-series modeling for financial decision systems
- Portfolio construction under practical constraints
- End-to-end ML research workflow (data to deployable signals)
- Quantitative performance evaluation for risk-aware investing

---

## Notes

- This is an academic/research project and not investment advice.
- If you are reviewing this via my resume, feel free to reach out for a walkthrough of methodology, experiments, and results.
