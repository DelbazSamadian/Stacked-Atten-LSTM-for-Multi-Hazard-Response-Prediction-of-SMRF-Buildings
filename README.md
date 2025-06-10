# Stacked-Atten-LSTM-for-Multi-Hazard-Response-Prediction-of-SMRF-Buildings

This project implements a stacked attention-based LSTM model to predict Maximum Interstory Drift Ratio (MIDR) for SMRF buildings subjected to earthquakes. It uses ground motion time histories and key structural features.

## Features

- Streamlit-based GUI for user-friendly predictions
- Upload paired ground motion components (Sa1 and Sa2)
- Enter top 10 important features manually
- Model trained on standardized data using PyTorch

## Prerequisites

- Python 3.8+
- PyTorch
- Streamlit
- Numpy, Pandas

Install dependencies:
```bash
pip install -r requirements.txt

