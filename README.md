# 🚦 Traffic Prediction Engine using Time-Series & Machine Learning

A complete end-to-end **Traffic Forecasting System** that predicts future traffic volume using **Time-Series models (ARIMA, LSTM)** and **Machine Learning (XGBoost)**.

This project demonstrates real-world ML pipeline development including **data preprocessing, feature engineering, model building, evaluation, and visualization**.

---

## 📌 Project Overview

Urban traffic congestion is a major problem in modern cities.
This project builds a **Traffic Prediction Engine** capable of forecasting traffic flow using historical data and environmental factors such as weather and temperature.

The system supports:

* Time-series forecasting
* Deep learning sequence modeling
* Gradient boosting regression

---

## 🧠 Models Implemented

* **ARIMA (AutoRegressive Integrated Moving Average)**
  Classical statistical model for time-series forecasting

* **LSTM (Long Short-Term Memory)**
  Deep learning model for sequential data

* **XGBoost Regressor**
  Powerful machine learning model using engineered features

---

## ⚙️ Tech Stack

* Python
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* statsmodels
* tensorflow / keras
* xgboost
* streamlit (for dashboard)

---

## 📂 Project Structure

```
traffic_prediction_engine/
│
├── dataset.csv
├── data_preprocessing.py
├── feature_engineering.py
├── arima_model.py
├── lstm_model.py
├── xgboost_model.py
├── evaluation.py
├── main.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset contains historical traffic data with the following features:

| Column Name    | Description              |
| -------------- | ------------------------ |
| date_time      | Timestamp of observation |
| traffic_volume | Number of vehicles       |
| temperature    | Temperature (Kelvin)     |
| rain           | Rainfall amount          |
| holiday        | Holiday indicator        |
| weather        | Weather condition        |

Example:

```
2012-10-02 09:00:00, 5545, 288.28, 0, None, Clear
```

---

## 🔄 Workflow Pipeline

```
Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training (ARIMA, LSTM, XGBoost)
      ↓
Evaluation (MAE, RMSE, MAPE)
      ↓
Visualization
      ↓
Future Traffic Prediction
```

---

## 🧪 Features Implemented

### ✅ Data Preprocessing

* Datetime conversion
* Missing value handling
* One-hot encoding
* Feature scaling

### ✅ Feature Engineering

* Hour, Day, Month extraction
* Lag features (lag_1, lag_2)
* Rolling mean

### ✅ Models

* ARIMA for statistical forecasting
* LSTM for sequence learning
* XGBoost for regression

### ✅ Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)

---

## 📈 Results (Sample)

```
MAE  : 0.038
RMSE : 0.051
MAPE : 4.21%
```

XGBoost performed best due to effective feature engineering.

---

## 📉 Visualization

* Traffic trends over time
* Correlation heatmap
* Actual vs Predicted comparison

---

## 🔮 Future Prediction

The system includes a function:

```python
predict_traffic(next_hours)
```

It predicts traffic for upcoming hours based on trained models.

---

## 🚀 How to Run

### 1. Clone Repository

```
git clone https://github.com/yourusername/traffic-prediction-engine.git
cd traffic-prediction-engine
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Main Pipeline

```
python main.py
```

### 4. Run Dashboard (Optional)

```
streamlit run dashboard.py
```

---

## 📊 Streamlit Dashboard

Interactive dashboard to:

* View dataset
* Predict traffic for future hours
* Visualize outputs

---

## 🎯 Use Cases

* Smart city traffic management
* Navigation systems
* Logistics & route optimization
* Urban planning

---

## 🔥 Key Highlights

* End-to-end ML pipeline
* Multiple model comparison
* Real-world dataset handling
* Modular code structure
* Dashboard integration

---

## 📌 Future Improvements

* Add Facebook Prophet model
* Use real-time traffic APIs
* Deploy using Flask / FastAPI
* Integrate with Google Maps API
* Hyperparameter tuning

---

## 👨‍💻 Author

**Your Name**
BTech CSE (AIML)
Aspiring Machine Learning Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

---
