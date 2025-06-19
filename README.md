# 📈 Stock Price Forecasting Using Time-Series Models

This project implements and compares multiple time-series forecasting models — ARIMA, LSTM, and Facebook Prophet — to predict stock prices based on historical market data. The objective is to evaluate the accuracy and efficiency of these models in real-world scenarios and identify the most suitable method for short-term financial forecasting.

---

## 🚀 Features

- 📊 Time-series analysis of stock prices
- 🧠 Forecasting using:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - LSTM (Long Short-Term Memory Neural Network)
  - Facebook Prophet (additive model for trend + seasonality)
- 📉 Performance evaluation using MAE, RMSE, and visual plots
- 🗂️ Data preprocessing and trend decomposition
- 📈 Visualization of actual vs. predicted prices

---

## 🛠️ Technologies Used

- Python 3.x
- NumPy, Pandas, Matplotlib
- Statsmodels (for ARIMA)
- TensorFlow / Keras (for LSTM)
- Facebook Prophet

---

## ⚙️ How to Run

1. **Clone the repository**

Run:

     git clone https://github.com/harshitha923/StockPrediction
     cd stock-price-forecasting

2. **Install dependencies**

Run:

      pip install numpy pandas matplotlib statsmodels tensorflow prophet
3. **Data**

yfinace python library is used for stock data. Use the following command to use it:

    pip install yfinance

4.**Run any model script**

    python arima_model.py
    python lstm_model.py
    python prophet_model.py
    
5. **Sample Output**
Plots of actual vs. predicted stock prices

MAE and RMSE printed in terminal

Time-series decomposition charts

Prophet forecast components (trend, seasonality)

---

# 📊 Evaluation Metrics
MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Visual comparison of model outputs for performance benchmarking

📌 Limitations
LSTM models may require GPU or high RAM for long sequences

ARIMA requires stationary data

Prophet performs best with longer time spans and clear seasonality
