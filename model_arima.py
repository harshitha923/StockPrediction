import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def test_stationarity(timeseries):
    """
    Test for stationarity using the Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Time series data to test
        
    Returns:
    --------
    bool
        True if stationary, False otherwise
    """
    # Perform Dickey-Fuller test
    result = adfuller(timeseries.dropna())
    p_value = result[1]
    
    # If p-value is less than 0.05, we can reject the null hypothesis and consider the series stationary
    return p_value < 0.05

def find_optimal_params(train_data):
    """
    Find optimal ARIMA/SARIMA parameters using a grid search approach.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with 'Close' column
        
    Returns:
    --------
    tuple
        (p, d, q, P, D, Q, s) parameters for SARIMA model
    """
    with st.spinner("Finding optimal ARIMA parameters..."):
        # Check if the series is stationary
        is_stationary = test_stationarity(train_data['Close'])
        
        # Set reasonable defaults based on stationarity
        d = 0 if is_stationary else 1
        
        # Try common parameter combinations
        p_values = range(0, 3)
        q_values = range(0, 3)
        
        # Keep seasonal parameters simple
        P, D, Q, s = 0, 0, 0, 0
        
        # Check different parameter combinations
        best_aic = float("inf")
        best_order = None
        
        # Only check a limited number of combinations for performance
        for p, q in itertools.product(p_values, q_values):
            try:
                model = ARIMA(
                    train_data['Close'],
                    order=(p, d, q),
                )
                results = model.fit()
                
                # Choose the model with the lowest AIC
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue
        
        # If no model could be fit, use sensible defaults
        if best_order is None:
            best_order = (1, d, 1)
            
        st.info(f"Selected ARIMA parameters: {best_order}")
        
        return best_order[0], best_order[1], best_order[2], P, D, Q, s

def train_arima_model(train_data, test_data, forecast_horizon=30):
    """
    Train an ARIMA/SARIMA model and make predictions.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with 'Close' column
    test_data : pandas.DataFrame
        Test data with 'Close' column
    forecast_horizon : int
        Number of days to forecast
        
    Returns:
    --------
    tuple
        (model, predictions, forecast)
    """
    # Find optimal parameters
    try:
        p, d, q, P, D, Q, s = find_optimal_params(train_data)
        st.info(f"Optimal SARIMA parameters: ({p},{d},{q})({P},{D},{Q}){s}")
    except Exception as e:
        st.warning(f"Error finding optimal parameters: {str(e)}. Using default parameters.")
        p, d, q, P, D, Q, s = 1, 1, 1, 0, 0, 0, 0
    
    # Train SARIMA model
    if P > 0 or D > 0 or Q > 0:
        model = SARIMAX(
            train_data['Close'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s)
        )
    else:
        # Use regular ARIMA if seasonal components are not significant
        model = ARIMA(train_data['Close'], order=(p, d, q))
    
    # Fit the model
    with st.spinner("Fitting ARIMA/SARIMA model..."):
        fitted_model = model.fit()
    st.info("Model trained")
    # Make predictions on the test set
    with st.spinner("Making predictions..."):
        predictions = fitted_model.forecast(steps=len(test_data))
    
    # Make future forecasts
    with st.spinner("Forecasting future values..."):
        future_forecast = fitted_model.forecast(steps=forecast_horizon)
    
    return fitted_model, predictions, future_forecast.values

def forecast_arima(data, forecast_horizon=30):
    """
    Generate forecasts with confidence intervals using the ARIMA/SARIMA model.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with 'Close' column
    forecast_horizon : int
        Number of days to forecast
        
    Returns:
    --------
    tuple
        (forecast values, confidence intervals)
    """
    # Find optimal parameters
    try:
        p, d, q, P, D, Q, s = find_optimal_params(data)
    except Exception:
        p, d, q, P, D, Q, s = 1, 1, 1, 0, 0, 0, 0
    
    # Train SARIMA model
    if P > 0 or D > 0 or Q > 0:
        model = SARIMAX(
            data['Close'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s)
        )
    else:
        # Use regular ARIMA if seasonal components are not significant
        model = ARIMA(data['Close'], order=(p, d, q))
    
    # Fit the model
    fitted_model = model.fit()
    
    # Make future forecasts with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
    forecast_values = forecast_result.predicted_mean
    lower_bounds = conf_int_df.iloc[:, 0].values
    upper_bounds = conf_int_df.iloc[:, 1].values
    confidence_intervals = np.column_stack((lower_bounds, upper_bounds))

    return forecast_values.values, confidence_intervals



