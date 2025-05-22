import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics for model evaluation.

    Parameters:
    -----------
    actual : array-like
        Actual values (e.g., pandas Series or numpy array)
    predicted : array-like
        Predicted values (e.g., pandas Series or numpy array)

    Returns:
    --------
    dict
        Dictionary containing RMSE, MAE, and MAPE
    """
    # Convert to numpy arrays for safety
    actual = np.array(actual.squeeze())
    predicted = np.array(predicted)
    
    # Ensure we only compare valid predictions (in case of different lengths)
    min_len = min(len(actual), len(predicted))
    actual = actual[-min_len:]
    predicted = predicted[-min_len:]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate MAPE (avoiding division by zero)
    mask = actual != 0
    if np.any(mask):
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan  # Or choose a fallback value like 0 or float("inf")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def generate_future_dates(last_date, horizon):
    """
    Generate future dates for forecasting.
    
    Parameters:
    -----------
    last_date : pandas.Timestamp or datetime.datetime
        Last date in the historical data
    horizon : int
        Number of days to forecast
        
    Returns:
    --------
    pandas.DatetimeIndex
        Index of future dates
    """
    import pandas as pd
    
    # Generate daily dates starting from the day after the last date
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    return future_dates

def calculate_trading_signals(data, short_window=20, long_window=50):
    """
    Calculate trading signals based on moving average crossover.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with 'Close' prices
    short_window : int
        Short moving average window
    long_window : int
        Long moving average window
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added signal columns
    """
    # Create a copy of the data
    signals = data.copy()
    
    # Create short and long moving averages
    signals['short_ma'] = signals['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = signals['Close'].rolling(window=long_window).mean()
    
    # Create signals: 1 for buy, -1 for sell, 0 for hold
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 
        1, 
        0
    )
    
    # Generate trading signal changes (entry/exit points)
    signals['position_change'] = signals['signal'].diff()
    
    return signals
