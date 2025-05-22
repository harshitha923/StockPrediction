import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt

def prepare_prophet_data(data):
    st.info("Data preparing...")
    prophet_data = pd.DataFrame({
        'ds': data.index,
        'y': data['Close'].squeeze()
    })
    st.info("Data prepared!...")
    return prophet_data

def train_prophet_model(train_data, test_data, forecast_horizon=30):
    # Prepare data for Prophet
    prophet_train = prepare_prophet_data(train_data)
    st.info("Data prepared")
    # Create and train the model
    with st.spinner("Training Prophet model..."):
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        # Add potentially relevant regressors from the dataset
        if 'rolling_mean_7' in train_data.columns:
            prophet_train['rolling_mean_7'] = train_data['rolling_mean_7'].values
            model.add_regressor('rolling_mean_7')
        
        if 'rolling_std_7' in train_data.columns:
            prophet_train['rolling_std_7'] = train_data['rolling_std_7'].values
            model.add_regressor('rolling_std_7')
        
        # Fit the model
        model.fit(prophet_train)
        st.info("Prophet model trained")
    # Create test data for prediction
    prophet_test = pd.DataFrame({'ds': test_data.index})
    st.info("Prophet test data created ")
    # Add the same regressors to test data
    if 'rolling_mean_7' in train_data.columns:
        prophet_test['rolling_mean_7'] = test_data['rolling_mean_7'].values
    
    if 'rolling_std_7' in train_data.columns:
        prophet_test['rolling_std_7'] = test_data['rolling_std_7'].values
    
    # Make predictions on the test set
    with st.spinner("Making Prophet predictions..."):
        test_forecast = model.predict(prophet_test)
        predictions = test_forecast['yhat'].values
    
    # Make future forecasts
    with st.spinner("Forecasting future values with Prophet..."):
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon, freq='D')
        
        # Prepare full regressor arrays
        if 'rolling_mean_7' in train_data.columns:
            full_rolling_mean_7 = pd.concat([
                prophet_train['rolling_mean_7'],
                pd.Series([train_data['rolling_mean_7'].iloc[-1]] * forecast_horizon)
            ], ignore_index=True)
        
            if len(future) == len(full_rolling_mean_7):
                future['rolling_mean_7'] = full_rolling_mean_7.values
            else:
                st.warning("Length mismatch for rolling_mean_7")

        if 'rolling_std_7' in train_data.columns:
            full_rolling_std_7 = pd.concat([
                prophet_train['rolling_std_7'],
                pd.Series([train_data['rolling_std_7'].iloc[-1]] * forecast_horizon)
            ], ignore_index=True)
        
            if len(future) == len(full_rolling_std_7):
                future['rolling_std_7'] = full_rolling_std_7.values
            else:
                st.warning("Length mismatch for rolling_std_7")
        # Make forecast
        
        forecast = model.predict(future)
        future_forecast = forecast.iloc[-forecast_horizon:]['yhat'].values
    
    return model, predictions, future_forecast

def forecast_prophet(data, forecast_horizon=30):
    # Prepare data for Prophet
    prophet_data = prepare_prophet_data(data)
    
    # Create and train the model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    # Add potentially relevant regressors
    if 'rolling_mean_7' in data.columns:
        prophet_data['rolling_mean_7'] = data['rolling_mean_7'].values
        model.add_regressor('rolling_mean_7')
    
    if 'rolling_std_7' in data.columns:
        prophet_data['rolling_std_7'] = data['rolling_std_7'].values
        model.add_regressor('rolling_std_7')
    
    # Fit the model
    model.fit(prophet_data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_horizon, freq='D')
    
    # Add the last known values of regressors (if used)
    if 'rolling_mean_7' in prophet_data.columns:
            full_rolling_mean_7 = pd.concat([
                prophet_data['rolling_mean_7'],
                pd.Series([prophet_data['rolling_mean_7'].iloc[-1]] * forecast_horizon)
            ], ignore_index=True)
        
            if len(future) == len(full_rolling_mean_7):
                future['rolling_mean_7'] = full_rolling_mean_7.values
            else:
                st.warning("Length mismatch for rolling_mean_7")

    if 'rolling_std_7' in prophet_data.columns:
            full_rolling_std_7 = pd.concat([
                prophet_data['rolling_std_7'],
                pd.Series([prophet_data['rolling_std_7'].iloc[-1]] * forecast_horizon)
            ], ignore_index=True)
        
            if len(future) == len(full_rolling_std_7):
                future['rolling_std_7'] = full_rolling_std_7.values
            else:
                st.warning("Length mismatch for rolling_std_7")
    forecast = model.predict(future)
    
    # Extract future forecast values and confidence intervals
    future_forecast = forecast.iloc[-forecast_horizon:]['yhat'].values
    confidence_intervals = np.column_stack([
        forecast.iloc[-forecast_horizon:]['yhat_lower'].values,
        forecast.iloc[-forecast_horizon:]['yhat_upper'].values
    ])
    
    return future_forecast, confidence_intervals
