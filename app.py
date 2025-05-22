import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(page_title="Stock Market Trend Analysis & Forecasting",page_icon="📈",layout="wide")

import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go

from data_handler import fetch_stock_data, preprocess_data
from model_arima import train_arima_model, forecast_arima
from model_prophet import train_prophet_model, forecast_prophet
from visualization import (plot_stock_data, plot_time_series_decomposition,
                           plot_forecast_comparison, plot_performance_metrics,
                           plot_forecast_with_confidence)
from utils import calculate_metrics
import plotly.express as px

# We'll focus solely on ARIMA and Prophet models

# Header information already set at the top of the file


# Set up caching for data fetching
@st.cache_data(ttl=3600)
def get_cached_stock_data(ticker, start_date, end_date):
    return fetch_stock_data(ticker, start_date, end_date)


def plot_actual_vs_forecast1(test_data, predictions, model_name="Model"):
    """
    Plot actual vs predicted values using Plotly in Streamlit.

    Parameters:
    - test_data (pd.DataFrame): DataFrame with datetime index and 'Close' column
    - predictions (array-like): Forecasted values (same length as test_data)
    - model_name (str): Name of the model used for title/legend
    """
    predictions = pd.Series(predictions)
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data['Close'].values ,
        'Forecast': predictions
    })

    # Melt for Plotly
    df_melted = df_plot.melt(id_vars='Date', value_vars=['Actual', 'Forecast'],
                             var_name='Type', value_name='Price')

    # Plot
    fig = px.line(df_melted, x='Date', y='Price', color='Type',
                  title=f'Actual vs Forecast - {model_name}',
                  labels={'Price': 'Stock Price', 'Date': 'Date'})

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend'
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.title("📈 Stock Market Trend Analysis & Forecasting")
    st.markdown("""
    This application performs time series analysis and forecasting on stock market data using multiple 
    forecasting models: ARIMA/SARIMA, Facebook Prophet, and LSTM neural networks.
    """)

    # Sidebar for inputs
    st.sidebar.header("Parameters")

    # Stock selection
    default_tickers = ["MSFT", "AAPL", "GOOGL", "AMZN", "TSLA"]
    ticker = st.sidebar.selectbox("Select a stock ticker:", default_tickers)

    # Allow custom ticker input
    custom_ticker = st.sidebar.text_input("Or enter a custom ticker:")
    if custom_ticker:
        ticker = custom_ticker.upper()

    # Date range selection
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=5 * 365)

    start_date = st.sidebar.date_input("Start date:", five_years_ago)
    end_date = st.sidebar.date_input("End date:", today)

    if start_date >= end_date:
        st.error("Error: End date must be after start date")
        return

    # Forecast horizon
    forecast_days = st.sidebar.slider("Forecast horizon (days):", 1, 90, 30)

    # Model selection - ARIMA and Prophet models
    model_options = ["ARIMA/SARIMA", "Facebook Prophet"]
    default_models = []

    models_to_use = st.sidebar.multiselect(
        "Select models to include in analysis:",
        model_options,
        default=default_models)

    # Load Data section
    st.header("Historical Stock Data")

    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch stock data
            df = get_cached_stock_data(ticker, start_date, end_date)

            if df.empty:
                st.error(
                    f"No data found for ticker {ticker}. Please try another stock symbol."
                )
                return

            # Display basic info
            st.success(
                f"Successfully loaded data for {ticker} from {start_date} to {end_date}"
            )

            # Show data statistics
            col1, col2, col3, col4 = st.columns(4)
            latest_price = float(df['Close'].iloc[-1])
            price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[0])
            percent_change = float(price_change) / float(df['Close'].iloc[0])

            col1.metric("Latest Price", f"${latest_price:.2f}")
            col2.metric("Change", f"${price_change:.2f}",
                        f"{percent_change:.2f}%")
            col3.metric("Average Volume", f"{float(df['Volume'].mean()):.0f}")
            col4.metric("Period", f"{float((end_date - start_date).days)} days")


            # Data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df)

            # Time Series Decomposition section
            st.header("🔍 Time Series Decomposition")
            st.write(
                "Decomposing the time series into trend, seasonal, and residual components:"
            )

            # Preprocess data for modeling
            df_processed = preprocess_data(df)

            # Plot time series decomposition
            decomp_fig = plot_time_series_decomposition(df_processed)
            st.plotly_chart(decomp_fig, use_container_width=True)

            # Forecasting section
            st.header("Forecasting Models")

            if not models_to_use:
                st.warning(
                    "Please select at least one forecasting model in the sidebar."
                )
                return

            # Prepare data for forecasting
            train_size = 0.8
            train_data = df_processed[:int(len(df_processed) * train_size)]
            test_data = df_processed[int(len(df_processed) * train_size):]

            forecasts = {}
            metrics = {}
            models_progress = st.progress(0)

            # Train and evaluate models based on user selection
            for i, model_name in enumerate(models_to_use):
                with st.spinner(f"Training {model_name} model..."):
                    if model_name == "ARIMA/SARIMA":
                        st.write(f"Model Name:ARIMA/ SARIMA")
                        model, predictions, forecast = train_arima_model(train_data, test_data, forecast_days)
                        st.info("Model Trained...")
                        #st.info(f"{predictions}")
                        forecasts["ARIMA/SARIMA"] = forecast
                        #st.info(f"{forecast}")
                        actual = test_data['Close'].reset_index(drop=True)
                        #st.info(f"Actual: {actual}")
                        predictions_ = pd.Series(predictions).reset_index(drop=True)
                        #st.info(f"Predictions: {predictions_}")
                        # Clean and flatten the data
                        actual = actual.iloc[:, 0].values  # Extract the only column from DataFrame
                        preds = predictions_.values            # Extract values from Series

                        # Create DataFrame for plotting
                        df_plot = pd.DataFrame({
                            'Index': range(len(actual)),
                            'Actual': actual,
                            'Predicted': preds
                        })

                        # Melt the DataFrame for easier plotting
                        df_melt = df_plot.melt(id_vars='Index', value_vars=['Actual', 'Predicted'],var_name='Type', value_name='Price')

                        fig = px.line(df_melt, x='Index', y='Price', color='Type',title=" Actual vs Predicted Prices",labels={'Price': 'Price ($)', 'Index': 'Time Step'})

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Actual vs Predicted Values"):
                            df_plot.set_index('Index', inplace=True)
                            st.dataframe(df_plot)
                        metrics["ARIMA/SARIMA"] = calculate_metrics(actual,predictions_)
                        
                    elif model_name == "Facebook Prophet":
                        st.write("Model Name: Facebook Prophet")
                        model, predictions, forecast = train_prophet_model(train_data, test_data, forecast_days)
                        st.info("Model Trained...")
                        forecasts["Facebook Prophet"] = forecast
                        st.info(f"{forecast}")
                        actual = test_data['Close'].reset_index(drop=True)
                        predictions_ = pd.Series(predictions).reset_index(drop=True)
                        actual = actual.iloc[:, 0].values  # Extract the only column from DataFrame
                        preds = predictions_.values            # Extract values from Series

                        # Create DataFrame for plotting
                        df_plot = pd.DataFrame({
                            'Index': range(len(actual)),
                            'Actual': actual,
                            'Predicted': preds
                        })

                        # Melt the DataFrame for easier plotting
                        df_melt = df_plot.melt(id_vars='Index', value_vars=['Actual', 'Predicted'],var_name='Type', value_name='Price')

                        fig = px.line(df_melt, x='Index', y='Price', color='Type',title=" Actual vs Predicted Prices",labels={'Price': 'Price ($)', 'Index': 'Time Step'})

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Actual vs Predicted Values"):
                            df_plot.set_index('Index', inplace=True)
                            st.dataframe(df_plot)
                        metrics["Facebook Prophet"] = calculate_metrics(test_data['Close'], predictions)

                    # Note: LSTM model removed from this version to ensure compatibility

                models_progress.progress((i + 1) / len(models_to_use))

            # Model Comparison section
            st.header("Model Comparison")
            # Performance metrics
            st.subheader("Performance Metrics")
            metrics_fig = plot_performance_metrics(metrics)
            st.plotly_chart(metrics_fig, use_container_width=True)

            # Display metrics in a table
            st.markdown("### Detailed Metrics")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df)

            # Display best model
            best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0]
            st.success(
                f"Best performing model based on RMSE: **{best_model}**")

            # Future forecast with the best model
            st.header("Future Stock Price Prediction")
            st.write(
                f"Using the best model ({best_model}) to forecast the next {forecast_days} days"
            )

            # Generate future dates for forecasting
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),periods=forecast_days)
            st.info(f"Future dates: {future_dates}")
            # Get confidence intervals for the best model forecast
            if best_model == "ARIMA/SARIMA":
                future_forecast, conf_int = forecast_arima(
                    df_processed, forecast_days)
            elif best_model == "Facebook Prophet":
                future_forecast, conf_int = forecast_prophet(
                    df_processed, forecast_days)
            else:
                future_forecast, conf_int = forecast_prophet(
                    df_processed, forecast_days)
            st.info(f"futrure forecast: {future_forecast}")
            
            future_forecast = np.array(future_forecast)
            
            # Create DataFrame for plotting
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': future_forecast
            })

            # Plot
            fig = px.line(forecast_df, x='Date', y='Forecast', title='Future Forecast (Next 30 Days)')
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Forecasted Price',
                
            )
            # Show in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Display forecast values
            with st.expander("View Detailed Forecast Values"):
                forecast_df = pd.DataFrame({
                    'Date':
                    future_dates,
                    'Forecast':
                    future_forecast,
                    'Lower Bound':
                    conf_int[:, 0] if conf_int is not None else [None] *
                    len(future_forecast),
                    'Upper Bound':
                    conf_int[:, 1] if conf_int is not None else [None] *
                    len(future_forecast)
                })
                forecast_df.set_index('Date', inplace=True)
                st.dataframe(forecast_df)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try a different stock ticker or date range.")


if __name__ == "__main__":
    main()
