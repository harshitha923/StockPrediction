import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_forecast_comparison(historical_data, test_data, forecasts, ticker):
    """
    Alternative forecast comparison using Plotly Express.
    """
    plot_df = pd.DataFrame({
        'Date': historical_data.index,
        'Price': historical_data['Close'],
        'Source': 'Historical'
    })

    # Append actual test data
    test_df = pd.DataFrame({
        'Date': test_data.index,
        'Price': test_data['Close'],
        'Source': 'Actual (Test)'
    })
    plot_df = pd.concat([plot_df, test_df])

    # Append forecasts
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, pd.DataFrame):
            forecast = forecast.values
        forecast = np.asarray(forecast).squeeze()  # ✅ flatten to 1D

        # Align with test_data dates
        forecast_dates = test_data.index[-len(forecast):]

        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Price': forecast,
            'Source': f'{model_name} Forecast'
        })
        plot_df = pd.concat([plot_df, forecast_df])

    # Plot using Plotly Express
    fig = px.line(
        plot_df,
        x='Date',
        y='Price',
        color='Source',
        title=f"{ticker} - Forecast Comparison"
    )

    fig.update_layout(
        height=500,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return fig

def plot_stock_data(df, ticker):
    """
    Create an interactive plotly visualization of stock data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with interactive stock chart
    """
    # Create a candlestick chart
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    df.index = pd.to_datetime(df.index)
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.3)'
        ),
        row=2, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'].rolling(window=20).mean(),
            line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
            name="20-day MA"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'].rolling(window=50).mean(),
            line=dict(color='rgba(255, 0, 0, 0.8)', width=2),
            name="50-day MA"
        ),
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_time_series_decomposition(df):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Close' column and datetime index
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with decomposition plots
    """
    # Perform time series decomposition
    decomposition = seasonal_decompose(
        df['Close'], 
        model='additive', 
        period=min(30, len(df) // 2)  # Use 30 days or half the length, whichever is smaller
    )
    
    # Create figure with 4 subplots
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual")
    )
    
    # Add traces for each component
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Close'], 
            mode='lines', 
            name='Original',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=decomposition.trend, 
            mode='lines', 
            name='Trend',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=decomposition.seasonal, 
            mode='lines', 
            name='Seasonal',
            line=dict(color='green')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=decomposition.resid, 
            mode='lines', 
            name='Residual',
            line=dict(color='purple')
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Time Series Decomposition",
        showlegend=False
    )
    
    return fig

def plot_forecast_comparisonx(historical_data, test_data, forecasts, ticker):
    """
    Plot comparison of different forecasting models.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Complete historical data
    test_data : pandas.DataFrame
        Test data used for evaluation
    forecasts : dict
        Dictionary of forecasts from different models
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with forecast comparisons
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        )
    )
    
    # Add actual test data
    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data['Close'],
            mode='lines',
            name='Actual (Test)',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add forecasts from different models
    colors = {
        'ARIMA/SARIMA': 'red',
        'Facebook Prophet': 'green',
        'LSTM': 'purple'
    }
    
    for model_name, forecast in forecasts.items():
        if len(forecast) > 0:  # Check if forecast data exists
            fig.add_trace(
                go.Scatter(
                    x=test_data.index[-len(forecast):],
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors.get(model_name, 'orange'))
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Model Comparison on Test Data",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        height=500
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_performance_metrics(metrics):
    """
    Create a bar chart comparing performance metrics across models.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of model performance metrics
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with performance metrics comparison
    """
    # Create dataframes for each metric
    models = list(metrics.keys())
    
    # Extract metrics
    rmse_values = [metrics[model]['RMSE'] for model in models]
    mae_values = [metrics[model]['MAE'] for model in models]
    mape_values = [metrics[model]['MAPE'] for model in models]
    
    # Create figure with 3 subplots side by side
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=("RMSE", "MAE", "MAPE (%)"),
        shared_yaxes=True
    )
    
    # Add bars for each metric
    colors = ['rgba(255, 99, 132, 0.7)', 'rgba(54, 162, 235, 0.7)', 'rgba(75, 192, 192, 0.7)']
    
    # RMSE
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name="RMSE",
            marker_color=colors[0]
        ),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae_values,
            name="MAE",
            marker_color=colors[1]
        ),
        row=1, col=2
    )
    
    # MAPE
    fig.add_trace(
        go.Bar(
            x=models,
            y=mape_values,
            name="MAPE (%)",
            marker_color=colors[2]
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Performance Metrics Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_forecast_with_confidence(historical_data, future_dates, forecast, confidence_intervals, ticker):
    """
    Plot future forecast with confidence intervals.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Recent historical data to show continuity
    future_dates : pandas.DatetimeIndex
        Dates for the forecast period
    forecast : numpy.array
        Forecast values
    confidence_intervals : numpy.array
        Confidence intervals (lower, upper bounds)
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with forecast and confidence intervals
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        )
    )
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=confidence_intervals[:, 0],
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=confidence_intervals[:, 1],
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            )
        )
    
    # Add forecast point annotations - handling potential Series objects
    try:
        # Get the first and last forecast values, ensuring they're plain numbers
        first_val = float(forecast[0]) if hasattr(forecast, "__getitem__") else float(forecast)
        last_val = float(forecast[-1]) if hasattr(forecast, "__getitem__") else float(forecast)
        
        # Create text labels with proper formatting
        first_label = f"${first_val:.2f}"
        last_label = f"${last_val:.2f}"
        
        fig.add_trace(
            go.Scatter(
                x=[future_dates[0], future_dates[-1]],
                y=[first_val, last_val],
                mode='markers+text',
                marker=dict(color='red', size=8),
                text=[first_label, last_label],
                textposition=["bottom center", "top center"],
                showlegend=False
            )
        )
    except Exception as e:
        # If annotation fails, continue without annotations
        print(f"Warning: Could not add price annotations to chart: {str(e)}")
        pass
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Future Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        height=500
    )
    
    # Add a vertical line to separate historical data from forecast
    fig.add_vline(
        x=historical_data.index[-1], 
        line_width=2, 
        line_dash="dash", 
        line_color="green",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    return fig
