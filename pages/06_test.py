# Import libraries
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)

# functions

# Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")         


# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/tickers.csv")
except:
    logging.error('Cannot find the CSV file')

# Get data for the selected ticker
ticker_data = yf.Ticker(ticker)

# Benchmark ticker - S&P Global index '^GSPC'
benchmark_ticker=yf.Ticker("^GSPC")

# Add start/end dates to streamlit sidebar
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("1997-1-1"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

# Create a new dataframe - add historical trading period for 1 day
ticker_df = ticker_data.history(period="1d", start=start_date, end=end_date)

# Query S&P Global historical prices
benchmark_ticker = benchmark_ticker.history(period="1d", start=start_date, end=end_date)

    
#-----------------------------------------------#

# Time Series Forecasting with Facebook Prophet
prophet_check_box = st.checkbox(label=f"Display {ticker} Prophet time series forecast data")
if prophet_check_box:
    # Input a streamlit slider with years of prediction values
    n_years = st.slider("Select year(s) for time series forecast", 1, 5)

    # Create Prophet model
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(ticker_df)

    # Make future predictions
    future_predictions = prophet_model.make_future_dataframe(periods=365*n_years)
    forecast = prophet_model.predict(future_predictions)

    # Plot predictions
    prophet_plot = plot_plotly(prophet_model, forecast)
    st.plotly_chart(prophet_plot)

    
#-----------------------------------------------#


# Time Series prediction using XGBoost
xgb_check_box = st.checkbox(label=f"Display {ticker} XGBoost time series prediction data")
if xgb_check_box:
    # Split data into train and test sets
    X = ticker_df.drop(['Close'], axis=1)
    y = ticker_df[['Close']]
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train XGBoost model
    xgb_model = XGBRegressor()
    parameters = {'nthread':[4], 
                  'objective':['reg:linear'],
                  'learning_rate': [.03, 0.05, .07], 
                  'max_depth': [5, 6, 7],
                  'min_child_weight': [4],
                  'silent': [1],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500]}
    xgb_grid = GridSearchCV(xgb_model, parameters, cv = 2, n_jobs = 5, verbose=True)
    xgb_grid.fit(X_train, y_train)

    # Make predictions
    predictions = xgb_grid.predict(X_test)
    st.write("Mean Squared Error: ", mean_squared_error(predictions, y_test))

    # Plot predictions
    st.line_chart(y_test.assign(pred=predictions))
    
    # Compare stock performance with benchmark
    performance_check_box = st.checkbox(label=f"Display {ticker} performance vs benchmark")
    if performance_check_box:
        # Merge dataframes and calculate returns
        merged_df = pd.merge(ticker_df, benchmark_ticker, on='Date')
        merged_df['Ticker_returns'] = merged_df['Close_x'].pct_change()
        merged_df['Benchmark_returns'] = merged_df['Close_y'].pct_change()
        
        # Plot returns
        st.line_chart(merged_df[['Ticker_returns', 'Benchmark_returns']])
        
        # Add summary section
        summary_check_box = st.checkbox(label=f"Display {ticker} summary statistics")
        if summary_check_box:
            st.write("---")
            st.write("Summary statistics for", ticker)
            st.write("---")
            st.write("Open: ", ticker_df['Open'].mean())
            st.write("Close: ", ticker_df['Close'].mean())
            st.write("High: ", ticker_df['High'].mean())
            st.write("Low: ", ticker_df['Low'].mean())
            st.write("Volume: ", ticker_df['Volume'].mean())

            
#-----------------------------------------------#