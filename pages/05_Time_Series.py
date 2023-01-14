# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
# import yfinance for stock data
import yfinance as yf
#import prophet libraries
from prophet import Prophet
from prophet.plot import plot_plotly
#import plotly for interactive graphs
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import requests
# import cufflinks for bollinger bands
import cufflinks as cf

from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Stocks analysis",page_icon="📈",layout="centered",initial_sidebar_state="auto")


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

# Benchmark ticker - S&P Global index '^GSPC'
benchmark_ticker=yf.Ticker("^GSPC")


# Display a selectbox for the user to choose a ticker
ticker = st.sidebar.selectbox("Select a ticker from the dropdown menu",tickers)


# Get data for the selected ticker
ticker_data = yf.Ticker(ticker)

# wrap header content in a streamlit container
with st.container():
        # 2 columns section:
        col1, col2 = st.columns([3, 2])
        with col1:           
            # Load title/info
            st.header(f"Time Series prediction")
            st.markdown(f"Facebook Prophet, XGBoost")
        with col2:
            st.empty()
st.write("---")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("1997-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# Create a new dataframe - add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)

# query S&P Global historical prices
benchmark_ticker=benchmark_ticker.history(period="1d",start=start_date,end=end_date)

# print(ticker_df.head())
####
#st.write('---')
# st.write(ticker_data.info)


#-----------------------------------------------#

# Time Series Forecasting with Facebook Prophet

# Display Prophet section
prophet_check_box=st.checkbox(label=f"Display {ticker} Prophet time series forecast data")
if prophet_check_box:
    with st.container():
            # 2 columns section:
            col1, col2 = st.columns([3, 2])
            with col1:           
                st.write("###") 
                st.write("###")
                # input a streamlit slider with years of prediction values
                n_years=st.slider("Select year(s) for time series forecast",1,5)
                
    
                # create a new dataframe from the ticker_df object
                df_plot = pd.DataFrame.from_dict(ticker_df, orient='columns')

                # select the 'Close' column
                df_plot = df_plot[['Close']]

                # rename the column to 'y'
                df_plot.columns = ['y']

                # add a 'ds' column with the dates, converting it to a datetime object and setting the timezone to None
                df_plot['ds'] = pd.to_datetime(df_plot.index).tz_localize(None)

                # Prophet requires a specific column format for the dataframe
                df_plot = df_plot[['ds', 'y']]


                # create the Prophet model and fit it to the data
                model = Prophet(daily_seasonality=True)
                model.fit(df_plot)

                # create a dataframe with future dates
                future_dates = model.make_future_dataframe(periods=365)

                # make predictions for the future dates
                forecast = model.predict(future_dates)

                # select the relevant columns for the plot
                plot_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                # Display data table
                forecast_data_check_box=st.checkbox(label=f"Display {ticker} forecast data & price prediction")
                if forecast_data_check_box:
                    st.subheader(f"{ticker} forecast dataset")
                    # Show tail of the Forecast data
                    st.write(forecast.tail())
                    st.write("---")

                    # create a plotly figure
                    fig = go.Figure()

                    # add the predicted values to the figure
                    fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat'], name='Prediction'))

                    # add the uncertainty intervals to the figure
                    fig.add_shape(
                            type='rect',
                            xref='x',
                            yref='paper',
                            x0=plot_df['ds'].min(),
                            y0=0,
                            x1=plot_df['ds'].max(),
                            y1=1,
                            fillcolor='#E8E8E8',
                            layer='below',
                            line_width=0
                        )
                    fig.add_shape(
                            type='rect',
                            xref='x',
                            yref='y',
                            x0=plot_df['ds'].min(),
                            y0=plot_df['yhat_upper'],
                            x1=plot_df['ds'].max(),
                            y1=plot_df['yhat_lower'],
                            fillcolor='#E8E8E8',
                            layer='below',
                            line_width=0
                        )

                    # add the actual values to the figure
                    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['y'], name='Actual'))

                    # set the plot's title and labels
                    fig.update_layout(
                        title=f"{ticker} stock price prediction",
                        xaxis_title='Date',
                        yaxis_title='Price (USD)'
                    )

                    # show the prediction plot
                    st.plotly_chart(fig)
    
                    # Display Prophet tools & components
                    forecast_component_check_box=st.checkbox(label=f"Display {ticker} Prophet forecast components")
                    if forecast_component_check_box:

                        # create a plotly figure for the model's components
                        st.subheader(f"{ticker} plot widget")
                        fig2 = plot_plotly(model, forecast)
                        # show the plot
                        st.plotly_chart(fig2)


                        # show the model's plots
                        st.subheader(f"{ticker} forecast components")
                        st.write(model.plot(forecast))

                        # show the model's plot_components
                        st.write(model.plot_components(forecast))

st.write("---")                        

#-----------------------------------------------#

# Time Series Forecasting with XGBoost

xgboost_check_box = st.checkbox(f"Display {ticker} XGBoost time series forecast data")
if xgboost_check_box:
    # Input a streamlit slider to select number of years to forecast
    n_years = st.slider("Select year(s) for time series forecast", 1, 5)
    st.write("---")

    # Create a new dataframe from the ticker_df object
    df_plot = pd.DataFrame(ticker_df["Close"])
    df_plot.reset_index(inplace=True)
    df_plot.rename(columns={"Date":"ds", "Close":"y"}, inplace=True)
    # XGBoost model expects the input data to be numeric but column is datetime - convert to numeric value
    df_plot['ds'] = pd.to_numeric(df_plot['ds'], errors='coerce')


    # Split data into training and testing sets using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(df_plot):
        X_train, X_test = df_plot.iloc[train_index], df_plot.iloc[test_index]

    # Fit XGBoost model
    xgb = XGBRegressor()
    xgb.fit(X_train[["ds"]], X_train["y"])

    # Make predictions