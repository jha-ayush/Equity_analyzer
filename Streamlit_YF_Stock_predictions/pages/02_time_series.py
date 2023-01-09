# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import date
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
import datetime

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="S&P500 ticker(s) analysis",page_icon="📈",layout="centered")


# Add cache to store ticker values after first time download in browser
@st.cache

# functions

# Create a function to access the json data of the Lottie animation using requests - if successful return 200 - data is good, show animation else return none
def load_lottieurl(url):
    """
    Loads the json data for a Lottie animation using the given URL.
    Returns None if there was an error.
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
#load lottie asset
lottie_coding=load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vktpsg4v.json")

# Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")         

# include params & data
# Read ticker symbols from a CSV file
tickers = pd.read_csv("./Resources/s&p500_tickers_2022.csv")

# Display a selectbox for the user to choose a ticker
ticker_symbol = st.sidebar.selectbox("Select a ticker from the dropdown menu", tickers)

# Get data for the selected ticker
ticker_data = yf.Ticker(ticker_symbol)

# wrap header content in a streamlit container
with st.container():
        # 2 columns section:
        col1, col2 = st.columns([3, 2])
        with col1:           
            # Load title/info
            st.header(f"Time Series forecast")
            st.text(f"With Facebook Prophet")
        with col2:
            # Load asset(s)
            # st_lottie(lottie_coding,height=150,key="finance")
            st.empty()
st.write("---")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("2007-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)
# print(ticker_df.head())
####
#st.write('---')
# st.write(ticker_data.info)

# add a streamlit multicheck box to include feature
# feature_section=st.sidebar.multiselect(label="Features to plot",options=ticker_df)

# Display data table
show_info_check_box=st.checkbox(label=f"Display {ticker_symbol} company info")
if show_info_check_box:
    # ticker information - logo
    ticker_logo="<img src=%s>" % ticker_data.info["logo_url"]
    st.markdown(ticker_logo,unsafe_allow_html=True)

    # ticker information - name
    ticker_name=ticker_data.info["longName"]
    st.header(f"{ticker_name}")

    # ticker information - symbol + sector
    ticker_symbol=ticker_data.info["symbol"]
    ticker_sector=ticker_data.info["sector"]
    st.text(f"{ticker_symbol} is part of the {ticker_sector} sector")

    # ticker information - summary
    ticker_summary=ticker_data.info["longBusinessSummary"]
    st.info(f"{ticker_summary}")

    # Bollinger bands - trendlines plotted between two standard deviations
    st.header(f"{ticker_symbol} Bollinger bands")
    qf=cf.QuantFig(ticker_df,title='First Quant Figure',legend='top',name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

# input a streamlit slider with years of prediction values
n_years=st.sidebar.slider("Select year(s) for time series forecast",1,5)

# Define a period
period=n_years*365


# Load stock data - define functions
def load_data(ticker):
    data=yf.download(ticker,start_date,end_date)
    # data.set_index("Date",inplace=True,append=True,drop=True)
    data.reset_index(inplace=True)
    return data

# data load complete message
data_load_state=st.sidebar.text("Loading data...⌛")  
data=load_data(ticker_symbol)
data_load_state.text("Data loading complete ✅")

# Display data table
raw_data_check_box=st.checkbox(label=f"Display {ticker_symbol} metrics")
if raw_data_check_box:
    # st.subheader(f"{ticker_symbol} raw data")
    # st.write(data)

    # Plot Open vs Close price data
    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))
        fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="stock_close"))
        fig.layout.update(title_text=(f"{ticker_symbol} raw data plot - Open vs Close price"),xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()

    # create a streamlit linechart for ticker Volume over time
    st.subheader(f"{ticker_symbol} trading volume over time")
    st.line_chart(ticker_df.Volume)

st.write("---")
st.subheader(f"{ticker_symbol} time series forecast data")
# Forecasting with Prophet

df_train=data[["Date","Close"]]
# Adjust for Prophet's x-axis [ds] & y-axis [y]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

# Create a facebook prophet Time Series model
m=Prophet(daily_seasonality=True)

# Fit model with training dataframe
m.fit(df_train)

# Create future forecast dataframe
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

# Display data table
forecast_data_check_box=st.checkbox(label=f"Display {ticker_symbol} forecast data")
if forecast_data_check_box:
    st.subheader(f"{ticker_symbol} forecast dataset")
    # Show tail of the Forecast data
    st.write(forecast.tail())

    # Plot Forecast data using plotly
    st.subheader(f"{ticker_symbol} forecast plot")
    fig1=plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.write("---")

    # Plot Forecast components
    st.subheader(f"{ticker_symbol} forecast components")
    fig2=m.plot_components(forecast)
    st.write(fig2)
    st.write("---")