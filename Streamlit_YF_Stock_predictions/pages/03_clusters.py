# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
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
st.set_page_config(page_title="Streamlit ticker(s) forecast app",page_icon="📶",layout="centered")


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
tickers = pd.read_csv("./Resources/ticker_symbols.csv")

# Display a selectbox for the user to choose a ticker
ticker_symbol = st.sidebar.selectbox("Select a ticker from the dropdown menu", tickers)

# Get data for the selected ticker
ticker_data = yf.Ticker(ticker_symbol)

# main content - wrap inside streamlit container
# with st.container:
# add title
st.title(f"K-Means Clustering for S&P 500 Stocks")
st.write("---")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("2012-01-01"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)
# print(ticker_df.head())
####
#st.write('---')
# st.write(ticker_data.info)


# Add cache to store ticker values after first time download in browser
@st.cache

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
raw_data_check_box=st.checkbox(label="Display raw dataset")
if raw_data_check_box:
    st.subheader(f"{ticker_symbol} raw data")
    st.write(data)

index_data=data.index
st.write(index_data)

import streamlit as st
import yfinance as yf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

st.title("")

# Get ticker data for all S&P 500 stocks
sp500_tickers = yf.Tickers('^GSPC')

# Fetch stock data
stock_data = {}
for ticker in sp500_tickers.tickers:
    stock_data[ticker.info['shortName']] = ticker.history(period="max")

# Get number of clusters from user
k = st.number_input("Enter number of clusters:")

# Perform k-means clustering on close price data
close_prices = pd.DataFrame()
for ticker, data in stock_data.items():
    close_prices[ticker] = data['Close']

kmeans = KMeans(n_clusters=k).fit(close_prices)

# Get cluster labels
labels = kmeans.labels_

# Create a scatter plot with the tickers and cluster labels
plt.scatter(range(len(close_prices)), [1]*len(close_prices), c=labels, cmap='viridis')
plt.xticks(range(len(close_prices)), close_prices.columns, rotation=90)
plt.xlabel('Ticker')
plt.ylabel('Cluster Label')
plt.title('K-Means Clustering for S&P 500 Stocks')

# Display the plot
st.pyplot()


    