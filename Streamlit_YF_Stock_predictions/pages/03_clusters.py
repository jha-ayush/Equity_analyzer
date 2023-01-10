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
import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Streamlit ticker(s) forecast app",page_icon="📶",layout="centered")


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

# include params & data
# Read ticker symbols from a CSV file
tickers = pd.read_csv("./Resources/s&p500_tickers_2022.csv")


# K-means clustering + PCA dimensional reduction - sklearn


st.title("S&P500 Clustering App")

# Get the list of tickers from user input
st.write("Select the stocks for which you want to perform clustering (S&P 500 is selected by default)")
ticker = st.multiselect("Select ticker(s)", ['^GSPC','AAPL','AMZN','GOOGL','MSFT','GE','PG','BA','CVX','WMT','BABA','SPY'], default = ['^GSPC'])

st.write("Select the date range for which you want to download data")
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("1997-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))

# Get the number of clusters
n_clusters = st.slider("Number of clusters", 2, 10, 4)

if ticker and start_date and end_date:
    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write("Data Loaded")

    # Selecting columns to be used for PCA
    data = data.loc[:, ('Adj Close', slice(None))]
    # Convert the data to a numpy array
    X = data.to_numpy()
    st.write("Data Pre-Processed")

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    st.write("Data Dimensionality Reduced")

    # Fit the model to the data and predict the clusters
    kmeans = KMeans(n_clusters=n_clusters)
    predictions = kmeans.fit_predict(X_pca)
    st.write("Data Clustered")

    # Visualize the results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    st.pyplot()
else:
    st.warning("Please select the stocks and provide the date range")
