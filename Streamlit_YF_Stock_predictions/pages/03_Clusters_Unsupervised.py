# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set page configurations
st.set_page_config(page_title="Streamlit Ticker(s) Clustering App", page_icon="📶", layout="centered")

# Read ticker symbols from a CSV file
tickers = pd.read_csv("./Resources/s&p500_tickers_2022.csv")

# Use local CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS file
local_css("./style/style.css")

# Include title and introductory text
st.title("S&P500 Clustering App")
st.write("-------")

# Get the list of tickers from user input
ticker = st.sidebar.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist())

# Get the date range from user input
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("1997-1-1"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

# Get the number of clusters from user input
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)

# Check if all necessary inputs are provided
if ticker and start_date and end_date:
    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Drop any missing values
    data.dropna(inplace=True)
    st.write("Data Loaded")

    # Selecting columns to be used for PCA
    data = data.loc[:, ('Adj Close')]

    # Convert the data to a numpy array
    X = data.to_numpy()
    st.write("Applying PCA...")

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    st.write("Data Dimensionality Reduced")

    # Fit the model to the data and predict the clusters
    kmeans = KMeans(n_clusters=n_clusters)
    predictions = kmeans.fit_predict(X_pca)
    st.write("Data Clustered")

    # Copy code
    # Visualize the results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    st.pyplot()