# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set page configurations
st.set_page_config(page_title="Streamlit Ticker(s) Clustering App", page_icon="📶", layout="centered",initial_sidebar_state="auto")

# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/s&p_global_tickers_2022.csv")
except:
    logging.error('Cannot find the CSV file')

# Use local CSS file
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        logging.error('Cannot find the CSS file')

# Load CSS file
local_css("./style/style.css")

st.title(f"PCA/K-means clustering app")
st.write("---")

# Get the list of tickers from user input
benchmark_ticker = ['SPGI']
if benchmark_ticker[0] in tickers.ticker.isin(benchmark_ticker):
    ticker = st.sidebar.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist(), default = benchmark_ticker)
else:
    ticker = st.sidebar.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist())


# Get the date range from user input
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("1997-1-1"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

# Get the number of clusters from user input
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)

# Check if all necessary inputs are provided
if ticker and start_date and end_date:
    with st.container():
            # 2 columns section:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"Plot for {ticker}")
                # Download the data
                data = yf.download(ticker, start=start_date, end=end_date)

                # Drop any missing values
                data.dropna(inplace=True)
                st.text("Data Loaded ✅")

                # Selecting columns to be used for PCA
                data = data.loc[:, ('Adj Close')]

                # Convert the data to a numpy array
                X = data.to_numpy()
                st.text("Applying PCA...⌛")

                # Perform PCA to reduce dimensionality
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                st.text("Data Dimensionality Reduced ✅")

                # Fit the model to the data and predict the clusters
                kmeans = KMeans(n_clusters=n_clusters)
                predictions = kmeans.fit_predict(X_pca)
                st.text("Data Clustering complete ✅")

                # Visualize the results
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis')
                plt.xlabel(f'PCA Dimension 1 ({pca.explained_variance_ratio_[0]:0.2f}%)')
                plt.ylabel(f'PCA Dimension 2 ({pca.explained_variance_ratio_[1]:0.2f}%)')
                plt.title(f"Clustering of Ticker(s) {ticker}")
                st.pyplot()
