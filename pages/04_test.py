import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="S&P Global ticker(s) analysis",page_icon="📈",layout="centered",initial_sidebar_state="auto")


# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)

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
tickers = pd.read_csv("./Resources/tickers.csv")

# Benchmark ticker - S&P Global index 'SPGI'
benchmark_ticker=yf.Ticker("SPGI")

# Display a selectbox for the user to choose a ticker
ticker = st.sidebar.selectbox("Select a ticker from the dropdown menu",tickers)

# Get data for the selected ticker
ticker_data = yf.Ticker(ticker)

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("1997-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))

# Create a new dataframe - add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)

# Create a new dataframe - query S&P Global historical prices
benchmark_ticker=benchmark_ticker.history(period="1d",start=start_date,end=end_date)

# Function to download stock data using yfinance
def get_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start_date, end_date)
        return stock_data
    except ValueError:
        st.error("Invalid Ticker Symbol")
        return None

# Function to perform hierarchical clustering on stock data
def hierarchical_clustering(data, n_clusters, linkage):
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        model.fit(data)
        return model.labels_
    except ValueError:
        st.error("Invalid number of clusters")
        return None

# Streamlit app
def main():
    st.title("Stock Clustering App")
    # ticker = st.text_input("Enter a Ticker Symbol:")
    # Get the number of clusters from user input
    n_clusters = st.sidebar.slider("Select the number of clusters:", min_value=1, max_value=10, value=4)
    linkage = st.selectbox('Select linkage method:', ['ward', 'single', 'complete', 'average'])
    if ticker and n_clusters:
        stock_data = get_stock_data(ticker)
        if stock_data is not None:
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
            # Perform hierarchical clustering
            labels = hierarchical_clustering(scaled_data, n_clusters, linkage)
            if labels is not None:
                st.write("Cluster labels:", labels)
                # Evaluation Metrics
                silhouette = silhouette_score(scaled_data, labels)
                db = davies_bouldin_score(scaled_data, labels)
                st.write("Silhouette score:", silhouette)
                st.write("Davies-Bouldin index:", db)
                # Visualize the results or further process the data with the labels

if __name__ == '__main__':
    main()
