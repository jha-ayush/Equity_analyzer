import streamlit as st
import yfinance as yf # yfinance library for downloading stock data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA # PCA for dimensionality reduction
from sklearn.cluster import KMeans # KMeans for clustering
from sklearn.cluster import DBSCAN # DBSCAN for clustering
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding # Other dimensionality reduction techniques
from sklearn.preprocessing import StandardScaler # Scaling data
from sklearn.metrics import silhouette_score # To evaluate the clustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.utils import resample

import matplotlib.pyplot as plt
import logging
import os

# Set page configurations
st.set_page_config(page_title="Stocks analysis", page_icon="📶", layout="centered",initial_sidebar_state="auto")

# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)


# Use local CSS file
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        logging.error('Cannot find the CSS file')

# Load CSS file
local_css("./style/style.css")

# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/tickers.csv")
except:
    logging.error('Cannot find the CSV file')
    
st.title(f"Clustering Model(s) evaluations")
st.write("---")

# Get the list of tickers from user input
# Provide a default benchmark ticker
benchmark_ticker = '^GSPC'
if benchmark_ticker in tickers.ticker.tolist():
    ticker = st.sidebar.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist(), default = [benchmark_ticker])
else:
    ticker = st.sidebar.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist())
    
# Get the date range from user input
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("1997-1-1"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))  

# Get the number of clusters from user input
# n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

# Get the data that the user wants to use for clustering
clustering_data = st.sidebar.multiselect("Select data to use for clustering", ['Open', 'High', 'Low','Close','Adj Close','Volume'], default = ['Adj Close'])

# Dimensionality reduction technique
reduction_method = st.sidebar.selectbox("Select dimensionality reduction technique", ['PCA','t-SNE','MDS','Isomap','LLE','SE'])

# Get the data and scale it
if ticker and start_date and end_date:
    with st.container():
            # 2 columns section:
            col1, col2 = st.columns([4, 1])
            with col1:
                if len(ticker) < 2:
                    st.error("Please select at least 2 tickers for analysis.")
                else:
                    show_plot_check_box=st.checkbox(label=f"Display clustering plot for {ticker}")
                    if show_plot_check_box:
                        
                        # Download the data
                        data = yf.download(ticker, start=start_date,end=end_date)
                        # Drop any missing values
                        data.dropna(inplace=True)
                        st.text(f"Data Loaded for {ticker} ✅")
                        
                        # Selecting columns to be used for clustering
                        data = data[clustering_data]
                        
                        
                        # Resample the data
                        data_resampled = resample(data, n_samples=len(data), random_state=1)
                        
                        # Scale the data
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data_resampled)
                        st.text("Data Scaled ✅")

        
                        # Monte Carlo simulation
                        num_simulations = 10 # Default number of MC simulations
                    
                        for i in range(num_simulations):
                            # Dimensionality reduction
                            if reduction_method == 'PCA':
                                pca = PCA(n_components=2)
                                data = pca.fit_transform(data_scaled)
                                # Get explained variance ratio
                                explained_variance_ratio = pca.explained_variance_ratio_
                                # Calculate confidence percent
                                x_conf = explained_variance_ratio[0]*100
                                y_conf = explained_variance_ratio[1]*100
                            else:
                                st.error("Invalid reduction method")
        
        
                        # add a select box for the user to choose the clustering algorithm
                        # Choose Clustering Algorithm
                        clustering_algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])
                        
                        if clustering_algorithm == "K-Means":
                            n_clusters = st.number_input("Enter number of clusters (default=3): ", min_value=3)
                            
                            if n_clusters < 2:
                                st.error("Please enter a valid number of clusters (minimum 2)")
                            else:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(data_scaled)
                                st.text("K-means Clustering Completed ✅")
                                # print refactored dataframe
                                st.write(data)
                                
                                # Elbow Method
                                wcss = []
                                for i in range(1, 11):
                                    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=1)
                                    kmeans.fit(data)
                                    wcss.append(kmeans.inertia_)
                                plt.plot(range(1, 11), wcss)
                                plt.title('Elbow Method')
                                plt.xlabel('Number of clusters')
                                plt.ylabel('WCSS')
                                st.pyplot()
                                st.write(f'The optimal number of clusters is the one that corresponds to the "elbow" point in the plot (default =3).',unsafe_allow_html=True)
                                
                        elif clustering_algorithm == "DBSCAN":
                            eps = st.number_input("Enter value of epsilon: ", min_value=0.0001)
                            if eps <= 0:
                                st.error("Please enter a valid value of epsilon (greater than 0)")
                            else:
                                min_samples = st.number_input("Enter value of min_samples: ", min_value=1)
                                dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
                                n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
                            
                            # Get cluster labels for each data point
                            labels = kmeans.labels_ if clustering_algorithm == "K-Means" else dbscan.labels_
                            
                            

                            # Plot the data
                            plt.scatter(data_scaled[:,0], data_scaled[:,1], c=labels, cmap='rainbow')
                            plt.title(f'{clustering_algorithm} Clustering of {clustering_data} features with {n_clusters} clusters for {ticker}')
                            plt.legend(title='Clusters')
                            st.pyplot()
                            
                            
                            
                        # Dimensionality reduction
                        if reduction_method == 'PCA':
                            pca = PCA(n_components=2)
                            data = pca.fit_transform(data_scaled)
                            # Get explained variance ratio
                            explained_variance_ratio = pca.explained_variance_ratio_
                            # Calculate confidence percent
                            x_conf = explained_variance_ratio[0]*100
                            y_conf = explained_variance_ratio[1]*100
                        elif reduction_method == 't-SNE':
                            tsne = TSNE(n_components=2)
                            data = tsne.fit_transform(data_scaled)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'MDS':
                            mds = MDS(n_components=2)
                            data = mds.fit_transform(data_scaled)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'Isomap':
                            iso = Isomap(n_components=2)
                            data = iso.fit_transform(data_scaled)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'LLE':
                            lle = LocallyLinearEmbedding(n_components=2)
                            data = lle.fit_transform(data_scaled)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'SE':
                            se = SpectralEmbedding(n_components=2)
                            data = se.fit_transform(data_scaled)
                            x_conf, y_conf = None, None
                        else:
                            st.error("Invalid reduction method")


                        # Run K-means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(data)
                        

                
                        # Get cluster labels for each data point
                        labels = kmeans.labels_

                        # Plot the data
                        plt.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap='rainbow')
                        plt.title(f'{reduction_method} reduction of {clustering_data} features with {n_clusters} clusters for {ticker}')
                        plt.xlabel(f'PCA1 of {clustering_data} ({x_conf:.2f}% confidence)' if x_conf is not None else 'First Principal Component')
                        plt.ylabel(f'PCA2 of {clustering_data} ({y_conf:.2f}% confidence)' if y_conf is not None else 'Second Principal Component')
                        plt.legend(title='Clusters')
                        st.pyplot()                        

                        
                    #Cluster metrics information
                    st.write("---")
                    show_cluster_metrics_check_box = st.checkbox(label=f"Display cluster data metrics")
                    if show_cluster_metrics_check_box:
                        clusters = [[] for _ in range(kmeans.n_clusters)]
                        for i, label in enumerate(labels):
                            clusters[label].append(data[i])

                        # Use Streamlit to display the clusters
                        for i, cluster in enumerate(clusters):
                            if i < (n_clusters):
                                st.markdown(f'<b>Cluster {i}</b>', unsafe_allow_html=True)
                                # Check for missing values and remove them
                                cluster = [c for c in cluster if not np.isnan(c).any()]

                                # Add a summary of the cluster
                                st.write(f"Mean value of cluster {i} is:", (np.mean(cluster)))
                                st.write(f"Median value of cluster {i} is:", (np.median(cluster)))
                                st.write(f"Variance value of cluster {i} is:", (np.var(cluster)))
                                

        
                    # Evaluation metrics
                    st.write("---")
                    show_evaluation_metrics_check_box=st.checkbox(label=f"Display evaluation metrics")
                    if show_evaluation_metrics_check_box:

                        # Evaluation Metrics
                        st.write("Silhouette score:", silhouette_score(data, labels.ravel()))
                        st.write("Calinski Harabasz score:", calinski_harabasz_score(data, labels.ravel()))
                        st.write("Davies Bouldin score:", davies_bouldin_score(data, labels.ravel()))

                        
                    #Cluster metrics information
                    st.write("---")
                    save_result_check_box=st.checkbox(label=f"Display save options")
                    if save_result_check_box:
                            
                        # Provide a summary of the results
                        if st.button('Summary'):
                            st.write("Add the code to provide a summary of the results.")    

                        # Compare the results of different dimensionality reduction techniques and clustering algorithms
                        if st.button('Compare Results'):
                            st.write("Add the code to compare the results of different dimensionality reduction techniques and clustering algorithms.")
                            
                        # Save the results to a file or a database
                        if st.button("Save Results"):
                            export_file = st.file_uploader("Choose a CSV file", type=["csv"])
                            if export_file is not None:
                                with open(export_file, "w") as f:
                                    writer = csv.writer(f)
                                    writer.writerows(clusters)
                                st.balloons("File exported successfully")