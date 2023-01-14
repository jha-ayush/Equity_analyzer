import streamlit as st
import yfinance as yf # yfinance library for downloading stock data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA # PCA for dimensionality reduction
from sklearn.cluster import KMeans # KMeans for clustering
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding # Other dimensionality reduction techniques
from sklearn.preprocessing import StandardScaler # Scaling data
from sklearn.metrics import silhouette_score # To evaluate the clustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import metrics

import matplotlib.pyplot as plt
import logging
import os

# Set page configurations
st.set_page_config(page_title="Stocks analysis", page_icon="📶", layout="centered",initial_sidebar_state="auto")

# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)

# Disable warning by disabling the config option
# @st.set_option('deprecation.showPyplotGlobalUse', False)

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
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)

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
                        st.text("Data Loaded ✅")
                        # Selecting columns to be used for clustering
                        data = data.loc[:, clustering_data]

                        # Scale the data
                        scaler = StandardScaler()
                        data = scaler.fit_transform(data)
                        st.text("Data Scaled ✅")


                        # Dimensionality reduction
                        if reduction_method == 'PCA':
                            pca = PCA(n_components=2)
                            data = pca.fit_transform(data)
                            # Get explained variance ratio
                            explained_variance_ratio = pca.explained_variance_ratio_
                            # Calculate confidence percent
                            x_conf = explained_variance_ratio[0]*100
                            y_conf = explained_variance_ratio[1]*100
                        elif reduction_method == 't-SNE':
                            tsne = TSNE(n_components=2)
                            data = tsne.fit_transform(data)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'MDS':
                            mds = MDS(n_components=2)
                            data = mds.fit_transform(data)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'Isomap':
                            iso = Isomap(n_components=2)
                            data = iso.fit_transform(data)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'LLE':
                            lle = LocallyLinearEmbedding(n_components=2)
                            data = lle.fit_transform(data)
                            x_conf, y_conf = None, None
                        elif reduction_method == 'SE':
                            se = SpectralEmbedding(n_components=2)
                            data = se.fit_transform(data)
                            x_conf, y_conf = None, None
                        else:
                            st.error("Invalid reduction method")


                        # Run K-means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(data)
                        st.text("K-means Clustering Completed ✅")


                        # Get cluster labels for each data point
                        labels = kmeans.labels_

                        # Plot the data
                        plt.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap='rainbow')
                        plt.title(f'{reduction_method} reduction of {clustering_data} features with {n_clusters} clusters for {ticker}')
                        plt.xlabel(f'PCA1 of {clustering_data} ({x_conf:.2f}% confidence)' if x_conf is not None else 'First Principal Component')
                        plt.ylabel(f'PCA2 of {clustering_data} ({y_conf:.2f}% confidence)' if y_conf is not None else 'Second Principal Component')
                        plt.legend(title='Clusters')
                        st.pyplot()

                    #write the final dataframe 
                    # st.write(data)

                    #line chart of the selected ticker
                    # st.line_chart(data[clustering_data])
                    # st.line_chart(data)
                    
                    #Number of clusters optimization methods
                    st.write("---")
                    show_cluster_opt_check_box=st.checkbox(label=f"Display cluster count optimization methods")
                    if show_cluster_opt_check_box:
                    
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
                        st.write(f'The optimal number of clusters is the one that corresponds to the "elbow" point in the plot',unsafe_allow_html=True)

                        # Silhouette Analysis
                        silhouette_scores = []
                        for i in range(2, 11):
                            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=1)
                            preds = kmeans.fit_predict(data)
                            silhouette_scores.append(metrics.silhouette_score(data, preds))
                        plt.plot(range(2, 11), silhouette_scores)
                        plt.title('Silhouette Analysis')
                        plt.xlabel('Number of clusters')
                        plt.ylabel('Silhouette Score')
                        st.pyplot()
                        st.write(f'The highest point on the y-axis represents the optimal number of clusters for the given x-axis: <b>{(metrics.silhouette_score(data, preds)):0.4f}</b>',unsafe_allow_html=True)


                    #Cluster metrics information
                    st.write("---")
                    show_cluster_metrics_check_box=st.checkbox(label=f"Display cluster data metrics")
                    if show_cluster_metrics_check_box:
                        # Group data points into clusters
                        clusters = [[] for _ in range(kmeans.n_clusters)]
                        for i, label in enumerate(labels):
                            clusters[label].append(data[i])

                        # Use Streamlit to display the clusters
                        for i, cluster in enumerate(clusters):
                            if i < (n_clusters +1):
                                st.markdown(f'<b>Cluster {i}</b>',unsafe_allow_html=True)
                                # Check for missing values and remove them
                                cluster = [c for c in cluster if not np.isnan(c).any()]

                                # Add a summary of the cluster
                                st.write(f"Mean value of cluster {i} is:",np.mean(cluster))
                                st.write(f"Median value of cluster {i} is:",np.median(cluster))
                                st.write(f"Variance value of cluster {i} is:",np.var(cluster))
        

                    st.write("---")
                    show_evaluation_metrics_check_box=st.checkbox(label=f"Display evaluation metrics")
                    if show_evaluation_metrics_check_box:

                        # Evaluation Metrics
                        st.write("Silhouette score:", silhouette_score(data, labels.ravel()))
                        st.write("Calinski Harabasz score:", calinski_harabasz_score(data, labels.ravel()))
                        st.write("Davies Bouldin score:", davies_bouldin_score(data, labels.ravel()))
                        # print("test1")
                        # st.write("Adjusted Rand score:", adjusted_rand_score(data, labels.ravel()))
                        # print("test2")
                        # st.write("Adjusted Mutual Info score:", adjusted_mutual_info_score(data, labels.ravel()))
                        # print("test3")

                        
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