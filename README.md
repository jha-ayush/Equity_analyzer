# Stocks analyzer & predictor data app

- A user is able to visualize historical stock data along with Financial metrics.
- A user is able to utilize the web app to verify whether selected stock(s) are a good recommendation to buy, sell or hold for a given period of time

We first resample the imbalanced stock data and apply K-means clustering and optimization (Monte Carlo simulation) to pull information about various metrics. Using unsupervised learning we can identify and figure out the behaviors of the selected stock(s).
By further implementing on the clustered stocks from Unsupervised Learning, we can layer Supervised Learning using Random Forest, SVM & KNN alogrithms to predict whether the stock will go up or down in the future. We can further measure which Supervised model provides the most optimization for the selected stocks.

Further analysis and models can be tested to optimize what is/are the best algorithm(s) for a given user preference
- Do I want to buy/sell?
- Do I want to hold a portfolio stock for X time period?
- How can I allocate pricing around the portfolio stocks with a given amount money?




## Setup
Create new environment in Terminal
- `conda create -n streamlit python=3.9`
- `conda activate streamlit`
- `pip install numpy pandas streamlit streamlit_lottie prophet cufflinks yfinance datetime watermark warnings sklearn plotly`
- OR `pip install requirements.txt` to install all package dependancies


## Run in localhost
- cd into the app folder where `main.py` sits
- Run `streamlit run main.py` in Terminal to start up server in `localhost`


## Technologies
<sub>Python implementation: CPython</sub>

<sub>Python version       : 3.7.13</sub>

<sub>IPython version      : 7.34.0</sub>


<sub>Compiler    : Clang 12.0.0</sub> 

<sub>OS          : Darwin</sub>

<sub>Release     : 22.2.0</sub>

<sub>Machine     : x86_64</sub>

<sub>Processor   : i386</sub>

<sub>CPU cores   : 8</sub>

<sub>Architecture: 64bit</sub>


<sub>plotly   : 5.11.0</sub>

<sub>requests : 2.28.1</sub>

<sub>streamlit: 1.16.0</sub>

<sub>cufflinks: 0.17.3</sub>

<sub>numpy    : 1.21.6</sub>

<sub>pandas   : 1.3.5</sub>

<sub>yfinance : 0.2.3</sub>



## Method
- Install new env as described above
- Install `yfinance`, `streamlit` and other required packages as described above
- Download & explore historical data from Company's `info` key
- Data setup that predict future prices using historical prices
- Test different Machine Learning models to find the optimal model for user's portfolio strategy
- Improve accuracy


## Download data
- Download stock ticker data from `yfinance`
- Review data structure & info
- Data cleanup by removing "Dividends" & "Stock splits" columns
- Drop na values

Available columns are:
- `Open` - the price the stock opened at
- `High` - the highest price during the day
- `Low` - the lowest price during the day
- `Close` - the closing price on the trading day
- `Volume` - how many shares were traded

Use **252 days** for a one year period



## Machine Learning models

### Unsupervised Clustering
We initially pick at least 2 tickers for data clustering (default ticker in the dropdown menu - S&P 500 `^GSPC` ticker). By grouping the ticker data a user can better understand how the stocks move in conjunction over time. To further improve clustering, we apply the "**Elbow Method**" & "**Silhouette Score**" to find the optimum number of clusters (default is set to **3**).
Furthermore, we used **Monte Carlo** simulation on resampled data to estimate the probability of different outcomes. By combining the results of **K-means** clustering with Monte Carlo simulation, a user can gain valuable insights into the behavior of a specific stock or portfolio of stocks, and use that information to make more informed investment decisions, although this model cannot predict the future with certainty.
**DBSCAN** was additionally used along with K-means.


### Supervised Classifiers
### Algorithmic Trading
### Time Series Analysis


## Next Steps

### Technique
- Figure out way(s) to calcuate amount of money a user can generate if trading with any of the models used in the app
- Streamlit cloud deployment
- Add the ability to perform the clustering on a subset of the data: The script currently performs the clustering on all the data. Adding the ability to perform the clustering on a subset of the data, such as the last 30 days
- Add the ability to compare the results of different clustering techniques/parameters: The script currently only shows the results of a single clustering run. Adding the ability to compare the results of different clustering techniques/parameters, such as different number of clusters or different clustering algorithms, would increase the usefulness of the script.

### Improve algorithm(s)
- Additional prediction algorthims can be trained
- App can include Neural Networks, Roboadvisors for a better UI interactivity
- Constrain data to a shorter time frame


