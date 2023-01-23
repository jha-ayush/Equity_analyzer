# Equity portfolio investment advisor

[Synthesia AI excerpt](https://share.synthesia.io/929ff00e-fa8e-418d-b67d-756b99c2bc1e)

[Streamlit deployment](https://jha-ayush-equity-analyzer-home-w7xb3c.streamlit.app/) - *WIP*

Utilizing financial ratio metrics and state-of-the-art machine learning techniques such as unsupervised and supervised learning, the Analyzer provides users with valuable insights into the stock market, allowing for informed decision making in regards to buying, selling, or holding stocks.

The user interface of the Equity Analyzer is intuitive and user-friendly, making it accessible for individuals of all levels of financial expertise. Its advanced features allow for the identification of profitable investments and the avoidance of costly mistakes.

The business case for the Equity Analyzer is clear, as it empowers investors of all experience levels to make informed decisions regarding their stock portfolio.


## Setup
Create new environment in Terminal
- `conda create -n streamlit python=3.9`
- `conda activate streamlit`
- `pip install numpy pandas streamlit streamlit_lottie prophet cufflinks yfinance datetime watermark warnings sklearn plotly`
- OR `pip install requirements.txt` to install all package dependancies


## Run in localhost
- cd into the app folder where `Home.py` sits
- Run `streamlit run Home.py` in Terminal to start up server in `localhost`


## Technologies
Library versions:

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
- Map ticker csv to yfinance historical ticker data
- Define start & end dates
- Data cleanup by removing "Dividends" & "Stock splits" columns
- Drop na values
- Use **252 days** for a one year period




## Machine Learning models

### Unsupervised Clustering
After gathering ticker information along with their sector & market cap information, we decided to first get the count of each companies in the various sectors. We identified the top 10 companies from each sector via market cap & daily returns calculations. After that, we moved ahead with grouping these companies into clusters using K-means and Silhouette scoring. Once the tickers were clustered, the data is passed through Monte Carlo simulation to better normalize the data.


### Supervised Classifiers

### Time Series Analysis - Facebook Prophet analysis for baseline ticker prediction


## Next Steps

### Technique
- Figure out way(s) to calcuate amount of money a user can generate if trading with any of the models used in the app
- Streamlit cloud deployment
- Add the ability to perform the clustering on a subset of the data: The script currently performs the clustering on all the data. Adding the ability to perform the clustering on a subset of the data, such as the last 30 days
- Add the ability to compare the results of different clustering techniques/parameters: The script currently only shows the results of a single clustering run. Adding the ability to compare the results of different clustering techniques/parameters, such as different number of clusters or different clustering algorithms, would increase the usefulness of the script.
- Include more error handling

### Improve algorithm(s)
- Additional prediction algorthims can be trained
- App can include Neural Networks, Roboadvisors for a better UI interactivity
- Constrain data to a shorter time frame


