# Stocks analyzer & predictor data app


Financial ratios, visualizations, Time series forecasting, Machine Learning (Supervised & Unsupervised), Algorithmic trading

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
- Python implementation: CPython
- Python version       : 3.7.13
- IPython version      : 7.34.0

- Compiler    : Clang 12.0.0 
- OS          : Darwin
- Release     : 22.2.0
- Machine     : x86_64
- Processor   : i386
- CPU cores   : 8
- Architecture: 64bit

- plotly   : 5.11.0
- requests : 2.28.1
- streamlit: 1.16.0
- cufflinks: 0.17.3
- numpy    : 1.21.6
- pandas   : 1.3.5
- yfinance : 0.2.3


## Enhancements
### Clustering

## Proposal
We have created an aggregator data web app to analyze historical S&P Global index market data to predict future prices and trends. This is a continuity from Project 1.

We used `yfinance` data on various Financial analysis tools, Machine Learning models, Algorithmic trading & Neural Networks.

## User story

- As a user, I can choose different models to evaluate multiple scenarios, so I can assess a "Buy" or "Sell".

## Machine Learning models

- Unsupervised Clustering: PCA, Standard Scaler, K-Means
- Supervised Classifiers: KNN, SVM, Random Forest
- Supervised Regression: ??
- Algorithmic Trading: ??

- Use silhouette score to measure efficacy of each model

## Next Steps
- Thoughts, Enhancements on WIP web app, Update to streamlit cloud

