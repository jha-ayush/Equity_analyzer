# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
# import yfinance for stock data
import yfinance as yf
#import prophet libraries
from prophet import Prophet
from prophet.plot import plot_plotly
#import plotly for interactive graphs
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
# import cufflinks for bollinger bands
import cufflinks as cf
from datetime import timedelta

import pandas_datareader as pdr

from sklearn.decomposition import PCA # PCA for dimensionality reduction
from sklearn.cluster import KMeans # KMeans for clustering
from sklearn.metrics import silhouette_score # Silhouette score

from graphviz import Digraph # Create flowcharts

from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns

from scipy.stats import gmean

from sklearn.preprocessing import StandardScaler # Scaling data
from sklearn.preprocessing import MinMaxScaler # MinMax scaler
from sklearn.metrics import silhouette_score # To evaluate the clustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.utils import resample

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import logging
import os

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


#______________________________________________________#

# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Stocks analysis",page_icon="📈",layout="centered",initial_sidebar_state="auto")


# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)

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
#load lottie asset
lottie_coding=load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vktpsg4v.json")

# Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")

#-----------------------------------------------#

# wrap header content in a streamlit container
with st.container():
        # 2 columns section:
        col1, col2 = st.columns([3, 2])
        with col1:           
            # Load title/info
            st.title(f"Equity portfolio analyzer")
            st.markdown(f"Our app utilizes advanced algorithms to analyze and predict the performance of your portfolio, providing valuable insights and recommendations to help optimize your investments.",unsafe_allow_html=True)
        with col2:
            # Load asset(s)
            st_lottie(lottie_coding,height=150,key="finance")   
                 
#------------------------------------------------------------------#

# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/tickers.csv")
except:
    logging.error('Cannot find the CSV file')



#------------------------------------------------------#
# Create Navbar tabs
st.write("###")
tab1, tab2, tab3, tab4= st.tabs(["Fin ratios", "Unsupervised", "Supervised", "What's next"])

with tab1:
    st.write(f"Select different boxes to view of an individual ticker over the selected period of time.",unsafe_allow_html=True)
    
    
    # Benchmark ticker - S&P Global index '^GSPC'
    benchmark_ticker=yf.Ticker("^GSPC")

    # Display a selectbox for the user to choose a ticker
    ticker = st.selectbox("Select a ticker from the dropdown menu",tickers)

    # Get data for the selected ticker
    ticker_data = yf.Ticker(ticker)

    #------------------------------------------------------------------#           

    # add start/end dates
    end_date=value=pd.to_datetime("today")
    # calculate start date as 20 years before end date
    start_date = end_date - pd.DateOffset(years=25)

    # Create a new dataframe - add historical trading period for 1 day
    ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)

    # query S&P index historical prices
    benchmark_ticker=benchmark_ticker.history(period="1d",start=start_date,end=end_date)

    # print(ticker_df.head())
    ####
    #st.write('---')
    # st.write(ticker_data.info)

    # Load stock data - define functions
    def load_data(ticker,start_date,end_date):
        data=yf.download(ticker,start_date,end_date)
        # convert the index to a datetime format
        data.index = pd.to_datetime(data.index)
        # use the .rename() function to rename the index to 'Date'
        data = data.rename_axis('Date')
        return data

    # data load complete message
    data_load_state=st.text("Loading data...⌛")  
    data=load_data(ticker,start_date,end_date)
    data_load_state.text("25 years historical data loaded ✅")
    
    
    
    st.subheader(f"Ticker info & financial ratios")
    #---------------------------------------------#
    # Display company info
    if st.checkbox(label=f"Display {ticker} company info"):
        try:
            ticker_data = yf.Ticker(ticker).info
            if isinstance(ticker_data, dict):
                logo_url = ticker_data.get('logo_url', '')
                if logo_url:
                    st.markdown(f"<img src={logo_url}>", unsafe_allow_html=True)
                else:
                    st.warning("Logo image is missing.")

                # Check if company name is available and display
                st.write("###")
                company_name = ticker_data.get('longName', '')
                if company_name:
                    st.markdown(f"<b>Company Name:</b> {company_name}", unsafe_allow_html=True)
                else:
                    st.warning("Company name is missing.")

                # Check if quoteType is available and display
                quoteType = ticker_data.get('quoteType', '')
                if quoteType:
                    st.markdown(f"<b>Quote type:</b> {quoteType}", unsafe_allow_html=True)
                else:
                    st.warning("Quote type is missing.")        

                # Check if sector is available and display
                sector = ticker_data.get('sector', '')
                if sector:
                    st.markdown(f"<b>Sector:</b> {sector}", unsafe_allow_html=True)
                else:
                    st.warning("Sector is missing.")

                # Check if industry is available and display
                industry = ticker_data.get('industry', '')
                if industry:
                    st.markdown(f"<b>Industry:</b> {industry}", unsafe_allow_html=True)
                else:
                    st.warning("Industry is missing.")        

                # Check if location is available and display
                city = ticker_data.get('city', '')
                country = ticker_data.get('country', '')
                if city and country:
                    st.markdown(f"<b>Location:</b> {city}, {country}", unsafe_allow_html=True)
                else:
                    st.warning("Location is missing.")
                
                # Check if website is available and display
                website = ticker_data.get('website', '')
                if website:
                    st.markdown(f"<b>Company Website:</b> {website}", unsafe_allow_html=True)
                else:
                    st.warning("Website is missing.")

                # Check if Business summary is available and display
                summary = ticker_data.get('longBusinessSummary', '')
                if summary:
                    st.info(f"{summary}")
                else:
                    st.warning("Business summary is missing.")
            else:
                st.warning("Data returned is not in the correct format.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        #---------------------------------------------#
        
    # Display data table
    raw_data_check_box=st.checkbox(label=f"Display {ticker} raw dataset")
    if raw_data_check_box:
        st.subheader(f"{ticker} raw data")
        st.write(data)        

    # Display charts
    charts_check_box=st.checkbox(label=f"Display {ticker} charts")
    if charts_check_box:
        # Bollinger bands - trendlines plotted between two standard deviations
        st.header(f"{ticker} Bollinger bands")
        st.info("Bollinger Bands are a technical analysis tool that measures volatility of a financial instrument by plotting three lines: a simple moving average and two standard deviation lines (upper and lower bands). They are used to identify possible overbought or oversold conditions in the market, trend changes and potential buy and sell signals. The upper band is plotted as the moving average plus two standard deviations and lower band is plotted as moving average minus two standard deviations. They should be used in conjunction with other analysis methods for a complete market analysis and not as a standalone method.")
        # Reset index back to original
        data.reset_index(inplace=True)
        # Add description for visualization
        qf=cf.QuantFig(ticker_df,title='Bollinger Quant Figure',legend='top',name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)


        # Plot Open vs Close price data
        def plot_raw_data():
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))
            fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="stock_close"))
            fig.layout.update(title_text=(f"{ticker} raw data plot - Open vs Close price"),xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
        plot_raw_data()

        # create a streamlit linechart for ticker Volume over time
        st.subheader(f"{ticker} trading volume over time")
        st.line_chart(ticker_df.Volume)

# ----------------------------------------------------------------- #
    # Functions - Financial Ratios

    # Returns
    def calculate_returns(ticker, start_date, end_date):
        """
        Calculate the daily returns for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        pandas.DataFrame: The dataframe having close, returns, and daily returns for the stock in the specified range of time.
        """
        # Download the stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Adding new columns for returns and daily returns
        stock_data['returns'] = 0.0
        stock_data['daily_returns'] = 0.0

        #Calculating the returns for each day
        stock_data['returns'] = (stock_data['Adj Close'] - stock_data['Adj Close'].shift(1)) / stock_data['Adj Close'].shift(1)
        stock_data['returns'] = stock_data['returns']*100

        #Calculating the daily returns for each day
        stock_data['daily_returns'] = stock_data['Adj Close'].pct_change()
        stock_data['daily_returns'] = stock_data['daily_returns']*100

        return stock_data.dropna()

    # Daily Returns
    def calculate_daily_returns(ticker, start_date, end_date):
        """
        Calculate the daily returns for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        pandas.Series: The daily returns.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate daily returns
        daily_returns = data['Adj Close'].pct_change()

        return daily_returns.dropna()

    # Mean
    def calculate_mean(ticker, start_date, end_date):
        """
        Calculate the mean of returns for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The mean of returns.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate returns
        returns = data['Adj Close'].pct_change()

        # Calculate mean of returns
        mean = np.mean(returns)

        return mean

    # Std Deviation
    def calculate_std_deviation(ticker, start_date, end_date):
        """
        Calculate the standard deviation of returns for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The standard deviation of returns.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate returns
        returns = data['Adj Close'].pct_change()

        # Calculate standard deviation of returns
        std = np.std(returns)

        return std

    # Variance
    def calculate_variance_returns(ticker, start_date, end_date):
        """
        Calculate the variance of returns for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The variance of returns.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate returns
        returns = data['Adj Close'].pct_change()

        # Calculate variance of returns
        variance = np.var(returns)

        return variance

    # Co-variance
    def calculate_covariance_returns(ticker, benchmark_ticker, start_date, end_date, split_ratio = 0.8):
        """
        Calculate the covariance of returns for two given stock tickers.
        Here we are using ^GSPC as the benchmark ticker.

        Parameters:
        ticker (str): The ticker symbol for the first stock.
        benchmark_ticker (str): The S&P Global symbol for the benchmark index.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        split_ratio (float): The ratio of the data to be used for training (default = 0.8)

        Returns:
        float: The covariance of returns.
        """

        print("1")
        print(benchmark_ticker)
        print("2")
        # Get stock data
        data1 = yf.download(ticker, start=start_date, end=end_date)
        print(f"DATA1: {data1}")
        print("3")
        data2 = yf.download(benchmark_ticker, start=start_date, end=end_date)
        print("4")
        print("5")

        # split data into training and testing sets
        split_point = int(split_ratio * len(data1))
        train_data1, test_data1 = data1[:split_point], data1[split_point:]
        train_data2, test_data2 = data2[:split_point], data2[split_point:]

        # Calculate returns for training data
        train_returns1 = train_data1['Adj Close'].pct_change()
        train_returns2 = train_data2['Adj Close'].pct_change()

        # Calculate covariance of returns
        covariance = np.cov(train_returns1, train_returns2)[0][1]

        return covariance

    # Alpha ratio
    def calculate_alpha_ratio(ticker, benchmark_ticker, start_date, end_date):
        """
        Calculate the alpha ratio for a given stock ticker.
        Here we are using ^GSPC as the benchmark ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        benchmark_ticker (str): The S&P Global symbol for the benchmark index.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The alpha ratio.
        """

        # Get stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

        # Calculate returns
        stock_returns = stock_data['Adj Close'].pct_change()
        benchmark_returns = benchmark_data['Adj Close'].pct_change()

        # Calculate alpha
        alpha = np.mean(stock_returns) - np.mean(benchmark_returns)

        # Calculate standard deviation of returns
        std = np.std(stock_returns)

        # Calculate alpha ratio
        alpha_ratio = alpha / std

        return alpha_ratio.dropna()

    # Beta Ratio
    def calculate_beta_ratio(ticker, benchmark_ticker, start_date, end_date):
        """
        Calculate the beta ratio for a given stock ticker.
        Here we are using ^GSPC as the benchmark ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        benchmark_ticker (str): The ticker symbol for the benchmark index.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The beta ratio.
        """

        # Get stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

        # Calculate returns
        stock_returns = stock_data['Adj Close'].pct_change()
        benchmark_returns = benchmark_data['Adj Close'].pct_change()

        # Calculate beta
        cov = np.cov(stock_returns, benchmark_returns)[0][1]
        var = np.var(benchmark_returns)
        beta = cov / var

        # Calculate standard deviation of returns
        std = np.std(stock_returns)

        # Calculate beta ratio
        beta_ratio = beta / std

        return beta_ratio.dropna()

    # Omega Ratio
    def calculate_omega_ratio(ticker, start_date, end_date, threshold=0):
        """
        Calculate the omega ratio for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        threshold (float): The threshold for calculating excess return and downside risk. Default is 0.

        Returns:
        float: The omega ratio.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate daily returns
        returns = data['Adj Close'].pct_change()

        # Calculate excess return over threshold
        excess_return = returns[returns > threshold].mean()

        # Calculate downside risk below threshold
        downside_risk = abs(returns[returns < threshold]).mean()

        # Calculate omega ratio
        omega_ratio = excess_return / downside_risk

        return omega_ratio

    # Sharpe Ratio
    def calculate_sharpe_ratio(ticker, start_date, end_date, risk_free_rate=0.03):
        """
        Calculate the Sharpe ratio for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        risk_free_rate (float): The risk-free rate of return. Default is 0.

        Returns:
        float: The Sharpe ratio.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate daily returns
        returns = data['Adj Close'].pct_change()

        # Calculate excess return over risk-free rate
        excess_return = returns - risk_free_rate

        # Calculate Sharpe ratio
        sharpe_ratio = excess_return.mean() / np.std(returns)

        return sharpe_ratio

    # Calmar Ratio
    def calculate_calmar_ratio(ticker, start_date, end_date):
        """
        Calculate the Calmar ratio for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

        Returns:
        float: The Calmar ratio.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate daily returns
        returns = data['Adj Close'].pct_change()

        # Calculate annualized compounded return
        compounded_return = (1 + returns).prod() ** (252 / len(returns)) - 1

        # Calculate maximum drawdown
        max_drawdown = (data['Adj Close'] / data['Adj Close'].cummax() - 1).min()

        # Calculate Calmar ratio
        calmar_ratio = compounded_return / max_drawdown

        return calmar_ratio

    # Sortino Ratio
    def calculate_sortino_ratio(ticker, start_date, end_date, threshold=0):
        """
        Calculate the Sortino ratio for a given stock ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        threshold (float): The threshold for calculating downside risk. Default is 0.

        Returns:
        float: The Sortino ratio.
        """

        # Get stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate daily returns
        returns = data['Adj Close'].pct_change()

        # Calculate downside risk below threshold
        downside_risk = np.sqrt(np.square(returns[returns < threshold]).mean())

        # Calculate Sortino ratio
        sortino_ratio = returns.mean() / downside_risk

        return sortino_ratio

    def calculate_treynor_ratio(ticker, start_date, end_date, benchmark_ticker, risk_free_rate=0.03):
        """
        Calculate the Treynor ratio for a given stock ticker.
        Here we are using ^GSPC as the benchmark ticker.

        Parameters:
        ticker (str): The ticker symbol for the stock.
        ^GSPC (str): The ticker symbol for the S&P 500 index.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        risk_free_rate (float): The risk-free rate of return. Default is 0.

        Returns:
        float: The Treynor ratio.
        """

        # Get stock & benchmark data
        stock_data = yf.download(ticker, start_date, end_date)
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

        # Calculate the stock's beta against the benchmark
        covariance = np.cov(stock_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance

        # Calculate the stock's excess return over the benchmark
        benchmark_return = benchmark_returns.mean()
        excess_return = stock_returns.mean() - benchmark_return

        # Calculate the Treynor ratio
        treynor_ratio = excess_return / beta

        return treynor_ratio



    # ----------------------------------------------------------------- #


    # Choose a financial ratio from dropdown menu 

    fin_ratios_check_box=st.checkbox(label=f"Display {ticker} related financial ratios")
    if fin_ratios_check_box:
        with st.container():
                # 2 columns section:
                col1, col2 = st.columns([6, 1])
                with col1:           
                    st.write("###") 
                    ratio_choice = st.selectbox("Choose from one of the financial ratios below",("Returns","Daily returns","Mean","Std-deviation","Variance","Co-variance","Alpha ratio","Beta ratio","Omega ratio","Sharpe ratio","Calmar ratio","Sortino ratio","Treynor ratio"),label_visibility="visible")

                    if ratio_choice == "Returns":
                        st.info("Returns is a measure of gain or loss on an investment over a certain period of time, usually expressed as a percentage of the initial investment. A positive return indicates a profit, while a negative return indicates a loss.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.write(calculate_returns(ticker, start_date, end_date))
                    elif ratio_choice == "Daily returns":
                        st.info("Daily returns calculates the percentage change in the adjusted closing price for each day, which gives the daily returns")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.write(calculate_daily_returns(ticker, start_date, end_date))
                    elif ratio_choice == "Mean":
                        st.info("Mean calcuates the arithmetic mean of the daily returns values")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Mean</b> value for <b>{ticker}</b> is: <b>{calculate_mean(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights the average price for the given time period.",unsafe_allow_html=True)
                    elif ratio_choice == "Std-deviation":
                        st.info("Std-dev is a statistical measure that shows how the data varies from the mean")                    
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Standard deviation</b> value for <b>{ticker}</b> is: <b>{calculate_std_deviation(ticker, start_date, end_date):0.4f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights the volatility of the ticker for the given time period.",unsafe_allow_html=True)
                    elif ratio_choice == "Variance":
                        st.info("Variance variance is a measure of the spread of the data around the mean to calculate risk. The larger the variance, the more spread out the data is, indicating a greater degree of volatility. A smaller variance value, on the other hand, indicates that the data is more tightly clustered around the mean and thus less volatile.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Variance</b> value for <b>{ticker}</b> is: <b>{calculate_variance_returns(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights volatility but only positive values.",unsafe_allow_html=True)
                    elif ratio_choice == "Co-variance":
                        st.info("Covariance is a measure of how two random variables are related and/or change together. A positive covariance indicates that the two variables are positively related, which means that as the value of one variable increases, the value of the other variable also tends to increase. A negative covariance indicates that the two variables are negatively related, which means that as the value of one variable increases, the value of the other variable tends to decrease. A covariance of zero indicates that there is no relationship between the two variables.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.write(calculate_covariance_returns(ticker, benchmark_ticker, start_date, end_date))
                        st.markdown(f"The value highlights how two tickers move in relation to each other",unsafe_allow_html=True)
                    elif ratio_choice == "Alpha ratio":
                        st.info("Alpha ratio is a measure of a stock's performance in relation to its benchmark. A positive alpha value indicates that the stock has performed better than the benchmark (^GSPC), while a negative alpha value indicates underperformance.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Alpha ratio</b> value for <b>{ticker}</b> is: <b>{calculate_alpha_ratio(ticker, benchmark_ticker, start_date, end_date)}</b>",unsafe_allow_html=True)
                        st.markdown(f"This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    elif ratio_choice == "Beta ratio":
                        st.info("Beta ratio is a measure of a stock's volatility in relation to its benchmark index. It compares the volatility of a stock to the volatility of a benchmark index (^GSPC), giving an idea of how much more or less volatile a stock is in relation to the benchmark index. A beta of 1 indicates that the stock's volatility is the same as the benchmark, while a beta greater than 1 indicates that the stock is more volatile than the benchmark, meaning its returns are more sensitive to market movements. Conversely, a beta less than 1 indicates that the stock is less volatile than the benchmark, meaning its returns are less sensitive to market movements.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Beta ratio</b> value for <b>{ticker}</b> is: <b>{calculate_beta_ratio(ticker, benchmark_ticker, start_date, end_date)}</b>",unsafe_allow_html=True)
                        st.markdown(f"This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    elif ratio_choice == "Omega ratio":
                        st.info("Omega ratio is a risk-adjusted performance measure that compares a stock's excess returns to its downside risk. The Omega ratio is similar to the Sharpe ratio, but it gives more weight to returns below a certain threshold, whereas the Sharpe ratio gives equal weight to all returns. A higher omega ratio indicates that the stock has a higher level of excess returns for a given level of downside risk.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Omega ratio</b> value for <b>{ticker}</b> is: <b>{calculate_omega_ratio(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights how well an investment strategy performs, taking into account both the potential returns and the potential risks of the strategy.",unsafe_allow_html=True)
                    elif ratio_choice == "Sharpe ratio":
                        st.info("Sharpe ratio is a measure of a stock's risk-adjusted performance, which compares the stock's excess returns to the volatility of its returns. A higher Sharpe ratio indicates that the stock has a higher level of excess returns for a given level of volatility, which means the stock is a better risk-adjusted performer.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Sharpe ratio</b> value for <b>{ticker}</b> is: <b>{calculate_sharpe_ratio(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights the measure of risk-adjusted performance that compares the excess return of an investment to its volatility.",unsafe_allow_html=True)
                    elif ratio_choice == "Calmar ratio":
                        st.info("Calmar ratio is a measure of a stock's risk-adjusted performance, which compares the stock's compound return to the maximum drawdown. A higher Calmar ratio indicates that the stock has a higher level of returns for a given level of drawdown, which means the stock is a better risk-adjusted performer.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Calmar ratio</b> value for <b>{ticker}</b> is: <b>{calculate_calmar_ratio(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights the profitability of a trading strategy.",unsafe_allow_html=True)
                    elif ratio_choice == "Sortino ratio":
                        st.info("Sortino ratio is a measure of a stock's risk-adjusted performance, which compares the stock's return to the downside risk. A higher Sortino ratio indicates that the stock has a higher level of return for a given level of downside risk, which means the stock is a better risk-adjusted performer.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True)
                        st.markdown(f"The <b>Sortino ratio</b> value for <b>{ticker}</b> is: <b>{calculate_sortino_ratio(ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"The value highlights the performance of trading strategies that are designed to minimize downside risk.",unsafe_allow_html=True)
                    elif ratio_choice == "Treynor ratio":
                        st.info("Treynor ratio is a measure of risk-adjusted return for a portfolio. Similar to the Sharpe ratio, which also measures risk-adjusted return, but the Treynor ratio uses beta as the measure of systematic risk, while the Sharpe ratio uses the standard deviation of returns. A higher Treynor ratio indicates that the portfolio has generated higher returns for the level of systematic risk taken on, as compared to a portfolio with a lower Treynor ratio.")
                        st.markdown(f"You've selected the following financial ratio - <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.",unsafe_allow_html=True) 
                        st.markdown(f"The <b>Treynor ratio</b> value for <b>{ticker}</b> is: <b>{calculate_treynor_ratio(ticker, benchmark_ticker, start_date, end_date):0.5f}</b>",unsafe_allow_html=True)
                        st.markdown(f"This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    else:
                        st.empty()
                    
#----------------------------------------------------#
    # Time Series Forecasting with Facebook Prophet
    # Display Prophet section
    st.subheader("Time series forecast")
    prophet_check_box=st.checkbox(label=f"Display {ticker} Prophet time series forecast data")
    if prophet_check_box:
        with st.container():
                # 2 columns section:
                col1, col2 = st.columns([3, 2])
                with col1:           
                    # input a streamlit slider with years of prediction values
                    n_years=st.slider("Select year(s) for time series forecast",1,5)


                    # create a new dataframe from the ticker_df object
                    df_plot = pd.DataFrame.from_dict(ticker_df, orient='columns')

                    # select the 'Close' column
                    df_plot = df_plot[['Close']]

                    # rename the column to 'y'
                    df_plot.columns = ['y']

                    # add a 'ds' column with the dates, converting it to a datetime object and setting the timezone to None
                    df_plot['ds'] = pd.to_datetime(df_plot.index).tz_localize(None)

                    # Prophet requires a specific column format for the dataframe
                    df_plot = df_plot[['ds', 'y']]


                    # create the Prophet model and fit it to the data
                    model = Prophet(daily_seasonality=True)
                    model.fit(df_plot)

                    # create a dataframe with future dates
                    future_dates = model.make_future_dataframe(periods=365)

                    # make predictions for the future dates
                    forecast = model.predict(future_dates)

                    # select the relevant columns for the plot
                    plot_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                    # Display data table
                    forecast_data_check_box=st.checkbox(label=f"Display {ticker} forecast data & price prediction")
                    if forecast_data_check_box:
                        st.subheader(f"{ticker} forecast dataset")
                        # Show tail of the Forecast data
                        st.write(forecast.tail())
                        st.write("---")

                        # create a plotly figure
                        fig = go.Figure()

                        # add the predicted values to the figure
                        fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat'], name='Prediction'))

                        # add the uncertainty intervals to the figure
                        fig.add_shape(
                                type='rect',
                                xref='x',
                                yref='paper',
                                x0=plot_df['ds'].min(),
                                y0=0,
                                x1=plot_df['ds'].max(),
                                y1=1,
                                fillcolor='#E8E8E8',
                                layer='below',
                                line_width=0
                            )
                        fig.add_shape(
                                type='rect',
                                xref='x',
                                yref='y',
                                x0=plot_df['ds'].min(),
                                y0=plot_df['yhat_upper'],
                                x1=plot_df['ds'].max(),
                                y1=plot_df['yhat_lower'],
                                fillcolor='#E8E8E8',
                                layer='below',
                                line_width=0
                            )

                        # add the actual values to the figure
                        fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['y'], name='Actual'))

                        # set the plot's title and labels
                        fig.update_layout(
                            title=f"{ticker} stock price prediction",
                            xaxis_title='Date',
                            yaxis_title='Price (USD)'
                        )

                        # show the prediction plot
                        st.plotly_chart(fig)

                        # Display Prophet tools & components
                        forecast_component_check_box=st.checkbox(label=f"Display {ticker} Prophet forecast components")
                        if forecast_component_check_box:

                            # create a plotly figure for the model's components
                            st.subheader(f"{ticker} plot widget")
                            fig2 = plot_plotly(model, forecast)
                            # show the plot
                            st.plotly_chart(fig2)


                            # show the model's plots
                            st.subheader(f"{ticker} forecast components")
                            st.write(model.plot(forecast))

                            # show the model's plot_components
                            st.write(model.plot_components(forecast))
                                
#-----------------------------------------------#
# Download flowchart

# Save the ticker-cluster probability data to a CSV file
    st.write("---")
    if st.button('Save workflow flowchart'):
        # Create a new flowchart
        flowchart = Digraph()

        # Add start node
        flowchart.node('Start')

        # Add a read CSV node
        flowchart.node('Read S&P500 tickers CSV')

        # Add a new column node
        flowchart.node('Map ticker to yfinance historical data')

        # Add a remove columns node
        flowchart.node('Data Cleanup & wrangling')

        # Add a sort dataframe node
        flowchart.node('Tab 1: Display single ticker info, raw data, Bollinger bands,Financial ratios, Prophet Time Series Forecast, Message Us')

        # Add a group by sector node
        flowchart.node('Tab 2: Unsupervised Learning using K-Means/Silhoutte score to get optimized clusters for Top 10 tickers (by marketcap) in the Top 10 sectors (by count), using daily returns. Data is saved as a csv in the "results" folder. Monte Carlo simulation is applied to the clustered data.')

        # Add a create dictionary node
        flowchart.node('Tab 3: Supervised Learning')

        # Add end node
        flowchart.node('End')

        # Connect the nodes with arrows
        flowchart.edge('Start', 'Read S&P500 tickers CSV')
        flowchart.edge('Read S&P500 tickers CSV', 'Map ticker to yfinance historical data')
        flowchart.edge('Map ticker to yfinance historical data', 'Data Cleanup & wrangling')
        flowchart.edge('Data Cleanup & wrangling', 'Tab 1: Display single ticker info, raw data, Bollinger bands,Financial ratios, Prophet Time Series Forecast, Message Us')
        flowchart.edge('Tab 1: Display single ticker info, raw data, Bollinger bands,Financial ratios, Prophet Time Series Forecast, Message Us', 'Tab 2: Unsupervised Learning using K-Means/Silhoutte score to get optimized clusters for Top 10 tickers (by marketcap) in the Top 10 sectors (by count), using daily returns. Data is saved as a csv in the "results" folder. Monte Carlo simulation is applied to the clustered data.')
        flowchart.edge('Tab 2: Unsupervised Learning using K-Means/Silhoutte score to get optimized clusters for Top 10 tickers (by marketcap) in the Top 10 sectors (by count), using daily returns. Data is saved as a csv in the "results" folder. Monte Carlo simulation is applied to the clustered data.', 'Tab 3: Supervised Learning')
        flowchart.edge('Tab 3: Supervised Learning', 'End')

        # Render the flowchart
        flowchart.render('./results/flowchart.png', view=True)
        st.balloons()


#------------------------------------------------------------------#                      
# Tab 2 - Unsupervised Learning                    
with tab2:
    with st.container():
                # 2 columns section:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Create a dataframe for the csv file
                    try:
                        symbols_df = pd.read_csv("./Resources/tickers.csv")
                    except:
                        logging.error('Cannot find the CSV file')
                        
                        
                    ticker_df_check_box=st.checkbox(label=f"Display tickers dataframe")
                    if ticker_df_check_box:
                        #Display tickers dataframe    
                        st.write(symbols_df)    
                    
                    # Add new column "Market Cap" to ticker_df also
                    # ticker_df["Market Cap"] = ticker_df["Close"] * ticker_df["Volume"]
                    # Remove 'Dividends' & 'Stock Splits' from `ticker_df`
                    # ticker_df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
                    # Sort the sectors by market capitalization
                    # ticker_df.sort_values(by='Market Cap', ascending=False, inplace=True)
                    # Display the sorted data in a table
                    # st.table(ticker_df)
                    
                    # Group the data by sector and count the number of companies in each sector
                    sector_counts = symbols_df['sector'].value_counts()


                    # Download symbols data
                    symbols_data = yf.download(ticker, start=start_date, end=end_date) 

                    # Define the sectors
                    # "XLF" represents the Financial Select Sector SPDR Fund which tracks the performance of the financial sector of the S&P 500 index.
                    # "XLE" represents the Energy Select Sector SPDR Fund which tracks the performance of the energy sector of the S&P 500 index.
                    # "XLK" represents the Technology Select Sector SPDR Fund which tracks the performance of the technology sector of the S&P 500 index.
                    # "XLP" represents the Consumer Staples Select Sector SPDR Fund which tracks the performance of the consumer staples sector of the S&P 500 index.
                    # "XLV" represents the Health Care Select Sector SPDR Fund which tracks the performance of the healthcare sector of the S&P 500 index.
                    # "XLY" represents the Consumer Discretionary Select Sector SPDR Fund which tracks the performance of the consumer discretionary sector of the S&P 500 index.
                    # "XLC" represents the Communications Services Select Sector SPDR Fund which tracks the performance of the communications services sector of the S&P 500 index.
                    # "XLI" represents the Industrials Select Sector SPDR Fund which tracks the performance of the industrials sector of the S&P 500 index.
                    # "XLB" represents the Materials Select Sector SPDR Fund which tracks the performance of the materials sector of the S&P 500 index.
                    # "XLRE" represents the Real Estate Select Sector SPDR Fund which tracks the performance of the real estate sector of the S&P 500 index.
                    # "XLU" represents the Utilities Select Sector SPDR Fund which tracks the performance of the utilities sector of the S&P 500 index.
                    
                    
                    sectors_check_box=st.checkbox(label=f"Display ticker sectors list")
                    if sectors_check_box:
                        #Display ticker sectors     
                        sectors = ["XLF - Financials","XLE - Energy","XLK - Information Technology","XLP - Consumer Staples","XLV - Health Care","XLY - Consumer Discretionary","XLC - Communications Services","XLI - Industrials","XLB - Materials","XLRE - Real Estate","XLU - Utilities"]
                        st.write(sectors)
                    
                    # Group the data by sector
                    sectors_df = symbols_df.groupby('sector')

                    
                    grouped_tickers_check_box=st.checkbox(label=f"Display tickers grouped in sectors")
                    if grouped_tickers_check_box:
                        #Display ticker sectors     
                        st.write(f"<b>Grouped tickers (int values) by sectors</b>",(sectors_df.groups),unsafe_allow_html=True)
                        # Create a dictionary that maps sector names to ticker symbols
                        sector_ticker_map = {"Financials": "XLF", "Energy": "XLE", "Information Technology": "XLK", "Consumer Staples": "XLP", "Health Care": "XLV", "Consumer Discretionary": "XLY", "Communications Services": "XLC", "Industrials": "XLI", "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU"}
                        # Use the map() function to replace the sector names with the corresponding ticker symbols
                        symbols_df["sector"] = symbols_df["sector"].map(sector_ticker_map)
                        
                    tickers_markcap_check_box=st.checkbox(label=f"Display tickers with Market cap")
                    if tickers_markcap_check_box:
                        # Display dataframe with sector ticker info
                        st.write(f"<b>Tickers with Market Cap</b>",unsafe_allow_html=True)
                        st.write(symbols_df)
                    
                    
                    sector_count_check_box=st.checkbox(label=f"Display ticker counts in each sectors")
                    if sector_count_check_box:
                        # Create a new DataFrame with the sector counts
                        sectors_df = pd.DataFrame({'sector': sector_counts.index, 'count': sector_counts.values})

                        st.write(f"<b>Sectors and number of companies in each sector</b>",unsafe_allow_html=True)
                        # Display the new DataFrame in a table using streamlit
                        st.table(sectors_df)
                    
                        # Group the symbols_df DataFrame by the 'sector' column
                        grouped_df = symbols_df.groupby('sector').size().reset_index(name='counts')

                        # Use the plot() function to create a bar chart of the groups
                        st.write(f"<b>Number of tickers in each sector</b>",unsafe_allow_html=True)
                        grouped_df.plot(kind='bar', x='sector', y='counts',color='green')

                        # Show the plot
                        st.pyplot()
                    
                        # Show Top 5 sectors
                        top_5_sectors = sectors_df.sort_values(by='count', ascending=False).head(5)
                        st.write(f"<b>Top 5 sectors by number of companies</b>",unsafe_allow_html=True)
                        st.table(top_5_sectors)
                        # Find data types/info
                        # st.write(symbols_df.dtypes)
                    
                        # Group the symbols_df dataframe by the 'sector' column
                        grouped_df = symbols_df.groupby('sector')

                        # Create an empty list to store the top 10 companies from each sector
                        top_10_companies = []

                        # Conver Market Cap objectype to int
                        symbols_df["market_cap"] = symbols_df["market_cap"].str.replace(',','')
                        symbols_df["market_cap"] = symbols_df["market_cap"].str.replace('$','')

                        # Convert market_cap to numeric
                        symbols_df["market_cap"] = pd.to_numeric(symbols_df["market_cap"])
                        # st.write(symbols_df)


                
                    
                        # Iterate over the sectors
                        for sector, group in grouped_df:
                            # Select the top 10 companies from the current sector based on market cap
                            top_10_companies.append(group.nlargest(10, 'market_cap'))

                        # Concatenate all the top 10 company dataframes into a single dataframe
                        top_10_companies_df = pd.concat(top_10_companies)
                        st.write(f"<b>Top 10 companies by Market Cap in each sector</b>",unsafe_allow_html=True)
                        st.write(f"<b>Total companies: ",{top_10_companies_df.shape},unsafe_allow_html=True)
                        st.write(top_10_companies_df)

                    
                    
                        # Shift the market_cap values up by 1 position
                        top_10_companies_df['market_cap_shifted'] = top_10_companies_df['market_cap'].shift(1)

                        # Calculate the daily return using the shifted values
                        top_10_companies_df['daily_return'] = (top_10_companies_df['market_cap'] - top_10_companies_df['market_cap_shifted']) / top_10_companies_df['market_cap_shifted']

                        # Drop the shifted column
                        top_10_companies_df.drop(columns=['market_cap_shifted'], inplace=True)

                        # Drop the rows with missing values
                        top_10_companies_df.dropna(inplace=True)

                        st.write(f"<b>Total companies with daily returns: ",{top_10_companies_df.shape},unsafe_allow_html=True)
                        st.write(top_10_companies_df)


                    
                    
                    # Save the ticker-cluster probability data to a CSV file
                    if st.button('Optimize with Silhouette score & run K-means algorithm'):
                        
                        # Select the columns from top_10_companies_df that will be used for clustering
                        X = top_10_companies_df[['daily_return', 'market_cap']]

                        # Initialize an empty list to store the silhouette scores
                        silhouette_scores = []

                        # Loop through a range of possible number of clusters
                        for n_clusters in range(2, 11):
                            # Initialize the KMeans model
                            kmeans = KMeans(n_clusters=n_clusters)
                            # Fit the model to the data
                            kmeans.fit(X)
                            # Predict the cluster labels for each data point
                            labels = kmeans.predict(X)
                            # Append the silhouette score to the list
                            silhouette_scores.append(silhouette_score(X, labels))
                            # Find the index of the highest silhouette score
                            optimal_number_of_clusters = np.argmax(silhouette_scores) + 2
                            # Show silhouette scores
                            # st.write(silhouette_scores)


                        # Re-initialize the model with the optimal number of clusters
                        kmeans = KMeans(n_clusters=optimal_number_of_clusters)
                        # Fit the model to the data
                        kmeans.fit(X)
                        # Assign each company to a cluster
                        top_10_companies_df['cluster'] = kmeans.predict(X)

                        # Create a new DataFrame with the cluster labels
                        st.write(f"<b>Tickers with cluster labels</b>",unsafe_allow_html=True)
                        cluster_df = top_10_companies_df[['ticker', 'name', 'cluster']]

                        # Display the table
                        st.table(cluster_df)
                    
                    
            
                    # Save the ticker-cluster probability data to a CSV file
                    if st.button('Save ticker-cluster probability data'):
                        try:
                            # Save to folder 'results'
                            cluster_df.to_csv('./results/cluster_df.csv', index=False)
                            st.success("Ticker-cluster probability data saved to CSV file successfully! ✅ ")
                            st.balloons()
                        except Exception as e:
                            st.error("Error saving ticker-cluster probability data to CSV file. ❌ Please try again! ")
                            st.exception(e)
                            
                                                 
                # Empty 2nd column    
                with col2:
                    st.empty()

                
#-------------------------------------------------------------------#

# Tab 3 - Supervised Learning
with tab3:
    stock_df=pd.DataFrame(data)
    #st.write(stock_df)   
    weekly_data=stock_df.resample('W').last()
    # st.write(f"Show weekly data")
    # st.write(weekly_data)
    signals_df = weekly_data.loc[:, ["Open","High","Low","Volume","Close","Adj Close"]]
    signals_df["Actual Returns"] = signals_df["Close"].pct_change()
    signals_df = signals_df.dropna()
    st.write(f"Show Actual returns of the signal data")
    st.write(signals_df)

    X = signals_df[["Open","High","Low","Volume","Adj Close"]]
    y = signals_df['Close']
    
    rf_mean = 0
    knn_mean = 0
    svm_mean = 0
    rf_r2 = 0
    knn_r2 = 0
    svm_r2 = 0

    #Random Foresh Regressor
    def random_forest(X,y):
        global rf_mean
        global rf_r2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        rf_mean=mean_squared_error(y_test, y_pred)
        rf_r2=r2_score(y_test, y_pred)
        return y_pred

    #KNN 
    def KNN(X,y):
        global knn_mean
        global knn_r2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        knn = KNeighborsRegressor(n_neighbors=2)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        knn_mean=mean_squared_error(y_test, y_pred)
        knn_r2=r2_score(y_test, y_pred)
        return y_pred

    #SVM
    def SVM(X,y):    
        global svm_mean
        global svm_r2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        svr = SVR(kernel='linear', gamma=0.1)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        svm_mean=mean_squared_error(y_test, y_pred)
        svm_r2=r2_score(y_test, y_pred)
        return y_pred  

    def best_accuracy_model(rf_mean, knn_mean):
        mean = [rf_mean, knn_mean]
        Best_Model = min(mean)
        return Best_Model


    if __name__ =="__main__":
        
        rf_func = random_forest(X,y)
        knn_func = KNN(X,y)
        
        model = st.selectbox("Choose from one of the models below",("Random Forest","KNN"),label_visibility="visible")
        if model == 'Random Forest':
            st.write(rf_func)
            st.write(f"",unsafe_allow_html=True)
        elif model == 'KNN':        
            st.write(knn_func)
            st.write(f"",unsafe_allow_html=True)
        #elif model == 'SVM':        
            #st.write(SVM(X,y))
        else:    
            st.write(f'Model is not valid') 
            
            
        st.write(f'Mean error of Random Forest model',rf_mean)
        st.write(f'Mean error of KNN model',knn_mean)
        #st.write(f'Mean of SVM model',svm_mean)

        st.write(f'Model with minimum error is',best_accuracy_model(rf_mean, knn_mean)) 
        st.write(f"Mean Error is a evaluation method to measure the efficiency of the model. Lower the mean error value, model is more efficient. R2 is an evaluation method to measure the efficiency of the model. Higher the R2 value, the more efficient the model is.",unsafe_allow_html=True)

        st.write(f'R2 of Random Forest model',rf_r2)
        st.write(f'R2 of KNN Forest model',knn_r2)
        #st.write(f'R2 of SVM model',svm_r2)

        st.write(f'Model with maximum r2 value is',max(rf_r2, knn_r2))            
            
#-------------------------------------------------------------------#

# Tab 4 - What's Next
with tab4:
    st.subheader("What's next: Business")
    st.warning("Optimizing the solution to make it a subscription model with robo advisors")
    st.warning("Fix 2")
    st.warning("Fix 3")
    st.subheader("What's next: Technical")
    st.warning("Error handling & user-login for features")
    st.warning("Use of superior metrics to fit in clustering")
    st.warning("External API that is more reliable than `yfinance`")
    
    #-----------------------------------------------#
    # Contact Form
    with st.container():
        st.write("---")
        st.subheader("Message us")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/jha.ayush85@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
    # Display form
    with st.container():    
        left_column, mid_column, right_column = st.columns(3)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
            # Display balloons
            # st.balloons()
            # st.snow()
        with mid_column:
            st.empty()
        with right_column:
            st.empty()
        
        