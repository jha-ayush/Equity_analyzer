# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
from streamlit_lottie import st_lottie
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
import requests
# import cufflinks for bollinger bands
import cufflinks as cf
import datetime

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="S&P Global ticker(s) analysis",page_icon="📈",layout="centered",initial_sidebar_state="auto")


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

# include params & data
# Read ticker symbols from a CSV file
tickers = pd.read_csv("./Resources/s&p_global_tickers_2022.csv")

# Benchmark ticker - S&P Global index 'SPGI'
benchmark_ticker=yf.Ticker("SPGI")


# Display a selectbox for the user to choose a ticker
ticker = st.sidebar.selectbox("Select a ticker from the dropdown menu",tickers)


# Get data for the selected ticker
ticker_data = yf.Ticker(ticker)

# wrap header content in a streamlit container
with st.container():
        # 2 columns section:
        col1, col2 = st.columns([3, 2])
        with col1:           
            # Load title/info
            st.header(f"S&P Global ticker healthcheck")
            st.markdown(f"Ne quis facete usu, vis nostro iudicabit ut, et ius ullum constituam. Vim suas molestie an, nam id fuisset lucilius. No duo elit labores prodesset, stet nemore usu ex, vis stet pertinacia efficiendi id. Aliquando signiferumque qui at.")
        with col2:
            # Load asset(s)
            st_lottie(lottie_coding,height=150,key="finance")
st.write("---")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("1997-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# Create a new dataframe - add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)

# query S&P Global historical prices
benchmark_ticker=benchmark_ticker.history(period="1d",start=start_date,end=end_date)

# print(ticker_df.head())
####
#st.write('---')
# st.write(ticker_data.info)


# Display company info
if st.checkbox(label=f"Display {ticker} company info"):
    ticker_data = yf.Ticker(ticker).info

    # Check if logo URL is available and display
    logo_url = ticker_data.get('logo_url')
    if logo_url:
        st.markdown(f"<img src={logo_url}>", unsafe_allow_html=True)
    else:
        st.warning("Logo image is missing.")

    # Check if company name is available and display
    st.write("###")
    company_name = ticker_data.get('longName')
    if company_name:
        st.markdown(f"<b>Company Name:</b> {company_name}", unsafe_allow_html=True)
    else:
        st.warning("Company name is missing.")

    # Check if sector is available and display
    sector = ticker_data.get('sector')
    if sector:
        st.markdown(f"<b>Sector:</b> {sector}", unsafe_allow_html=True)
    else:
        st.warning("Sector is missing.")

    # Check if location is available and display
    city = ticker_data.get('city')
    country = ticker_data.get('country')
    if city and country:
        st.markdown(f"<b>Location:</b> {city}, {country}", unsafe_allow_html=True)
    else:
        st.warning("Location is missing.")

    # Check if website is available and display
    website = ticker_data.get('website')
    if website:
        st.markdown(f"<b>Company Website:</b> {website}", unsafe_allow_html=True)
    else:
        st.warning("Website is missing.")

    # Check if Business summary is available and display
    summary = ticker_data.get('longBusinessSummary')
    if summary:
        st.info(f"{summary}")
    else:
        st.warning("Business summary is missing.")
        

# Load stock data - define functions
def load_data(ticker):
    data=yf.download(ticker,start_date,end_date)
    # data.set_index("Date",inplace=True,append=True,drop=True)
    data.reset_index(inplace=True)
    return data

# data load complete message
data_load_state=st.sidebar.text("Loading data...⌛")  
data=load_data(ticker)
data_load_state.text("Data loading complete ✅")


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
    
    return stock_data

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

    return daily_returns

# Mean
def calculate_mean_returns(ticker, start_date, end_date):
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
def calculate_covariance_returns(ticker1, ticker2, start_date, end_date):
    """
    Calculate the covariance of returns for two given stock tickers.
    Can be used for comparison against SPGI as well.
    
    Parameters:
    ticker1 (str): The ticker symbol for the first stock.
    ticker2 (str): The ticker symbol for the second stock.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
    float: The covariance of returns.
    """
    
    # Get stock data
    data1 = yf.download(ticker1, start=start_date, end=end_date)
    data2 = yf.download(ticker2, start=start_date, end=end_date)

    # Calculate returns
    returns1 = data1['Adj Close'].pct_change()
    returns2 = data2['Adj Close'].pct_change()

    # Calculate covariance of returns
    covariance = np.cov(returns1, returns2)[0][1]

    return covariance

# Alpha ratio
def calculate_alpha_ratio(ticker1, benchmark_ticker, start_date, end_date):
    """
    Calculate the alpha ratio for a given stock ticker.
    Here we are using SPGI as the benchmark ticker.
    
    Parameters:
    ticker (str): The ticker symbol for the stock.
    benchmark_ticker (str): The S&P Global symbol for the benchmark index.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
    float: The alpha ratio.
    """
    
    # Get stock data
    stock_data = yf.download(ticker1, start=start_date, end=end_date)
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

    return alpha_ratio

# Beta Ratio
def calculate_beta_ratio(ticker1, benchmark_ticker, start_date, end_date):
    """
    Calculate the beta ratio for a given stock ticker.
    Here we are using SPGI as the benchmark ticker.
    
    Parameters:
    ticker (str): The ticker symbol for the stock.
    benchmark_ticker (str): The ticker symbol for the benchmark index.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
    float: The beta ratio.
    """
    
    # Get stock data
    stock_data = yf.download(ticker1, start=start_date, end=end_date)
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

    return beta_ratio

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
def calculate_sharpe_ratio(ticker, start_date, end_date, risk_free_rate=0):
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

# Treynor Ratio
def calculate_treynor_ratio(ticker1, benchmark_ticker, start_date, end_date, risk_free_rate=0):
    """
    Calculate the Treynor ratio for a given stock ticker.
    Here we are using SPGI as the benchmark ticker.
    
    Parameters:
    ticker (str): The ticker symbol for the stock.
    spgi (str): The ticker symbol for the S&P 500 index.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.
    risk_free_rate (float): The risk-free rate of return. Default is 0.
    
    Returns:
    float: The Treynor ratio.
    """
    # Get stock data
    stock_data = get_stock_data(ticker1, start_date, end_date)
    
    # Calculate returns
    returns = calculate_returns(stock_data)
    
    # Calculate beta
    beta = calculate_beta(ticker1, benchmark_ticker, start_date, end_date)
    
    # Calculate mean of returns
    mean = calculate_mean(returns)
    
    # Calculate Treynor ratio
    treynor_ratio = (mean - risk_free_rate) / beta
    
    return treynor_ratio


# ----------------------------------------------------------------- #


# Choose a financial ratio from dropdown menu 
# Display financial ratios
fin_ratios_check_box=st.checkbox(label=f"Display {ticker} related financial ratios")
if fin_ratios_check_box:
    with st.container():
            # 2 columns section:
            col1, col2 = st.columns([3, 2])
            with col1:           
                st.write("###") 
                ratio_choice = st.selectbox("Choose from one of the financial ratios below (UNDER CONSTRUCTION)",("Returns","Daily returns","Mean","Std-deviation","Variance","Co-variance","Alpha ratio","Beta ratio","Omega ratio","Sharpe ratio","Calmar ratio","Sortino ratio","Treynor ratio"),label_visibility="visible")

                if ratio_choice == "Returns":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Returns is a measure of gain or loss on an investment over a certain period of time, usually expressed as a percentage of the initial investment. A positive return indicates a profit, while a negative return indicates a loss.")
                    pass
                elif ratio_choice == "Daily returns":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Daily returns calculates the percentage change in the adjusted closing price for each day, which gives the daily returns")
                    pass
                elif ratio_choice == "Mean":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Mean calcuates the arithmetic mean of the daily returns values")
                    pass
                elif ratio_choice == "Std-deviation":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Std-dev is a statistical measure that shows how the data varies from the mean")
                    pass
                elif ratio_choice == "Variance":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Variance variance is a measure of the spread of the data around the mean to calculate risk. The larger the variance, the more spread out the data is, indicating a greater degree of volatility. A smaller variance value, on the other hand, indicates that the data is more tightly clustered around the mean and thus less volatile.")
                    pass
                elif ratio_choice == "Co-variance":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Covariance is a measure of how two random variables are related and/or change together. A positive covariance indicates that the two variables are positively related, which means that as the value of one variable increases, the value of the other variable also tends to increase. A negative covariance indicates that the two variables are negatively related, which means that as the value of one variable increases, the value of the other variable tends to decrease. A covariance of zero indicates that there is no relationship between the two variables.")
                    pass
                elif ratio_choice == "Alpha ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Alpha ratio is a measure of a stock's performance in relation to its benchmark. A positive alpha value indicates that the stock has performed better than the benchmark (SPGI), while a negative alpha value indicates underperformance.")
                    pass
                elif ratio_choice == "Beta ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Beta ratio is a measure of a stock's volatility in relation to its benchmark index. It compares the volatility of a stock to the volatility of a benchmark index (SPGI), giving an idea of how much more or less volatile a stock is in relation to the benchmark index. A beta of 1 indicates that the stock's volatility is the same as the benchmark, while a beta greater than 1 indicates that the stock is more volatile than the benchmark, meaning its returns are more sensitive to market movements. Conversely, a beta less than 1 indicates that the stock is less volatile than the benchmark, meaning its returns are less sensitive to market movements.")
                    pass
                elif ratio_choice == "Omega ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Omega ratio is a risk-adjusted performance measure that compares a stock's excess returns to its downside risk. The Omega ratio is similar to the Sharpe ratio, but it gives more weight to returns below a certain threshold, whereas the Sharpe ratio gives equal weight to all returns. A higher omega ratio indicates that the stock has a higher level of excess returns for a given level of downside risk.")
                    pass
                elif ratio_choice == "Sharpe ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Sharpe ratio is a measure of a stock's risk-adjusted performance, which compares the stock's excess returns to the volatility of its returns. A higher Sharpe ratio indicates that the stock has a higher level of excess returns for a given level of volatility, which means the stock is a better risk-adjusted performer.")
                    pass
                elif ratio_choice == "Calmar ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Calmar ratio is a measure of a stock's risk-adjusted performance, which compares the stock's compound return to the maximum drawdown. A higher Calmar ratio indicates that the stock has a higher level of returns for a given level of drawdown, which means the stock is a better risk-adjusted performer.")
                    pass
                elif ratio_choice == "Sortino ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Sortino ratio is a measure of a stock's risk-adjusted performance, which compares the stock's return to the downside risk. A higher Sortino ratio indicates that the stock has a higher level of return for a given level of downside risk, which means the stock is a better risk-adjusted performer.")
                    pass
                elif ratio_choice == "Treynor ratio":
                    st.markdown(f"You've selected the following financial ratio, <b>{ratio_choice}</b>, for the ticker <b>{ticker}</b>, from the S&P Global index, between <b>{start_date}</b> and <b>{end_date}</b>.<br>This highlights some of the following XYZ actions...",unsafe_allow_html=True)
                    st.info("Treynor ratio is a measure of risk-adjusted return for a portfolio. Similar to the Sharpe ratio, which also measures risk-adjusted return, but the Treynor ratio uses beta as the measure of systematic risk, while the Sharpe ratio uses the standard deviation of returns. A higher Treynor ratio indicates that the portfolio has generated higher returns for the level of systematic risk taken on, as compared to a portfolio with a lower Treynor ratio.")
                    pass            
                else:
                    st.empty()

#-----------------------------------------------#

# Time Series Forecasting with Facebook Prophet

# Display Prophet section
prophet_check_box=st.checkbox(label=f"Display {ticker} Prophet time series forecast data")
if prophet_check_box:
    with st.container():
            # 2 columns section:
            col1, col2 = st.columns([3, 2])
            with col1:           
                st.write("###") 
                st.write("###")
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