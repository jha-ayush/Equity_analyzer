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
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import requests
# import cufflinks for bollinger bands
import cufflinks as cf

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


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

# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/tickers.csv")
except:
    logging.error('Cannot find the CSV file')

# Benchmark ticker - S&P Global index '^GSPC'
benchmark_ticker=yf.Ticker("^GSPC")

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
            st.title(f"Stocks analyzer & predictor")
            st.markdown(f"Ne quis facete usu, vis nostro iudicabit ut, et ius ullum constituam. Vim suas molestie an, nam id fuisset lucilius. No duo elit labores prodesset, stet nemore usu ex, vis stet pertinacia efficiendi id. Aliquando signiferumque qui at.")
        with col2:
            # Load asset(s)
            st_lottie(lottie_coding,height=150,key="finance")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("1997-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# Create a new dataframe - add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)

# query S&P index historical prices
benchmark_ticker=benchmark_ticker.history(period="1d",start=start_date,end=end_date)

# print(ticker_df.head())
####
#st.write('---')
# st.write(ticker_data.info)

st.write("---")
st.subheader(f"Ticker info & financial ratios")
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
        
    # Check if quoteType is available and display
    quoteType = ticker_data.get('quoteType')
    if quoteType:
        st.markdown(f"<b>Quote type:</b> {quoteType}", unsafe_allow_html=True)
    else:
        st.warning("Quote type is missing.")        

    # Check if sector is available and display
    sector = ticker_data.get('sector')
    if sector:
        st.markdown(f"<b>Sector:</b> {sector}", unsafe_allow_html=True)
    else:
        st.warning("Sector is missing.")
        
    # Check if industry is available and display
    industry = ticker_data.get('industry')
    if industry:
        st.markdown(f"<b>Industry:</b> {industry}", unsafe_allow_html=True)
    else:
        st.warning("Industry is missing.")        

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
def load_data(ticker,start_date,end_date):
    data=yf.download(ticker,start_date,end_date)
    # convert the index to a datetime format
    data.index = pd.to_datetime(data.index)
    # use the .rename() function to rename the index to 'Date'
    data = data.rename_axis('Date')
    return data

# data load complete message
data_load_state=st.sidebar.text("Loading data...⌛")  
data=load_data(ticker,start_date,end_date)
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

def calculate_treynor_ratio(ticker, start_date, end_date, benchmark_ticker, risk_free_rate=0.3):
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
                    
                    
#-----------------------------------------------# 

# Clusters Unsupervised
st.write("---")
st.subheader(f"Clustering models evaluations")

unsup_check_box=st.checkbox(label=f"Display Unsupervised Clusters Machine Learning models for {ticker}")
if unsup_check_box:
    st.write(f"HELLLLLLLOOOOO UNSUPERVISED")
    
    if benchmark_ticker in tickers.ticker.tolist():
        ticker = st.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist(), default = [benchmark_ticker])
    else:
        ticker = st.multiselect("Select ticker(s) for clustering", tickers.ticker.tolist())
        # Get the number of clusters from user input
        n_clusters = st.slider("Number of clusters", 2, 10, 4)

# Get the data that the user wants to use for clustering
clustering_data = st.multiselect("Select data to use for clustering", ['Open', 'High', 'Low','Close','Adj Close','Volume'], default = ['Adj Close'])

# Dimensionality reduction technique
reduction_method = st.selectbox("Select dimensionality reduction technique", ['PCA','t-SNE','MDS','Isomap','LLE','SE'])

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
                                st.write(f"Mean value of cluster {i} is:",(np.mean(cluster)))
                                st.write(f"Median value of cluster {i} is:",(np.median(cluster)))
                                st.write(f"Variance value of cluster {i} is:",(np.var(cluster)))
        

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

#-----------------------------------------------#

# Classifiers Supervised
st.write("---")
st.subheader(f"Clustering models evaluations")

sup_check_box=st.checkbox(label=f"Display Supervised Classifiers Machine Learning models for {ticker}")
if sup_check_box:
    st.write(f"HELLLLLLLOOOOO SUPERVISED")

#-----------------------------------------------#

# Time Series Forecasting with Facebook Prophet

# Display Prophet section
st.write("---")
st.subheader("Time series forecast")
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
        
        