# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
# from streamlit_lottie import st_lottie
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
import datetime

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="S&P500 ticker(s) analysis",page_icon="📈",layout="centered")


# Add cache to store ticker values after first time download in browser
@st.cache(suppress_st_warning=True)

# functions


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
tickers = pd.read_csv("./Resources/s&p500_tickers_2022.csv")

# Display a selectbox for the user to choose a ticker
ticker = st.sidebar.selectbox("Select a ticker from the dropdown menu", tickers)

# Get data for the selected ticker
ticker_data = yf.Ticker(ticker)

# wrap header content in a streamlit container
with st.container():
        # 2 columns section:
        col1, col2 = st.columns([3, 2])
        with col1:           
            # Load title/info
            st.header(f"Time Series forecast")
            st.text(f"With Facebook Prophet")
        with col2:
            st.empty()
st.write("---")

# add start/end dates to streamlit sidebar
start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("2007-1-1"))
end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
# add historical trading period for 1 day
ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)
# st.write(ticker_df.head())
# st.write(ticker_data.info)


# Display company information
show_info_check_box=st.checkbox(label=f"Display {ticker} company info")
if show_info_check_box:
    # ticker information - logo
    ticker_logo="<img src=%s>" % ticker_data.info["logo_url"]
    st.markdown(ticker_logo,unsafe_allow_html=True)
    st.markdown(f"<b>Company Name:</b> {ticker_data.info['longName']}", unsafe_allow_html=True)
    st.markdown(f"<b>Exchange:</b> {ticker_data.info['exchange']}", unsafe_allow_html=True)
    st.markdown(f"<b>Sector:</b> {ticker_data.info['sector']}", unsafe_allow_html=True)
    st.markdown(f"<b>Industry:</b> {ticker_data.info['industry']}", unsafe_allow_html=True)
    st.markdown(f"<b>Full Time Employees:</b> {ticker_data.info['fullTimeEmployees']}", unsafe_allow_html=True)
    st.markdown(f"<b>Website:</b> <a href='{ticker_data.info['website']}'>{ticker_data.info['website']}</a>", unsafe_allow_html=True)
    st.markdown(f"<b>Business Summary:</b>",unsafe_allow_html=True)
    st.info(f"{ticker_data.info['longBusinessSummary']}")



# input a streamlit slider with years of prediction values
n_years=st.sidebar.slider("Select year(s) for time series forecast",1,5)

# Define a yearly period
period=n_years*365


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



st.write("---")   
    
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
forecast_data_check_box=st.checkbox(label=f"Display {ticker} forecast data")
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

# show the plot widget
st.plotly_chart(fig)

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
    
    