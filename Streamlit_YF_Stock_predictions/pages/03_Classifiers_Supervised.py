# import libraries - yfinance, prophet, streamlit, plotly
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hvplot.pandas
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
st.set_page_config(page_title="Streamlit ticker(s) forecast app",page_icon="📶",layout="centered")

# Wrap entire code in a streamlit container
with st.container():

    # Create a function to access the json data of the Lottie animation using requests - if successful return 200 - data is good, show animation else return none
    def load_lottieurl(url):
        r=requests.get(url)
        if r.status_code !=200:
            return None
        return r.json()

    # Use local style.css file
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)     
    local_css("./style/style.css") 

    # add title
    st.title(f"Explore different Classifier models")
    st.write("---")

    # import ticker values as tuple values for the streamlit selectbox
    # tickers=("AAPL","AMT","AMZN","GOOG","IYR","META","NFLX","RTH","SPY","XLE","XOM")
    # pd read csv file for tickers list
    # tickers=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt")
    tickers=pd.read_csv("./Resources/s&p500_tickers_2022.csv")

    # add ticker to streamlit sidebar as a selectbox
    ticker_symbol=st.sidebar.selectbox("Select a ticker from the dropdown menu",tickers)
    ticker_data=yf.Ticker(ticker_symbol) 


    # add start/end dates to streamlit sidebar
    # start_date=st.sidebar.multiselect.date_input("Select a start date",datetime.date(2017,1,1))
    # end_date=st.sidebar.multiselect.date_input("Select an end date",datetime.date(2022,12,16))
    start_date=st.sidebar.date_input("Start date",value=pd.to_datetime("2002-01-01"))
    end_date=st.sidebar.date_input("End date",value=pd.to_datetime("today"))
    # add historical trading period for 1 day
    ticker_df=ticker_data.history(period="1d",start=start_date,end=end_date)
    # print(ticker_df.head())
    ####
    #st.write('---')
    # st.write(ticker_data.info)

    # ticker information - logo
    # ticker_logo="<img src=%s>" % ticker_data.info["logo_url"]
    # st.markdown(ticker_logo,unsafe_allow_html=True)

    # ticker information - name
    # ticker_name=ticker_data.info["longName"]
    # st.header(f"{ticker_name}")

    # ticker information - symbol + sector
    # ticker_symbol=ticker_data.info["symbol"]
    # ticker_sector=ticker_data.info["sector"]
    # st.text(f"{ticker_symbol} is part of the {ticker_sector} sector")
    
    # ticker information - summary
    # ticker_summary=ticker_data.info["longBusinessSummary"]
    # st.info(f"{ticker_summary}")

    # Add cache to store ticker values after first time download in browser
    @st.cache

    # Load stock data - define functions
    def load_data(ticker):
        data=yf.download(ticker,start_date,end_date)
        data.reset_index(inplace=True)
        return data
        
    # Classifier models
    # add selectbox for classifiers
    classifier_name=st.selectbox("Select a classifier model from the dropdown menu", ("KNN", "SVM", "Random Forest"))

    # Select classifier
    st.text(f"You have selected the following classifier for {ticker_symbol}:")
    st.subheader(f"{classifier_name}")

    # Get different params for each of the classifiers
    def add_parameter_ui(clf_name):
        params=dict()
        st.text(f"Select parameters from the slider below for {classifier_name}:")
        if clf_name=="KNN":
            K=st.slider("K",1,15)
            params["K"]=K
        elif clf_name=="SVM":
            C=st.slider("C",0.01,10.0) 
            params["C"]=C 
        else:
            max_depth=st.slider("max_depth",2,15)
            n_estimators=st.slider("n_estimators",1,100) 
            params["max_depth"]=max_depth
            params["n_estimators"]=n_estimators     
        return params

    # call add_parameter_ui function with the classifier name
    params=add_parameter_ui(classifier_name)
    
    # data load complete message
    data_load_state=st.sidebar.text("Loading data...⌛")  
    data=load_data(ticker_symbol)
    data_load_state.text("Data loading complete ✅")

    # Create a function for each classifier
    def get_classifier(clf_name,params):
        if clf_name=="KNN":
            clf=KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name=="SVM":
            clf=SVC(C=params["C"]) 
        else:
            clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1)     
        return clf

    # call the function
    clf=get_classifier(classifier_name,params)

    # data classification using train_test_spit
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1) 

    # train classifier by using fit()
    clf.fit(X_train,y_train)

    # call predict method
    y_pred=clf.predict(X_test)

    # create accuracy
    acc=accuracy_score(y_test,y_pred)
    st.text(f"Classifier Type:")
    st.write(f"{classifier_name}")
    st.text(f"Accuracy score:")
    st.write(f"{acc}")

    # PLOT using PCA method and specify number of dimensions
    pca=PCA(2)
    X_projected=pca.fit_transform(X)

    x1=X_projected[:,0]
    x2=X_projected[:,1]

    fig=plt.figure()
    plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.colorbar()

    #plt.show()
    st.pyplot()    