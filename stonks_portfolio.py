import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import sys

def app():
    st.title('Portfolio metrics')

    @st.cache(allow_output_mutation=True)
    def get_data():
        return []
	
    col_add = st.beta_columns(2)
    ticker_to_add = col_add[0].text_input("add ticker")
    if col_add[1].button("add to list"):
        get_data().append(ticker_to_add)

    #st.text("get_data object:")
    #st.write(pd.DataFrame(get_data()))

    if "" in get_data():
        get_data().remove("")
    else: 
        pass
    
    col_mod = st.beta_columns(2)
    newlist = col_mod[0].multiselect("selected tickers: ", get_data(), get_data())

    #st.text("get_data object:")
    #st.write(pd.DataFrame(get_data()))
    #st.text("newlist list:")
    #st.write(newlist)

    if col_mod[1].button("update list"):
        get_data().clear()
        get_data().extend(newlist)

    #st.text("get_data object:")
    #st.write(pd.DataFrame(get_data()))
    #st.text("newlist list:")
    #st.write(newlist)
    
    @st.cache
    def load_ticker_info(ticker_list):
        stocks_info = []
        for ticker in ticker_list:
            ticker_info = yf.Ticker(ticker).info
            stocks_info.append(ticker_info)
            #st.write(ticker_info)
        return stocks_info
    
    data_load_state = st.text('Loading data...')
    stocks_info = load_ticker_info(get_data())
    data_load_state.text('Loading data... done!')
    
    stocks_basic_data = pd.DataFrame(columns = ['Ticker', 'Company name'])    
    stocks_name = []
    for idx, stock_info in enumerate(stocks_info):
        if len(stock_info)>1:
            
            df_to_append = pd.DataFrame([[stock_info['symbol'],stock_info['longName']]], columns = ['Ticker', 'Company name'])
        else:
            df_to_append = pd.DataFrame([[get_data()[idx],"no data"]], columns = ['Ticker', 'Company name'])
        #st.write(df_to_append)    
        stocks_basic_data = stocks_basic_data.append(df_to_append)
        #stocks_name.append(stock_info['longName'])
    
    st.text("Selected stocks:")
    st.write(stocks_basic_data)