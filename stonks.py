import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import sys

def app():
	START = "2015-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")

	st.title('Single stock info and indices')

	# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
	# selected_stock = st.selectbox('Select dataset for prediction', stocks)


	selected_stock = st.text_input("Enter stock ticker", 'MSFT')

	@st.cache
	def load_data(ticker):
		data = yf.download(ticker, START, TODAY)
		data.reset_index(inplace=True)
		# adding simple and log rate of returns to dataframe
		data['simple_return'] = (data['Close'] / data['Close'].shift(1)) - 1
		data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
		stock_info = yf.Ticker(ticker).info
		return data, stock_info


	data_load_state = st.text('Loading data...')
	data, stock_info = load_data(selected_stock)
	data_load_state.text('Loading data... done!')

	st.markdown('Selected stock: **'+stock_info['longName']+'**.')

	#st.write(stock_info['longName'])

	st.subheader('Raw data (tail)')
	st.write(data.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Open and close prices', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()


 
 
	def plot_rate_of_returns():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['simple_return'], name="simple_return"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['log_return'], name="log_return"))
		fig.update_xaxes(title_text="Date")
		# Set y-axes titles
		fig.update_yaxes(title_text="Rate of return")
		fig.layout.update(title_text='Simple and log returns', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
  
	plot_rate_of_returns()
	
	simple_return_d = data['simple_return'].mean() * 100
	simple_return_a = data['simple_return'].mean() * 250 * 100
	log_return_d = data['log_return'].mean() * 100
	log_return_a = data['log_return'].mean() * 250 * 100 
 
	st.text(f"Mean deily simple return: {round(simple_return_d, 3)} [%]")
	st.text(f"Mean annual simple return: {round(simple_return_a, 3)} [%]")
	st.text(f"Mean deily log return: {round(log_return_d, 3)} [%]")
	st.text(f"Mean annual log return: {round(log_return_a, 3)} [%]")
	

	