import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go


def app():
	START = "2015-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")

	st.title('Mater pochovat')

	# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
	# selected_stock = st.selectbox('Select dataset for prediction', stocks)


	selected_stock = st.text_input("Vyber stonk", 'GOOG')

	@st.cache
	def load_data(ticker):
		data = yf.download(ticker, START, TODAY)
		data.reset_index(inplace=True)
		stock_info = yf.Ticker(ticker).info
		return data, stock_info


	data_load_state = st.text('Loading data...')
	data, stock_info = load_data(selected_stock)
	data_load_state.text('Loading data... done!')

	st.markdown('Vybrau si **'+stock_info['longName']+'**.')

	#st.write(stock_info['longName'])

	st.subheader('Raw data')
	st.write(data.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()
