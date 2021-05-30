import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from prediction_fcns import arma


def app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title("Single stock info and indices")

    selected_stock = st.text_input("Enter stock ticker", "MSFT")

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        # adding simple and log rate of returns to dataframe
        data["simple_return"] = (data["Close"] / data["Close"].shift(1)) - 1
        data["log_return"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
        stock_info = yf.Ticker(ticker).info
        return data, stock_info

    data_load_state = st.text("Loading data...")
    data, stock_info = load_data(selected_stock)
    data_load_state.text("Loading data... done!")

    if len(data) > 1:
        st.markdown("Selected stock: **" + stock_info["longName"] + "**.")

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
            fig.add_trace(
                go.Scatter(x=data["Date"], y=data["Close"], name="stock_close")
            )
            fig.layout.update(
                title_text="Open and close prices", xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig)

        plot_raw_data()

        st.subheader("Stock N steps ahead prediction")

        st.text("ARMA model")
        N = st.number_input("Prediction horizon [days]", 1, 60, 5)
        col_pars = st.beta_columns(3)
        na = col_pars[0].number_input("Order of autoregressive part", 1, 10, 4)
        nc = col_pars[1].number_input("Order of moving average part", 1, 10, 4)
        fz = col_pars[2].number_input("Forgetting factor", 0.01, 1.0, 1.0)

        pred_load_state = st.text("Running prediction algorithm...")
        ypredN, theta, thetak = arma(data["Close"], na, nc, N, fz)
        pred_load_state.text("Running prediction algorithm...done!")

        def plot_prediction():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data["Date"], y=data["Close"], name="stock_close")
            )
            fig.add_trace(
                go.Scatter(x=data["Date"], y=np.squeeze(ypredN), name="prediction")
            )
            fig.layout.update(
                title_text=f"Prediction {N}-steps/days ahead",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        plot_prediction()
        # st.write(theta)

        def plot_parameters():
            fig = go.Figure()
            for i in range(na + nc):
                fig.add_trace(
                    go.Scatter(
                        x=data["Date"], y=np.squeeze(theta[:, i]), name=f"theta_{i+1}"
                    )
                )
            fig.layout.update(
                title_text="Evolution of model parameters",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        with st.beta_expander("Model parameters"):
            st.text("Final parameters:")
            st.write(thetak)
            plot_parameters()

    else:
        st.warning("No stock data found for selected ticker")
