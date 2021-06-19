import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from prediction_fcns import arma, moving_average, arma_prediction
from statsmodels.tsa.stattools import adfuller


def prediction_app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.subheader("Stock price forecast")

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

    show_moving_avg = st.checkbox('Show moving average')
    if show_moving_avg:
        ma_order = st.slider('Moving average order [days]', 1, 60, 30)
        ma_filtered_data = moving_average(data["Close"], ma_order)

    if len(data) > 1:
        st.markdown("Selected stock: **" + stock_info["longName"] + "**.")

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            # fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
            fig.add_trace(
                go.Scatter(x=data["Date"], y=data["Close"], name="stock_close")
            )
            if show_moving_avg:
                fig.add_trace(
                    go.Scatter(x=data["Date"], y=np.squeeze(ma_filtered_data), name="moving average")
                )
            fig.layout.update(
                title_text="Close prices", xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig)

        plot_raw_data()

        st.subheader('Augmented Dickey-Fuller test - testing for a unit root')
        result = adfuller(data["Close"])
        st.text(f'Test statistic: {result[0]}')
        st.text(f'P-value: {result[1]}')
        st.text(f'Test statistics critical values: {result[4]}')

        st.subheader("Stock N steps ahead prediction")

        st.text("ARMA model")
        N = st.number_input("Prediction horizon [days]", 1, 60, 5)
        col_pars = st.beta_columns(3)
        na = col_pars[0].number_input("Order of autoregressive part", 1, 10, 4)
        nc = col_pars[1].number_input("Order of moving average part", 1, 10, 4)
        fz = col_pars[2].number_input("Forgetting factor", 0.01, 1.0, 1.0)

        pred_load_state = st.text("Running prediction algorithm...")
        ypredN, theta, thetak, yhat, resid = arma(data["Close"], na, nc, N, fz)
        pred_load_state.text("Running prediction algorithm...done!")

        """
        testcols = st.beta_columns(3)
        testcols[0].write(data["Close"])
        testcols[1].write(data["Close"][N:])
        testcols[2].write(ypredN)
        st.write(theta)
        st.write(resid)
        """
        def plot_prediction():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data["Date"][N:], y=data["Close"][N:], name="stock_close")
            )
            fig.add_trace(
                go.Scatter(x=data["Date"][N:], y=np.squeeze(ypredN)[:-N], name="prediction")
            )
            fig.layout.update(
                title_text=f"Prediction {N}-steps/days ahead",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)
        pred_error = data["Close"][N:]-np.squeeze(ypredN)[:-N]
        plot_prediction()
        # st.write(theta)
        st.subheader(f'Prediction of stock evolution {N} steps ahead')
        cutoff_at = st.number_input('Select day to predict from', na+nc, len(data)-N, 600)
        yp = data["Close"][cutoff_at-na:cutoff_at][::-1]
        ep = resid[cutoff_at-nc:cutoff_at, 0][::-1]
        theta_at_cutoff = theta[cutoff_at-1, :]
        prediction = arma_prediction(yp, ep, theta_at_cutoff, N)
        
        """
        st.text('yp')
        st.write(yp)
        st.text('ep')
        st.write(ep)
        st.text('theta')
        st.write(theta_at_cutoff)
        st.write(data["Close"][cutoff_at:cutoff_at+N])
        st.write(prediction)
        """




        def plot_prediction_after_cutoff():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data["Date"][cutoff_at:cutoff_at+N], y=data["Close"][cutoff_at:cutoff_at+N], name="stock_close")
            )
            fig.add_trace(
                go.Scatter(x=data["Date"][cutoff_at:cutoff_at+N], y=np.squeeze(prediction), name="prediction")
            )
            fig.layout.update(
                title_text=f"Prediction {N}-steps/days ahead",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig)
        
        plot_prediction_after_cutoff()

        def plot_residuals():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data["Date"], y=np.squeeze(resid), name="residual")
            )
            fig.layout.update(
                title_text=f"Residual",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        def plot_pred_error():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data["Date"][N:], y=pred_error, name="residual")
            )
            fig.layout.update(
                title_text=f"Prediction error",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        with st.beta_expander("Show residuals and N-step prediction errors"):
            plot_residuals()
            plot_pred_error()
            
        def plot_parameters():
            fig = go.Figure()
            for i in range(na + nc):
                if i < na:
                    fig.add_trace(
                            go.Scatter(
                                x=data["Date"], y=np.squeeze(theta[:, i]), name=f"theta_a{i+1}"
                            )                     
                    )
                else:
                    fig.add_trace(
                            go.Scatter(
                                x=data["Date"], y=np.squeeze(theta[:, i]), name=f"theta_d{i+1-na}"
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
