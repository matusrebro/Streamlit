import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from prediction_fcns import (
    arma,
    moving_average,
    arma_prediction,
    arima,
    arima_prediction,
)
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

    show_moving_avg = st.checkbox("Show moving average")
    if show_moving_avg:
        ma_order = st.slider("Moving average order [days]", 1, 60, 30)
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
                    go.Scatter(
                        x=data["Date"],
                        y=np.squeeze(ma_filtered_data),
                        name="moving average",
                    )
                )
            fig.layout.update(title_text="Close prices", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        st.subheader("Augmented Dickey-Fuller test - testing for a unit root")
        result = adfuller(data["Close"])
        st.text(f"Test statistic: {result[0]}")
        st.text(f"P-value: {result[1]}")
        st.text(f"Test statistics critical values: {result[4]}")

        st.subheader("Stock N steps ahead prediction")

        models = ["ARMA", "ARIMA"]

        selected_model = st.selectbox("Prediction model", models)

        if selected_model == "ARMA":
            st.text("ARMA model")
            with st.expander("Model structure and prediction algorithm"):
                st.markdown(r"Model output $y(t)$ is given by:")
                st.latex(
                    r"""
                        y(t)=\frac{D(z^{-1})}{A(z^{-1})}\varepsilon(t)
                        """
                )
                st.markdown(
                    r"""where $z^{-1}$ is a discrete lag operator, $\varepsilon(t)$ represents 
                            unknown process dynamics - measurement noises, unmodeled dynamics, etc and $D(z^{-1})$ together with $A(z^{-1})$ are polynomials in the form:"""
                )
                st.latex(
                    r"""
                        \begin{array}{l}
                        A(z^{-1})&=1+a_{1}z^{-1}+\ldots  \\
                        D(z^{-1})&=1+d_{1}z^{-1}+\ldots  
                        \end{array}
                        """
                )
                st.markdown(
                    r"""
                            Proposed parameter estimation method is the recursive least-squares (RLS) identification. Using this type of algorithm, 
                            estimation can be done for every sample period, without using too much of a computational power, 
                            using only current measurements and few past measurements (depending on model orders). 
                            General form of RLS algorithm is as follows:
                            """
                )
                st.latex(
                    r"""
                        \begin{array}{ll}
                        L(t)&=\frac{P(t-1)h(t)}{1+h(t)^TP(t-1)h(t)}  \\
                        P(t)&=\frac{1}{\lambda}(P(t-1)-L(t)h(t)^TP(t-1)) \\
                        \theta(t)&=\theta(t-1)+L(t)e(t)
                        \end{array}
                        """
                )
                st.markdown(
                    r"""
                            where $\theta(t)$ is parameter vector to adapt (estimate), $h(t)$ is a vector of regressors 
                            (containing past measurements and inputs), $e(t)$ is a one step ahead prediction error 
                            (difference between predicted model output and a actual measurement), $L(t)$ is an adaptation gain, 
                            $P(t)$ is a dispersion matrix and $\lambda$ is a forgetting factor
                            """
                )
                st.markdown(r"Parameter vector and regressor vector have the form:")
                st.latex(
                    r"""
                        \begin{array}{ll}
                        \theta^T&= [a_{1} \ldots  d_1 \ldots] \\
                        h^T&=[-y(t-1) \ldots \varepsilon(t-1) \ldots]
                        \end{array}
                        """
                )
                st.markdown(
                    r"""
                            Regressor vector includes past samples of signal $\varepsilon(t)$, which is not known. 
                            This signal is estimated:
                            """
                )
                st.latex(
                    r"""
                        \varepsilon(t)=e(t)=y(t)-h^T\theta
                        """
                )
                st.markdown(
                    r"""
                            Note that the regressor values (specifically $\varepsilon$) are dependent on the identified parameters and 
                            thus it is no longer called a linear regression but a pseudolinear regression.
                            """
                )

            N = st.number_input("Prediction horizon [days]", 1, 60, 5)
            col_pars = st.columns(3)
            na = col_pars[0].number_input("Order of autoregressive part", 1, 10, 4)
            nc = col_pars[1].number_input("Order of moving average part", 1, 10, 4)
            fz = col_pars[2].number_input("Forgetting factor", 0.01, 1.0, 1.0)

            pred_load_state = st.text("Running prediction algorithm...")
            ypredN, theta, thetak, yhat, resid = arma(data["Close"], na, nc, N, fz)
            pred_load_state.text("Running prediction algorithm...done!")

        elif selected_model == "ARIMA":
            st.text("ARIMA model (todo)")
            with st.expander("Model structure and prediction algorithm"):
                st.markdown(r"Model output $y(t)$ is given by:")
                st.latex(
                    r"""
                        y(t)=\frac{D(z^{-1})}{(1-z^{-1})A(z^{-1})}\varepsilon(t)
                        """
                )
                st.markdown(
                    r"""where $z^{-1}$ is a discrete lag operator, $\varepsilon(t)$ represents 
                            unknown process dynamics - measurement noises, unmodeled dynamics, etc and $D(z^{-1})$ together with $A(z^{-1})$ are polynomials in the form:"""
                )
                st.latex(
                    r"""
                        \begin{array}{l}
                        A(z^{-1})&=1+a_{1}z^{-1}+\ldots  \\
                        D(z^{-1})&=1+d_{1}z^{-1}+\ldots  
                        \end{array}
                        """
                )
                st.markdown(r"We can rewrite the model in the form:")
                st.latex(
                    r"""
                        y(t)(1-z^{-1})=\frac{D(z^{-1})}{A(z^{-1})}\varepsilon(t)
                        """
                )
                st.markdown(r"which is:")
                st.latex(
                    r"""
                        y(t)-y(t-1)=\frac{D(z^{-1})}{A(z^{-1})}\varepsilon(t)
                        """
                )
                st.markdown(
                    r"and so the model output is the same as that of ARMA model with model output being is ones step difference:"
                )
                st.latex(
                    r"""
                        \Delta y(t)=\frac{D(z^{-1})}{A(z^{-1})}\varepsilon(t)
                        """
                )
                st.markdown(
                    r"This means that we can use the same RLS estimation of parameters as in the case of ARMA model"
                )
                st.markdown(
                    r"""
                            Proposed parameter estimation method is the recursive least-squares (RLS) identification. Using this type of algorithm, 
                            estimation can be done for every sample period, without using too much of a computational power, 
                            using only current measurements and few past measurements (depending on model orders). 
                            General form of RLS algorithm is as follows:
                            """
                )
                st.latex(
                    r"""
                        \begin{array}{ll}
                        L(t)&=\frac{P(t-1)h(t)}{1+h(t)^TP(t-1)h(t)}  \\
                        P(t)&=\frac{1}{\lambda}(P(t-1)-L(t)h(t)^TP(t-1)) \\
                        \theta(t)&=\theta(t-1)+L(t)e(t)
                        \end{array}
                        """
                )
                st.markdown(
                    r"""
                            where $\theta(t)$ is parameter vector to adapt (estimate), $h(t)$ is a vector of regressors 
                            (containing past measurements and inputs), $e(t)$ is a one step ahead prediction error 
                            (difference between predicted model output and a actual measurement), $L(t)$ is an adaptation gain, 
                            $P(t)$ is a dispersion matrix and $\lambda$ is a forgetting factor
                            """
                )
                st.markdown(r"Parameter vector and regressor vector have the form:")
                st.latex(
                    r"""
                        \begin{array}{ll}
                        \theta^T&= [a_{1} \ldots  d_1 \ldots] \\
                        h^T&=[-\Delta y(t-1) \ldots \varepsilon(t-1) \ldots]
                        \end{array}
                        """
                )
                st.markdown(
                    r"""
                            Regressor vector includes past samples of signal $\varepsilon(t)$, which is not known. 
                            This signal is estimated:
                            """
                )
                st.latex(
                    r"""
                        \varepsilon(t)=e(t)=\Delta y(t)-h^T\theta
                        """
                )
                st.markdown(
                    r"""
                            Note that the regressor values (specifically $\varepsilon$) are dependent on the identified parameters and 
                            thus it is no longer called a linear regression but a pseudolinear regression.
                            """
                )
                st.markdown(
                    r"To get the actual one step ahead prediction we need to sum up (integrate) the model output:"
                )
                st.latex(
                    r"""
                         \hat{y}(t)= h^T\theta + y(t-1)
                        """
                )
            N = st.number_input("Prediction horizon [days]", 1, 60, 5)
            col_pars = st.columns(3)
            na = col_pars[0].number_input("Order of autoregressive part", 1, 10, 4)
            nc = col_pars[1].number_input("Order of moving average part", 1, 10, 4)
            fz = col_pars[2].number_input("Forgetting factor", 0.01, 1.0, 1.0)

            pred_load_state = st.text("Running prediction algorithm...")
            ypredN, theta, thetak, yhat, resid = arima(data["Close"], na, nc, N, fz)
            pred_load_state.text("Running prediction algorithm...done!")
        """
        testcols = st.columns(3)
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
                go.Scatter(
                    x=data["Date"][N:], y=np.squeeze(ypredN)[:-N], name="prediction"
                )
            )
            fig.layout.update(
                title_text=f"Prediction {N}-steps/days ahead",
                xaxis_rangeslider_visible=True,
                yaxis_range=[
                    np.min(data["Close"][N:]) * 0.5,
                    np.max(data["Close"][N:]) * 1.5,
                ],
            )
            st.plotly_chart(fig)

        pred_error = data["Close"][N:] - np.squeeze(ypredN)[:-N]
        plot_prediction()
        # st.write(theta)
        st.subheader(f"Prediction of stock evolution {N} steps ahead")
        cutoff_at = st.number_input(
            "Select day to predict from", na + nc, len(data) - N, 600
        )
        ep = resid[cutoff_at - nc : cutoff_at, 0][::-1]
        theta_at_cutoff = theta[cutoff_at - 1, :]
        if selected_model == "ARMA":
            yp = data["Close"][cutoff_at - na : cutoff_at][::-1]
            prediction = arma_prediction(yp, ep, theta_at_cutoff, N)
        elif selected_model == "ARIMA":
            yp = data["Close"][cutoff_at - na - 1 : cutoff_at][::-1]
            prediction = arima_prediction(yp, ep, theta_at_cutoff, N)

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
                go.Scatter(
                    x=data["Date"][cutoff_at : cutoff_at + N],
                    y=data["Close"][cutoff_at : cutoff_at + N],
                    name="stock_close",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["Date"][cutoff_at : cutoff_at + N],
                    y=np.squeeze(prediction),
                    name="prediction",
                )
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
            fig.add_trace(go.Scatter(x=data["Date"][N:], y=pred_error, name="residual"))
            fig.layout.update(
                title_text=f"Prediction error",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        with st.expander("Show residuals and N-step prediction errors"):
            plot_residuals()
            plot_pred_error()

        def plot_parameters():
            fig = go.Figure()
            for i in range(na + nc):
                if i < na:
                    fig.add_trace(
                        go.Scatter(
                            x=data["Date"],
                            y=np.squeeze(theta[:, i]),
                            name=f"theta_a{i+1}",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=data["Date"],
                            y=np.squeeze(theta[:, i]),
                            name=f"theta_d{i+1-na}",
                        )
                    )
            fig.layout.update(
                title_text="Evolution of model parameters",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig)

        with st.expander("Model parameters"):
            st.text("Final parameters:")
            st.write(thetak)
            plot_parameters()

    else:
        st.warning("No stock data found for selected ticker")
