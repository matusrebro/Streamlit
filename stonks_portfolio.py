import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import sys
from scipy.optimize import minimize

# from scipy.optimize import least_squares

# Initialization of session state
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []


def portfolio_app():
    st.title("Portfolio metrics")

    START_default = date(2015, 1, 1)
    TODAY = date.today().strftime("%Y-%m-%d")
    START = st.date_input("Enter starting date", START_default)
    # code before session state api
    # @st.cache(allow_output_mutation=True)
    # def get_data():
    #     return []

    col_add = st.columns(2)
    ticker_to_add = col_add[0].text_input("add ticker")
    col_add[1].empty()
    
    # code before session state api
    # if col_add[1].button("add to list"):
    #     get_data().append(ticker_to_add)

    st.session_state['tickers'].append(ticker_to_add)
    st.session_state['tickers'] = list(set(st.session_state['tickers']))

    if "" in st.session_state['tickers']:
        st.session_state['tickers'].remove("")
    else:
        pass

    # code before session state api
    # if "" in get_data():
    #     get_data().remove("")
    # else:
    #     pass

    col_mod = st.columns(2)
    newlist = col_mod[0].multiselect("selected tickers: ", st.session_state['tickers'], st.session_state['tickers'])
    st.session_state['tickers'] = newlist[:]
    
    # code before session state api
    # if col_mod[1].button("update list"):
    #     get_data().clear()
    #     get_data().extend(newlist)

    @st.cache
    def load_ticker_info(ticker_list):
        stocks_info = []
        for ticker in ticker_list:
            ticker_info = yf.Ticker(ticker).info
            stocks_info.append(ticker_info)
            # st.write(ticker_info)
        return stocks_info

    @st.cache
    def load_data(ticker_list):
        stocks_data = []
        for ticker in ticker_list:
            ticker_data = yf.download(ticker, START, TODAY)
            ticker_data.reset_index(inplace=True)
            stocks_data.append(ticker_data)
        return stocks_data

    tickers_added = len(st.session_state['tickers']) > 0

    if tickers_added:
        data_load_state = st.text("Loading ticker data...")
        stocks_info = load_ticker_info(st.session_state['tickers'])
        data_load_state.text("Loading data... done!")

        weights = []
        for idx, _ in enumerate(stocks_info):
            weights.append(1 / len(stocks_info))

        col_mod_weight = st.columns(2)
        selected_ticker_idx = col_mod_weight[0].selectbox(
            label="Modify stock weight",
            options=[index for index, _ in enumerate(st.session_state['tickers'])],
            format_func=lambda i: st.session_state['tickers'][i],
        )

        weight_to_update = col_mod_weight[1].number_input(
            label="Enter new weight",
            min_value=0.0,
            max_value=1.0,
            value=weights[selected_ticker_idx],
        )
        weights[selected_ticker_idx] = weight_to_update
        balance = st.button("click to autobalance")
        if balance:
            correction = (1.0 - np.sum(weights)) / (len(weights) - 1)
            for idx, _ in enumerate(weights):
                if idx != selected_ticker_idx:
                    weights[idx] = weights[idx] + correction

        if np.sum(weights) != 1.0:
            st.warning("Weights are unbalanced")

        column_names = ["Ticker", "Company name", "Weight"]
        stocks_basic_data = pd.DataFrame(columns=column_names)
        for idx, stock_info in enumerate(stocks_info):
            if len(stock_info) > 1:
                df_to_append = pd.DataFrame(
                    [[stock_info["symbol"], stock_info["longName"], weights[idx]]],
                    columns=column_names,
                )
            else:
                df_to_append = pd.DataFrame(
                    [[st.session_state['tickers'][idx], "no data"], weights[idx]], columns=column_names
                )
            # st.write(df_to_append)
            stocks_basic_data = stocks_basic_data.append(df_to_append)
            # stocks_name.append(stock_info['longName'])

        st.text("Selected stocks:")
        st.write(stocks_basic_data)

        data_load_state = st.text("Loading ticker datasets...")
        stocks_data = load_data(st.session_state['tickers'])
        data_load_state.text("Loading data... done!")

        column_names = list(st.session_state['tickers'])
        column_names.append("Date")
        stocks_values = pd.DataFrame(columns=column_names)

        if len(st.session_state['tickers']) > 0:
            stocks_values["Date"] = stocks_data[0]["Date"]

        for idx, ticker in enumerate(st.session_state['tickers']):
            stocks_values[ticker] = stocks_data[idx]["Close"]

        stocks_values = stocks_values.set_index("Date")

        normalize = st.checkbox("Normalize data", value=False)

        def plot_raw_data():
            fig = go.Figure()
            for idx, stock_data in enumerate(stocks_data):
                if normalize:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data["Date"],
                            y=stock_data["Close"] / stock_data["Close"].iloc[0] * 100,
                            name=str(st.session_state['tickers'][idx]),
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data["Date"],
                            y=stock_data["Close"],
                            name=str(st.session_state['tickers'][idx]),
                        )
                    )
            fig.layout.update(
                title_text="Close prices of selected tickers",
                xaxis_rangeslider_visible=True,
            )

            st.plotly_chart(fig)

        plot_raw_data()

        stocks_returns = (stocks_values / stocks_values.shift(1)) - 1
        portfolio_returns = np.dot(stocks_returns, np.array(weights))

        def plot_portfolio_returns():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stocks_values.index, y=portfolio_returns))
            fig.layout.update(
                title_text="Portfolio returns", xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig)

        plot_portfolio_returns()

        annual_stock_returns = stocks_returns.mean()
        st.text("Annual stock returns:")
        st.write(annual_stock_returns * 250 * 100)

        portfolio_returns_d = np.dot(annual_stock_returns * 100, np.array(weights))
        portfolio_returns_a = np.dot(
            annual_stock_returns * 250 * 100, np.array(weights)
        )

        st.text(f"Mean deily portfolio returns: {round(portfolio_returns_d, 3)} [%]")
        st.text(f"Mean annual portfolio returns: {round(portfolio_returns_a, 3)} [%]")

        cov_matrix = stocks_returns.cov()
        corr_matrix = stocks_returns.corr()

        st.text("Covariance matrix:")
        st.write(cov_matrix)
        st.text("Correlation matrix:")
        st.write(corr_matrix)

        portfolio_var_annual = np.dot(
            np.array(weights).T, np.dot(cov_matrix * 250, np.array(weights))
        )

        st.text(
            f"Annual portfolio volatility: {round(portfolio_var_annual ** 0.5 * 100, 3)} [%]"
        )

        if len(st.session_state['tickers']) > 1:
            st.subheader("Markowitz portfolio analysis")
            weight_combin_no = 1000

            weights_rand_comb_str = []
            portfolio_returns_a_arr = np.zeros(weight_combin_no)

            portfolio_var_annual_arr = np.zeros(weight_combin_no)

            for i in range(weight_combin_no):
                weights_rand = np.random.random(len(st.session_state['tickers']))
                weights_rand /= np.sum(weights_rand)
                weights_rand_l = weights_rand.tolist()
                weights_rand_l = ["{:.2f}".format(x) for x in weights_rand_l]
                weights_rand_comb_str.append(",".join(weights_rand_l))
                portfolio_returns_a_arr[i] = np.dot(
                    annual_stock_returns * 250 * 100, np.array(weights_rand)
                )
                portfolio_var_annual_arr[i] = np.dot(
                    np.array(weights_rand).T,
                    np.dot(cov_matrix * 250, np.array(weights_rand)),
                )

            column_names = ["returns", "variance", "weights"]
            df_Markowitz = pd.DataFrame(columns=column_names)

            df_Markowitz["returns"] = portfolio_returns_a_arr
            df_Markowitz["variance"] = portfolio_var_annual_arr
            df_Markowitz["weights"] = weights_rand_comb_str
            # , text=df_Markowitz['weights']

            # lets minimize variance
            def fcn_obj_weights(weights, cov_matrix):
                if np.abs(np.sum(weights) - 1) > 1e-5:
                    return 1e6
                else:
                    return np.dot(weights.T, np.dot(cov_matrix * 250, weights))

            run_simplex = st.button("Find optimal weights (minimizing variance)")
            opt_weights = None
            if run_simplex:
                w0 = np.ones(len(st.session_state['tickers'])) * 1 / len(st.session_state['tickers'])
                data_load_state = st.text("Running Simplex optimization...")
                res = minimize(
                    fcn_obj_weights,
                    w0,
                    args=(cov_matrix,),
                    method="Nelder-Mead",
                    options={"fatol": 0.0000001},
                )
                data_load_state.text("...Done")
                st.text("Optimal weights:")
                opt_weights = res.x
                df_optimal_weights = pd.DataFrame(columns=st.session_state['tickers'])
                df_optimal_weights = df_optimal_weights.append(
                    pd.DataFrame([opt_weights], columns=st.session_state['tickers'])
                )
                st.write(df_optimal_weights)
                opt_variance = fcn_obj_weights(opt_weights, cov_matrix)
                opt_returns = np.dot(
                    annual_stock_returns * 250 * 100, np.array(opt_weights)
                )

            def plot_Markowitz(opt_weights=None):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_Markowitz["variance"],
                        y=df_Markowitz["returns"],
                        mode="markers",
                        text=df_Markowitz["weights"],
                        name="Set of random weights",
                    )
                )
                if opt_weights is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[
                                opt_variance,
                            ],
                            y=[
                                opt_returns,
                            ],
                            mode="markers",
                            fillcolor="red",
                            name="Optimal weights",
                        )
                    )
                fig.layout.update(
                    title_text="Markowitz plot", xaxis_rangeslider_visible=False
                )
                fig.update_xaxes(title_text="variance [-]")
                fig.update_yaxes(title_text="returns [%]")
                st.plotly_chart(fig)

            plot_Markowitz(opt_weights)
