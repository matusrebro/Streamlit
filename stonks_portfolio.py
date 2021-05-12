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

def app():
    st.title('Portfolio metrics')
    
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
 
    @st.cache(allow_output_mutation=True)
    def get_data():
        return []
	
    col_add = st.beta_columns(2)
    ticker_to_add = col_add[0].text_input("add ticker")
    if col_add[1].button("add to list"):
        get_data().append(ticker_to_add)

    if "" in get_data():
        get_data().remove("")
    else: 
        pass
    
    col_mod = st.beta_columns(2)
    newlist = col_mod[0].multiselect("selected tickers: ", get_data(), get_data())

    if col_mod[1].button("update list"):
        get_data().clear()
        get_data().extend(newlist)
    
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

    tickers_added = len(get_data()) > 0

    if tickers_added:
        data_load_state = st.text('Loading ticker data...')
        stocks_info = load_ticker_info(get_data())
        data_load_state.text('Loading data... done!')
    
        weights = []
        for idx, _ in enumerate(stocks_info):
            weights.append(1/len(stocks_info))
    
        col_mod_weight = st.beta_columns(2)
        selected_ticker_idx = col_mod_weight[0].selectbox(label="Modify stock weight", 
                                                        options=[index for index, _ in enumerate(get_data())], 
                                                        format_func=lambda i: get_data()[i])
        
        
        weight_to_update = col_mod_weight[1].number_input(label="Enter new weight", 
                                                        min_value=0.0, 
                                                        max_value=1.0,
                                                        value=weights[selected_ticker_idx])
        weights[selected_ticker_idx] = weight_to_update
        balance = st.button("click to autobalance")
        if balance:
            correction = (1.0 - np.sum(weights))/(len(weights) - 1)
            for idx, _ in enumerate(weights):
                if idx != selected_ticker_idx:
                    weights[idx] = weights[idx] + correction
                    
        if np.sum(weights) != 1.0:
            st.warning("Weights are unbalanced")
            
        column_names = ['Ticker', 'Company name', "Weight"]
        stocks_basic_data = pd.DataFrame(columns=column_names)    
        for idx, stock_info in enumerate(stocks_info):
            if len(stock_info) > 1:
                df_to_append = pd.DataFrame([[stock_info['symbol'], stock_info['longName'], weights[idx]]], columns=column_names)
            else:
                df_to_append = pd.DataFrame([[get_data()[idx],"no data"], weights[idx]], columns=column_names)
            # st.write(df_to_append)    
            stocks_basic_data = stocks_basic_data.append(df_to_append)
            # stocks_name.append(stock_info['longName'])
        

        st.text("Selected stocks:")
        st.write(stocks_basic_data)

        data_load_state = st.text('Loading ticker datasets...')
        stocks_data = load_data(get_data())
        data_load_state.text('Loading data... done!')

        column_names = list(get_data())
        column_names.append("Date")
        stocks_values = pd.DataFrame(columns=column_names)
        
        if len(get_data()) > 0:
            stocks_values['Date'] = stocks_data[0]['Date']   

        for idx, ticker in enumerate(get_data()):
            stocks_values[ticker] = stocks_data[idx]['Close']
        
        stocks_values = stocks_values.set_index('Date')
        
        normalize = st.checkbox("Normalize data", value=False)
        
        def plot_raw_data():
            fig = go.Figure()
            for idx, stock_data in enumerate(stocks_data):
                if normalize:
                    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close']/stock_data['Close'].iloc[0]*100, name=str(get_data()[idx])))
                else:
                    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name=str(get_data()[idx])))
            fig.layout.update(title_text='Close prices of selected tickers', xaxis_rangeslider_visible=True)
            
            st.plotly_chart(fig)
            
        plot_raw_data()
        
        stocks_returns = (stocks_values / stocks_values.shift(1)) - 1
        portfolio_returns = np.dot(stocks_returns, np.array(weights))
        
        def plot_portfolio_returns():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stocks_values.index, y=portfolio_returns))
            fig.layout.update(title_text='Portfolio returns', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_portfolio_returns()
        
        annual_stock_returns = stocks_returns.mean()
        st.text("Annual stock returns:")
        st.write(annual_stock_returns*250*100)
        
        portfolio_returns_d = np.dot(annual_stock_returns*100, np.array(weights))
        portfolio_returns_a = np.dot(annual_stock_returns*250*100, np.array(weights))
        
        
        st.text(f"Mean deily portfolio returns: {round(portfolio_returns_d, 3)} [%]")
        st.text(f"Mean annual portfolio returns: {round(portfolio_returns_a, 3)} [%]")
    
        cov_matrix = stocks_returns.cov()
        corr_matrix = stocks_returns.corr()
        
        st.text("Covariance matrix:")
        st.write(cov_matrix)
        st.text("Correlation matrix:")
        st.write(corr_matrix)
        
        portfolio_var_annual = np.dot(np.array(weights).T, np.dot(cov_matrix * 250, np.array(weights)))
        
        st.text(f"Annual portfolio volatility: {round(portfolio_var_annual ** 0.5 * 100, 3)} [%]")
        
        if len(get_data()) > 1:
            st.subheader("Markowitz portfolio analysis")
            weight_combin_no = 1000
            
            weights_rand_comb_str = []
            portfolio_returns_a_arr = np.zeros(weight_combin_no)
                                               
            portfolio_var_annual_arr = np.zeros(weight_combin_no)
                                                
            for i in range(weight_combin_no):
                weights_rand = np.random.random(len(get_data()))
                weights_rand /= np.sum(weights_rand)
                weights_rand_l = weights_rand.tolist()
                weights_rand_l = ['{:.2f}'.format(x) for x in weights_rand_l]
                weights_rand_comb_str.append(','.join(weights_rand_l))
                portfolio_returns_a_arr[i] = np.dot(annual_stock_returns*250*100, np.array(weights_rand))
                portfolio_var_annual_arr[i] = np.dot(np.array(weights_rand).T, np.dot(cov_matrix * 250, np.array(weights_rand)))
            
            column_names = ['returns', 'variance', 'weights']
            df_Markowitz = pd.DataFrame(columns=column_names)
            
            df_Markowitz['returns'] = portfolio_returns_a_arr
            df_Markowitz['variance'] = portfolio_var_annual_arr
            df_Markowitz['weights'] = weights_rand_comb_str
            # , text=df_Markowitz['weights']
            
            # lets minimize variance
            def fcn_obj_weights(weights, cov_matrix):
                if np.abs(np.sum(weights)-1) > 1e-5:
                    return 1e6
                else:
                    return np.dot(weights.T, np.dot(cov_matrix * 250, weights))

            run_simplex = st.button("Find optimal weights (minimizing variance)")
            opt_weights = None
            if run_simplex:
                w0 = np.ones(len(get_data())) * 1/len(get_data())
                data_load_state = st.text('Running Simplex optimization...')
                res = minimize(fcn_obj_weights, w0, args=(cov_matrix,), method='Nelder-Mead', options={'fatol': 0.0000001})
                data_load_state.text('...Done')
                st.text("Optimaal weights:")
                opt_weights = res.x
                df_optimal_weights = pd.DataFrame(columns=get_data())
                df_optimal_weights = df_optimal_weights.append(pd.DataFrame([opt_weights], columns=get_data()))
                st.write(df_optimal_weights)
                opt_variance = fcn_obj_weights(opt_weights, cov_matrix)
                opt_returns = np.dot(annual_stock_returns*250*100, np.array(opt_weights))
                
            def plot_Markowitz(opt_weights=None):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_Markowitz['variance'], 
                                         y=df_Markowitz['returns'], 
                                         mode='markers', 
                                         text=df_Markowitz['weights'],
                                         name='Set of random weights'))
                if opt_weights is not None:
                    fig.add_trace(go.Scatter(x=[opt_variance, ], 
                                             y=[opt_returns, ], 
                                             mode='markers', 
                                             fillcolor='red',
                                             name='Optimal weights'))
                fig.layout.update(title_text='Markowitz plot', xaxis_rangeslider_visible=False)
                fig.update_xaxes(title_text="variance [-]")
                fig.update_yaxes(title_text="returns [%]")
                st.plotly_chart(fig)
              
            plot_Markowitz(opt_weights)
            