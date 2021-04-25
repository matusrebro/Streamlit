import stonks
import stonks_portfolio
import simulations
import streamlit as st

PAGES = {
    "Stock info": stonks,
    "Stocks portfolio": stonks_portfolio,
    "Simulators": simulations
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
