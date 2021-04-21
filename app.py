import stonks, stonks_portfolio
import streamlit as st

PAGES = {
    "Stonks": stonks,
    "Stonks portoflio": stonks_portfolio
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()