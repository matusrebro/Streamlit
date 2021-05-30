import landing_page
import stonks
import stonks_portfolio
import simulations
import prediction
import streamlit as st

st.set_page_config(
    page_title="Rebrr app",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="auto",
)

PAGES = {
    "Landing page": landing_page,
    "Stock info": stonks,
    "Stocks portfolio": stonks_portfolio,
    "Simulators": simulations,
    "Prediction showcase": prediction,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
