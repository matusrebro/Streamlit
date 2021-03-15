import stonks
import app2
import streamlit as st

PAGES = {
    "Stonks": stonks,
    "TODO": app2
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
