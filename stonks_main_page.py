import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from stonks import stonks_app
from stonks_portfolio import portfolio_app


def app():
    st.title("Stocks info and portfolio experiments")
    selections = ["Single stock info and indices", "Portfolio metrics"]
    simulationSelection = st.selectbox("Select stocks app", options=selections)
    if simulationSelection == "Single stock info and indices":
        stonks_app()
    elif simulationSelection == "Portfolio metrics":
        portfolio_app()
