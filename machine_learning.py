import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from Diabetes import minimal_model
from prediction import prediction_app
from classification import classification_app


def app():
    st.title("Machine learning algorithms showcase")
    selections = ["Stock market prediction/forecast", "Classification"]
    simulationSelection = st.selectbox("Select machine learning demo", options=selections)
    if simulationSelection == "Stock market prediction/forecast":
        prediction_app()
    elif simulationSelection == "Classification":
        classification_app()
