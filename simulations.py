import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from Diabetes import minimal_model
from glucose_metabolism import glucose_metabolism_app
from adaptive_control import adaptive_control_app
from body_mass_sim import body_mass_app


def app():
    st.title("Simulators of dynamic systems")
    selections = ["Glucose metabolism", "Adaptive control for T1DM", "Body mass change"]
    simulationSelection = st.selectbox("Simulation", options=selections)
    if simulationSelection == "Glucose metabolism":
        glucose_metabolism_app()
    elif simulationSelection == "Adaptive control for T1DM":
        adaptive_control_app()
    elif simulationSelection == "Body mass change":
        body_mass_app()
