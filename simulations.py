import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from Diabetes import minimal_model

def app():
    st.title('Simulators of dynamic systems')
    
    st.header('Simulations of glucose metabolism')
    
    optionsSim = ['IVGTT simulation', 'OGTT simulation']
    
    simulationPage = st.selectbox("Simulation", options = optionsSim)
    
    if simulationPage == 'IVGTT simulation':
        st.subheader("IVGTT simulator")
        options = ['normal', 'obese', 't2dm']
        pars = st.selectbox("Model parameters", options = options)
        
        basal_cols = st.beta_columns(2)
        
        Gb = basal_cols[0].number_input("Basal glucose concentration [mmol/L]", min_value=3, max_value=10, value=5)
        Ib = basal_cols[1].number_input("Basal insulin concentration [mU/L]", min_value=1, max_value=40, value=6)
        # glycemic response vs. glycemic index 
        # initialize model for normal subject
        model = minimal_model.iv(Gb, Ib, parameters = pars)
        
        input_cols = st.beta_columns(2)
        
        glucose_dose = input_cols[0].slider("Glucose bolus dose [g/kg]", min_value=0.1, max_value=1.0, value=0.3)
        glucose_bolus_min = input_cols[0].slider("Duration of glucose administration [min]", min_value=0.5, max_value=5.0, value=2.0)

        insulin_dose = input_cols[1].slider("Insulin bolus dose [mU/kg]", min_value=1, max_value=50, value=20)
        insulin_bolus_min = input_cols[1].slider("Duration of insulin administration [min]", min_value=0.5, max_value=10.0, value=5.0)
        insulin_dosage_time = input_cols[1].slider("Time of insulin administration (from the start) [min]", min_value=1, max_value=40, value=20)
        
        t, G, I  = model.ivgtt(glucose_dose, glucose_bolus_min, insulin_dose, insulin_bolus_min, insulin_dosage_time, plot=False)
        
        def plot_simulation_results():
            #fig = go.Figure()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=t, y=G, name="Glucose conc. [mmol/L]"), secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=I, name="Insulin conc. [mU/L]"), secondary_y=True)
            fig.layout.update(title_text='IVGTT simulation', xaxis_rangeslider_visible=False)
            fig.update_xaxes(title_text="time [min]")
            fig.update_yaxes(title_text="Glucose conc. [mmol/L]", secondary_y=False)
            fig.update_yaxes(title_text="Insulin conc. [mU/L]", secondary_y=True)
            st.plotly_chart(fig)

        plot_simulation_results()
        
    elif simulationPage == 'OGTT simulation':
        st.subheader("OGTT simulator")
        
        options = ['normal', 't2dm']
        pars = st.selectbox("Model parameters", options = options)
        
        basal_cols = st.beta_columns(2)
        
        Gb = basal_cols[0].number_input("Basal glucose concentration [mmol/L]", min_value=3, max_value=10, value=5)
        Ib = basal_cols[1].number_input("Basal insulin concentration [mU/L]", min_value=1, max_value=40, value=6)
        # glycemic response vs. glycemic index 
        # initialize model for normal subject
        model = minimal_model.oral(Gb, Ib, parameters = pars)
        
        glucose  = st.slider("Glucose amount [g]", min_value=1, max_value=100, value=50)
        BW  = st.slider("Bodyweight of a subject [g]", min_value=50, max_value=120, value=70)
        # simulation for normal subject - high glycemic index
        
        gly_index_options = ['glucose', 'low', 'medium', 'high']
        gly_index = st.selectbox(label= "Glycemic index", options = gly_index_options)
        
        t, G, I = model.ogtt(glucose, BW, gly_index, plot=False)
        
        def plot_simulation_results():
            #fig = go.Figure()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=t, y=G, name="Glucose conc. [mmol/L]"), secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=I, name="Insulin conc. [mU/L]"), secondary_y=True)
            fig.layout.update(title_text='OGTT simulation', xaxis_rangeslider_visible=False)
            fig.update_xaxes(title_text="time [min]")
            fig.update_yaxes(title_text="Glucose conc. [mmol/L]", secondary_y=False)
            fig.update_yaxes(title_text="Insulin conc. [mU/L]", secondary_y=True)
            st.plotly_chart(fig)

        plot_simulation_results() 
        