import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from Diabetes import minimal_model

def basal_metabolic_rate(par, body_mass, height, age):
    a, b, c, d = par
    return a*body_mass + b*height + c*age + d
    

def body_mass_app():
    st.header("Simulation of body mass change")
    
    st.markdown(r"Although there are several empiric calculators for basal metabolic rate, the general formula is as follows:")
    st.latex(r"P = a m + b h + c a + d")
    st.markdown(
        r"""
    where $P$ [kcal/day] is basal energy expenditure (at rest) per day, $m$ [kg] is body mass, $h$ [cm] is height and $a$ [years] is an age. 
    Parameters of this model are $a$ [kcal/day per kg], $b$ [kcal/day per cm], $c$ [kcal/day per year] and $d$ [kcal/day]. 
    These parameters vary from calculator to calculator and may be also dependent on sex of the subject
    """
    )
    
    parameter_sets = [
        "original Harris–Benedict",
        "revised Harris–Benedict",
        "Mifflin St Jeor"
    ]
    
    parameters = {
        "original Harris–Benedict": ([13.7516, 5.0033, -6.7550, 66.4730], [9.5634, 1.8496, -4.6756, 655.0955]),
        "revised Harris–Benedict": ([13.397, 4.799, -5.677, 88.362], [9.247, 3.098, -4.330, 447.593]),
        "Mifflin St Jeor": ([10.0, 6.25, -5.0, 5], [10.0, 6.25, -5.0, -161])
    }
    
    selected_parset = st.selectbox("Select parameter set", parameter_sets)
    selected_par = parameters[selected_parset]
    
    
    
    
    col_metab_rate_pars = st.columns(2)
    
    col_metab_rate_pars[0].text("Male")
    col_metab_rate_pars[0].latex(
            r"""
            \begin{array}{lll}
            a &= """
    + str(selected_par[0][0])
    + r"""& \textrm{[kcal/day per kg]} \\
            b &= """
    + str(selected_par[0][1])
    + r"""& \textrm{[kcal/day per cm]} \\
            c &= """
    + str(selected_par[0][2])
    + r"""& \textrm{[kcal/day per year]} \\
            d &= """
    + str(selected_par[0][3])
    + r"""& \textrm{[kcal/day]} \\
            \end{array}
            """
        
    )
    
    col_metab_rate_pars[1].text("Female")
    col_metab_rate_pars[1].latex(
            r"""
            \begin{array}{lll}
            a &= """
    + str(selected_par[1][0])
    + r"""& \textrm{[kcal/day per kg]} \\
            b &= """
    + str(selected_par[1][1])
    + r"""& \textrm{[kcal/day per cm]} \\
            c &= """
    + str(selected_par[1][2])
    + r"""& \textrm{[kcal/day per year]} \\
            d &= """
    + str(selected_par[1][3])
    + r"""& \textrm{[kcal/day]} \\
            \end{array}
            """
        
    )
    
    col_metab_rate = st.columns(4)
    
    sex = col_metab_rate[0].radio("Sex", ["Male", "Female"])
    body_mass = col_metab_rate[1].number_input("Bodyweight [kg]", 0.0, 300.0, 70.0, 0.1, key=1)
    height = col_metab_rate[2].number_input("Height [cm]", 0.0, 300.0, 170.0, 0.1, key=2)
    age = col_metab_rate[3].number_input("Age [years]", 0, 150, 25, 1, key=3)
    
    if sex == "Male":
        metab_rate_par = selected_par[0]
    else:
        metab_rate_par = selected_par[1]
    
    
    P = basal_metabolic_rate(metab_rate_par, body_mass, height, age)
    
    
    st.columns(3)[1].markdown(f'#### P = **{str(round(P, 2))}** kcal/day')
    