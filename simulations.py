import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from Diabetes import minimal_model

def app():
    st.title('Simulators of dynamic systems')
    selections = ['Glucose metabolism']
    simulationSelection = st.selectbox("Simulation", options=selections)
    if simulationSelection == 'Glucose metabolism':
        st.header('Simulations of glucose metabolism')
        optionsSim = ['IVGTT simulation', 
                      'OGTT simulation', 
                      'Glucose clamp simulation']
        simulationPage = st.selectbox("Select specific simulator", 
                                      options=optionsSim)
        
        if simulationPage == 'IVGTT simulation':
            st.subheader('Insulin-modified intravenous glucose tolerance test')
            with st.beta_expander("Model equations"):
                st.text('Minimal (Bergman) model is given by three differential equations:')
                st.latex(r'''
                \frac{dG(t)}{dt}=-k_xX(t)G(t)-\frac{1}{T_G}(G(t)-G_b)+\frac{1}{V_G}Ra_G(t)  \\
                \frac{dX(t)}{dt}=-\frac{1}{T_x}X(t)+\frac{1}{T_x}(I(t)-I_b) \\
                \frac{dI(t)}{dt}=-\frac{1}{T_i}(I(t)-I_b)+S(t) +\frac{1}{V_I}Ra_I(t)
                ''')
                st.text('Initial equilibrium state is given by:')
                st.latex(r'G(0)=G_b \quad X(0)=0 \quad I(0)=I_b')

                st.markdown(r'''
                where $G(t)$ [mmol/L] is glycemia, $G_b$ [mmol/L] is its basal value, signal $X(t)$ [mU/L] represents so called remote insulin and $I(t)$ [mU/L] 
                is plasma insulin and $I_b$ [mU/L] its basal value. System inputs are $Ra_G(t)$ [mmol/kg/min] - glucose infusion 
                (or more general - glucose rate of appearance which can be also caused by oral glucose intake) and 
                $Ra_I(t)$ [mU/kg/min] is insulin infusion. Parameters $V_G$ and $V_I$ [L/kg] represent distribution volumes of glucose a insulin in blood. 
                Last signal $S(t)$ [mU/L/min] represents pancreas insulin secretion.

                Other model parameters are $k_x$ [L/(mU.min)] - insulin sensitivity index, $T_G$ [min] - glucose compartment time constant, 
                its inverse is often called as glucose effectiveness (insulin independent index of glucose lowering capacity). Other time constants $T_x$, $T_i$ [min] 
                govern dynamics of remote insulin and insulin compartments.
                ''')
                st.markdown(r'Secretion $S(t)$ is given by')
                st.latex(r'''
                S(t)=\left\{\begin{array}{ll}
                S^{+}(t)  & ;S^{+}(t) \geq 0 \\
                0 & ;\textrm{else} 
                \end{array} \right.
                ''')
                st.markdown(r'where  $S^{+}(t)$ has PD controller-like structure:')
                st.latex(r'''
                S^{+}(t)= K_{G1}(G(t)-G_b) + K_{G2} \frac{s}{T_Ps+1} (G(t)-G_b)
                ''')
                st.markdown(r'''
                Here the parameters are: $K_{G1}$ [mU/min per mmol] is proportional gain (pancreas sensitivity to glycemia deviation from equilibrium), 
                $K_{G2}$ [mU per mmol] is derivative gain (pancreas sensitivity to rate of change of glycemia devation from equilibrium) and 
                $T_P$ [min] is time constant of derivative part.
                ''')

            options = ['normal', 'obese', 't2dm']
            pars = st.selectbox("Choose model parameters", options=options)
            
            disp_par = st.beta_container()

            basal_cols = st.beta_columns(2)
            
            Gb = basal_cols[0].number_input("Basal glucose concentration [mmol/L]", 
                                            min_value=3, 
                                            max_value=10, 
                                            value=5)
            Ib = basal_cols[1].number_input("Basal insulin concentration [mU/L]", 
                                            min_value=1, 
                                            max_value=40, 
                                            value=6)
            # glycemic response vs. glycemic index 
            # initialize model for normal subject
            model = minimal_model.iv(Gb, Ib, parameters=pars)
            
            with disp_par.beta_expander("See parameter values"):
                st.write(model.parameters)

            input_cols = st.beta_columns(2)
            
            glucose_dose = input_cols[0].slider("Glucose bolus dose [g/kg]", 
                                                min_value=0.1, 
                                                max_value=1.0, 
                                                value=0.3)
            glucose_bolus_min = input_cols[0].slider("Duration of glucose administration [min]", 
                                                     min_value=0.5, 
                                                     max_value=5.0, 
                                                     value=2.0)

            insulin_dose = input_cols[1].slider("Insulin bolus dose [mU/kg]", 
                                                min_value=1, 
                                                max_value=50, 
                                                value=20)
            insulin_bolus_min = input_cols[1].slider("Duration of insulin administration [min]", 
                                                     min_value=0.5, 
                                                     max_value=10.0, 
                                                     value=5.0)
            insulin_dosage_time = input_cols[1].slider("Time of insulin administration (from the start) [min]", 
                                                       min_value=1, 
                                                       max_value=40, 
                                                       value=20)
            
            t, G, I  = model.ivgtt(glucose_dose, 
                                   glucose_bolus_min, 
                                   insulin_dose, 
                                   insulin_bolus_min, 
                                   insulin_dosage_time, 
                                   plot=False)
            
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
            st.subheader("Oral glucose tolerance test")
            
            options = ['normal', 't2dm']
            pars = st.selectbox("Model parameters", options=options)
            
            basal_cols = st.beta_columns(2)
            
            Gb = basal_cols[0].number_input("Basal glucose concentration [mmol/L]", 
                                            min_value=3, 
                                            max_value=10, 
                                            value=5)
            Ib = basal_cols[1].number_input("Basal insulin concentration [mU/L]", 
                                            min_value=1, 
                                            max_value=40, 
                                            value=6)
            # glycemic response vs. glycemic index 
            # initialize model for normal subject
            model = minimal_model.oral(Gb, Ib, parameters = pars)
            
            glucose = st.slider("Glucose amount [g]", min_value=1, max_value=100, value=50)
            BW = st.slider("Bodyweight of a subject [kg]", min_value=50, max_value=120, value=70)
            # simulation for normal subject - high glycemic index
            
            gly_index_options = ['glucose', 'low', 'medium', 'high']
            gly_index = st.selectbox(label= "Glycemic index", options=gly_index_options)
            
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
        
        elif simulationPage == 'Glucose clamp simulation':
            st.subheader("Hyperinsulinemic euglycemic glucose clamp simulator")
            options = ['normal', 'obese', 't2dm']
            pars = st.selectbox("Model parameters", options=options)
            
            basal_cols = st.beta_columns(2)
            
            Gb = basal_cols[0].number_input("Basal glucose concentration [mmol/L]", 
                                            min_value=3, 
                                            max_value=10, 
                                            value=5)
            Ib = basal_cols[1].number_input("Basal insulin concentration [mU/L]", 
                                            min_value=1, 
                                            max_value=40, 
                                            value=6)
            # glycemic response vs. glycemic index 
            # initialize model for normal subject
            model = minimal_model.iv(Gb, Ib, parameters=pars)
            
            BW = st.slider("Bodyweight of a subject [kg]", min_value=50, max_value=120, value=70)
            
            t, G, I, RaG_iv = model.hyperinsulinemic_euglycemic_glucose_clamp(BW, plot=False)
            
            def plot_simulation_results():
                #fig = go.Figure()
                fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1)
                fig.append_trace(go.Scatter(x=t, y=G, name="Glucose conc. [mmol/L]"), row=1, col=1)
                fig.append_trace(go.Scatter(x=t, y=I, name="Insulin conc. [mU/L]"), row=2, col=1)
                fig.append_trace(go.Scatter(x=t, y=RaG_iv, name="Glucose infusion rate [mmol/min/kg]"), row=3, col=1)
                fig.layout.update(title_text='IVGTT simulation', 
                                  xaxis_rangeslider_visible=False, 
                                  showlegend=False,
                                  autosize=False,
                                  height=1000,
                                  width=800)
                fig.update_xaxes(title_text="time [min]", row=3, col=1)
                fig.update_yaxes(title_text="Glucose conc. [mmol/L]", row=1, col=1)
                fig.update_yaxes(title_text="Insulin conc. [mU/L]", row=2, col=1)
                fig.update_yaxes(title_text="Glucose infusion rate [mmol/min/kg]", row=3, col=1)
                st.plotly_chart(fig)

            plot_simulation_results()