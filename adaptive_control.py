import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from adaptive_control_fcns import sim_MRAC
import matplotlib.pyplot as plt
from scipy import signal


def adaptive_control_app():
    st.header('Demo of adaptive control in type 1 diabetes')
    st.subheader('A 4-day simulation of automated insulin administration uwing real data of type 1 diabetic patient (carbohydrate intake)')
    # import of carbohydrate intake data
    DataSetName = 'Dat_test_4days'
    Dat_carb = np.loadtxt('Data/'+DataSetName + '_carb.csv', delimiter=',', ndmin=2) # 10 g
    

    Ts = 5
    t_start = 0
    t_stop = 4*24*60
    idx_final = int((t_stop - t_start)/Ts) + 1
    
    # main time vector:
    tt = np.zeros([idx_final, 1])

    for idx in range(1,idx_final):
        tt[idx] = idx*Ts
        
    dsig = np.zeros([idx_final, 1]) # mmol/min
    for carbRow in Dat_carb:
        dsig[int(carbRow[0]/Ts), 0] = (carbRow[1] * 10)/Ts
        
    Cdata = dsig * Ts  # carbohydrates [g]    
    
    with st.beta_expander('Show data of carbohydrate intake'):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tt[Cdata>0]/60/24, y=Cdata[Cdata>0], name="Carb intake [g]"))
        fig.layout.update(title_text='Carbohydrate intake', xaxis_rangeslider_visible=False)
        fig.update_xaxes(title_text="time [days]")
        fig.update_yaxes(title_text="Carb amount [g]")
        st.plotly_chart(fig)

    # Hovorka model parameters
    Hp = np.load('paro02c.npy')

    t_I, V_I, k_I, A_G, t_G, k_12, V_G, EGP_0, F_01, k_b1, k_b2, k_b3, k_a1, k_a2, k_a3, t_cgm = Hp

    Gb=7 # basal (estimated) value of glycemia

    # reference signal
    ra = 0.5
    rper = 24*60
    r = -ra*signal.square(tt/rper*2*np.pi)

    dsigm = dsig*1000/180 # carbohydrate intake as seen by model (insulin-glucose system)
    dsigc = dsig*Ts # this signal is for disturbance rejection algorithm
    
    
    # adaptive control simulation
    data_load_state = st.text('Simulation in progress...')
    x, u, ud, vb = sim_MRAC(tt, Hp, dsigm, dsigc, r, Gb)
    data_load_state.text('Simulation in progress...done')

    # signals for plotting
    Vbasdata = (u+vb)/1e3*60 # basal insulin [U/h]
    Vboldata = -ud/1e3*Ts # bolus insulin [U]
    Vboldata = np.abs(Vboldata)
    Gcgm = x[:, 10]
    
    def plot_adaptive_control_sim():
        fig = make_subplots(rows=4, cols=1, vertical_spacing=0.1)
        fig.append_trace(go.Scatter(x=np.squeeze(tt/60/24,), y=np.squeeze(Gcgm), name="Glucose conc. [mmol/L]"), row=1, col=1)
        fig.append_trace(go.Bar(x=tt[Cdata>0]/60/24, y=Cdata[Cdata>0], name="Carb intake [g]"), row=2, col=1)
        fig.append_trace(go.Scatter(x=np.squeeze(tt/60/24), y=np.squeeze(Vbasdata), name="Basal insulin [U/h]"), row=3, col=1)
        fig.append_trace(go.Bar(x=tt[Vboldata>0]/60/24, y=Vboldata[Vboldata>0], name="Bolus insulin [U]"), row=4, col=1)
        fig.layout.update(title_text='Adaptive control simulation', 
                            xaxis_rangeslider_visible=False, 
                            showlegend=False,
                            autosize=False,
                            height=1000,
                            width=800)
        fig.update_xaxes(title_text="time [days]", row=3, col=1)
        fig.update_yaxes(title_text="Glucose conc. [mmol/L]", row=1, col=1)
        fig.update_yaxes(title_text="Carb amount [g]", row=2, col=1)
        fig.update_yaxes(title_text="Basal insulin [U/h]", row=3, col=1)
        fig.update_yaxes(title_text="Bolus insulin [U]", row=4, col=1)
        st.plotly_chart(fig)
    plot_adaptive_control_sim()
    