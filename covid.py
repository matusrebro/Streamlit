import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.metrics import accuracy_score

from plotly.subplots import make_subplots

def app():

    st.header("COVID-19")

    daily_stats_slovakia_url = "https://raw.githubusercontent.com/Institut-Zdravotnych-Analyz/covid19-data/main/DailyStats/OpenData_Slovakia_Covid_DailyStats.csv"

    daily_stats_slovakia_data = pd.read_csv(daily_stats_slovakia_url, sep=';')

    # st.write(daily_stats_slovakia_data['Datum'].values)
    tdata = np.arange(0, len(daily_stats_slovakia_data['Datum'].values)) # days from 6.3..2020
    fig = plt.figure()
    plt.plot(tdata, daily_stats_slovakia_data['Pocet.umrti'])
    st.pyplot(fig)
    # confirmed_data = df_confirmed[df_confirmed.columns[4:]]
    # confirmed_data_global = confirmed_data.sum(0)

    # deaths_data = df_deaths[df_deaths.columns[4:]]
    # deaths_data_global = deaths_data.sum(0)

    # recovered_data = df_recovered[df_recovered.columns[4:]]
    # recovered_data_global = recovered_data.sum(0)


    # def plot_daily_stats():
    #     # fig = go.Figure()
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
    #     fig.add_trace(
    #         go.Scatter(x=t, y=G, name="Glucose conc. [mmol/L]"), secondary_y=False
    #     )
    #     fig.add_trace(
    #         go.Scatter(x=t, y=I, name="Insulin conc. [mU/L]"), secondary_y=True
    #     )
    #     fig.layout.update(
    #         title_text="OGTT simulation", xaxis_rangeslider_visible=False
    #     )
    #     fig.update_xaxes(title_text="time [min]")
    #     fig.update_yaxes(title_text="Glucose conc. [mmol/L]", secondary_y=False)
    #     fig.update_yaxes(title_text="Insulin conc. [mU/L]", secondary_y=True)
    #     st.plotly_chart(fig)

    #plot_daily_stats()