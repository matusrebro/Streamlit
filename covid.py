from re import L
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.metrics import accuracy_score
from datetime import datetime

from plotly.subplots import make_subplots

def app():

    st.header("COVID-19")

    daily_stats_slovakia_url = "https://raw.githubusercontent.com/Institut-Zdravotnych-Analyz/covid19-data/main/DailyStats/OpenData_Slovakia_Covid_DailyStats.csv"

    covid_deaths_agegroup_district = "https://raw.githubusercontent.com/Institut-Zdravotnych-Analyz/covid19-data/main/Deaths/OpenData_Slovakia_Covid_Deaths_AgeGroup_District.csv"

    @st.cache
    def get_covid_data():
        daily_stats = pd.read_csv(daily_stats_slovakia_url, sep=';')
        daily_stats['Datum'] = pd.to_datetime(daily_stats['Datum'], format='%Y-%m-%d')
        covid_deaths = pd.read_csv(covid_deaths_agegroup_district, sep=';', encoding='cp1250')
        covid_deaths['Date'] = pd.to_datetime(covid_deaths['Date'], format='%d.%m.%Y')
        return daily_stats, covid_deaths

    covid_data, covid_deaths = get_covid_data()
    #t = np.arange(0, len(covid_data['Datum'].values)) # days from 6.3..2020

    # st.write(covid_deaths)
    
    regions = [val for val in covid_deaths['Region'].unique() if isinstance(val, str)]
    
    deaths_per_region = covid_deaths[covid_deaths['Region']=='Trnavský'].groupby(['Date'])['Region'].count().rename('Deaths')
    deaths_per_age = covid_deaths.groupby(['AgeGroup'])['AgeGroup'].count().rename('Deaths').to_frame()
    st.text('Number of deaths per region')
    st.write(covid_deaths['Region'].value_counts())
    
    # st.write(covid_deaths[covid_deaths['Region']=='Trnavský'])
    
    # st.write(deaths_per_region)
    
    # st.write(deaths_per_age)
    
    def plot_deaths_per_age_hist():
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=covid_deaths['AgeGroup'])
            )
        fig.layout.update(
            title_text="Deaths per age group", xaxis_rangeslider_visible=False, autosize=False, height=600, width=850
        )
        fig.update_xaxes(title_text="Age Group")
        fig.update_yaxes(title_text="Deaths")
        st.plotly_chart(fig)
    
    plot_deaths_per_age_hist()
    
    def plot_daily_stats():
        # fig = go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=covid_data['Datum'], y=covid_data['Dennych.PCR.prirastkov'], name="positive PCR count"), secondary_y=True
        )
        fig.add_trace(
            go.Bar(x=covid_data['Datum'], y=covid_data['AgPosit'], name="positive Ag count"), secondary_y=True
        )

        fig.add_trace(
            go.Scatter(x=covid_data['Datum'], y=covid_data['Pocet.umrti'], name="Total deaths", line_color="#000000", opacity=1), secondary_y=False
        )

        fig.layout.update(
            title_text="Deaths and positive cases", xaxis_rangeslider_visible=False, autosize=False, height=600, width=850
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Deaths", secondary_y=False)
        fig.update_yaxes(title_text="Positive tests", secondary_y=True)
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig)

    def plot_death_stats():
        # fig = go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=covid_data['Datum'], y=covid_data['Pocet.umrti'], name="Total deaths"), secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=covid_data['Datum'], y=np.gradient(covid_data['Pocet.umrti']), name="Deaths per day"), secondary_y=True
        )

        fig.layout.update(
            title_text="Deaths", xaxis_rangeslider_visible=False, height=600, width=850
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Deaths", secondary_y=False)
        fig.update_yaxes(title_text="Deaths per day", secondary_y=True)
        st.plotly_chart(fig)

    def plot_death_stats_per_year():
        # fig = go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        data_2020 = covid_data[covid_data['Datum'].dt.year==2020]
        data_2021 = covid_data[covid_data['Datum'].dt.year==2021]
        data_2022 = covid_data[covid_data['Datum'].dt.year==2022]
        data_2021['Datum'] = data_2021['Datum'].mask(data_2021['Datum'].dt.year == 2021, 
                             data_2021['Datum'] + pd.offsets.DateOffset(year=2020))
        data_2022['Datum'] = data_2022['Datum'].mask(data_2022['Datum'].dt.year == 2022, 
                             data_2022['Datum'] + pd.offsets.DateOffset(year=2020))
        fig.add_trace(
            go.Scatter(x=data_2020['Datum'], y=np.gradient(data_2020['Pocet.umrti']), name="2020"), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=data_2021['Datum'], y=np.gradient(data_2021['Pocet.umrti']), name="2021"), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=data_2022['Datum'], y=np.gradient(data_2022['Pocet.umrti']), name="2022"), secondary_y=False
        )
        # fig.add_trace(
        #     go.Scatter(x=covid_data['Datum'], y=np.gradient(covid_data['Pocet.umrti']), name="Deaths per day"), secondary_y=True
        # )

        fig.layout.update(
            title_text="Death rate by year", xaxis_rangeslider_visible=False, xaxis=dict(tickformat="%b"), height=600, width=850
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Deaths per day", secondary_y=False)
        # fig.update_yaxes(title_text="Deaths per day", secondary_y=True)
        st.plotly_chart(fig)


    plot_daily_stats()

    plot_death_stats()
    
    plot_death_stats_per_year()
