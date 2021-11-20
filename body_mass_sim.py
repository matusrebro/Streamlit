import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from scipy.integrate import odeint
from copy import deepcopy

rho_cal = 9e3  # fat energy density [9000 kcal/kg]


def basal_metabolic_rate(par, body_mass, height, age):
	a, b, c, d = par
	return a*body_mass + b*height + c*age + d


def fcn_m_dot(x, t, p, height, age, eee, dci):
	return 1/rho_cal * (-basal_metabolic_rate(p, x, height, age) - eee + dci)

def eq_body_mass(p, height, age, eee, dci):
	a, b, c, d = p
	return 1/a*(-b*height - c*age - d - eee + dci)

def sim_m(bw_0, days, par, height, age, eee, dci):
	x = np.zeros(days)
	x[0] = bw_0
	for i in range(1, days):
		y = odeint(
			fcn_m_dot,
			x[i - 1],
			np.linspace((i - 1), i),
			args=(
				par,
				height,
				age,
				eee,
				dci,
			),
		)
		x[i] = y[-1]
	return x

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
	
	sex_idx = 0
	if sex == "Male":
		sex_idx = 0
	else:
		sex_idx = 1
	
	P = basal_metabolic_rate(selected_par[sex_idx], body_mass, height, age)
	
	
	st.columns(3)[1].markdown(f'#### P = **{str(round(P, 2))}** kcal/day')
	
	
	def plot_bmr():
		bw_arr = np.arange(body_mass - 10, body_mass + 10, 0.1)
		bmr_arr = np.zeros_like(bw_arr)
		fig = go.Figure()
		for par in parameter_sets:
			for idx, bw in enumerate(bw_arr):
				bmr_arr[idx] = basal_metabolic_rate(parameters[par][sex_idx], bw, height, age)
			fig.add_trace(
				go.Scatter(x=bw_arr, y=bmr_arr, name=f"{par} - {sex}")
				)
			fig.layout.update(
				title_text="Basal metabolic rate", 
				xaxis_rangeslider_visible=True
				)
   
		fig.update_xaxes(title_text="body weight [kg]")
		fig.update_yaxes(title_text="P [kcal/day]")
		st.plotly_chart(fig)
		
		
		
	plot_bmr()
 

 
	st.subheader("Simulation of body weight change")
	st.markdown(r"Simple model of body mass change can be designed as differential equation where positive flow of mass is daily caloric intake and negative are basal metabolic rate and daily energy expenditure:")
	st.latex(r"\frac{dm}{dt} = \frac{1}{\rho_{cal}} \left( -P(m) -DEE +DCI \right)")
	st.markdown(r"""
		where $M$ [kg] is body mass, $\rho_{cal} = 9000$ [kcal/kg] is fat energy density and $P$, $DEE$, $DCI$ [kcal/day] are basal metabolic rate, 
		daily energy expenditure and daily energy caloric intake respectively.
	""")
	st.markdown(r"equilibrium - steady state:")
	st.latex(r"0= \frac{1}{\rho_{cal}} \left( -P(m_{eq}) -DEE +DCI \right)")
	st.markdown(r"gives body mass in equilibrium:")
	st.latex(r"m_{eq}= \frac{-bh -ca -d -DEE +DCI}{a}")
	days = st.number_input("Simulation time range [days]", 1, 1000, 10, 1)
	eee = st.number_input("Daily energy expenditure [kcal/day]", 0, 1000, 0, 1)
	dci = st.number_input("Daily calorie intake [kcal/day]", 0, 10000, 2000, 1)



	m_eq = eq_body_mass(selected_par[sex_idx], height, age, eee, dci)
	
	
	st.columns(3)[1].markdown(r'#### $m_{eq}$ = **'+ str(round(m_eq, 2))  +'** kg')

	sim_output = sim_m(body_mass, days, selected_par[sex_idx], height, age, eee, dci)

	def plot_b_sim():
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(x=np.arange(0, days), y=sim_output, name="simulation result")
		)
		fig.layout.update(
			title_text="Simulation of body weight change", 
			xaxis_rangeslider_visible=True
		)
		fig.update_xaxes(title_text="tine [day]")
		fig.update_yaxes(title_text="body weight [kg]")
		st.plotly_chart(fig)

	plot_b_sim()
	# st.write(sim_output)