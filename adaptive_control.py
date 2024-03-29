import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from adaptive_control_fcns import sim_MRAC
import matplotlib.pyplot as plt
from scipy import signal
from control_systems import lin_model

# import time


def adaptive_control_app():
    st.header("Demo of adaptive control in type 1 diabetes")
    st.subheader(
        "A 4-day simulation of automated insulin administration uwing real data of type 1 diabetic patient (carbohydrate intake)"
    )
    # import of carbohydrate intake data
    DataSetName = "Dat_test_4days"
    Dat_carb = np.loadtxt(
        "Data/" + DataSetName + "_carb.csv", delimiter=",", ndmin=2
    )  # 10 g

    Ts = 5
    t_start = 0
    t_stop = 4 * 24 * 60
    idx_final = int((t_stop - t_start) / Ts) + 1

    # main time vector:
    tt = np.zeros([idx_final, 1])

    for idx in range(1, idx_final):
        tt[idx] = idx * Ts

    dsig = np.zeros([idx_final, 1])  # mmol/min
    for carbRow in Dat_carb:
        dsig[int(carbRow[0] / Ts), 0] = (carbRow[1] * 10) / Ts

    Cdata = dsig * Ts  # carbohydrates [g]

    with st.expander("Show data of carbohydrate intake"):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=tt[Cdata > 0] / 60 / 24, y=Cdata[Cdata > 0], name="Carb intake [g]"
            )
        )
        fig.layout.update(
            title_text="Carbohydrate intake", xaxis_rangeslider_visible=False
        )
        fig.update_xaxes(title_text="time [days]")
        fig.update_yaxes(title_text="Carb amount [g]")
        st.plotly_chart(fig)

    # Hovorka model parameters
    Hp = np.load("paro02c.npy")

    (
        t_I,
        V_I,
        k_I,
        A_G,
        t_G,
        k_12,
        V_G,
        EGP_0,
        F_01,
        k_b1,
        k_b2,
        k_b3,
        k_a1,
        k_a2,
        k_a3,
        t_cgm,
    ) = Hp

    Gb = 7  # basal (estimated) value of glycemia

    with st.expander("Simulation model details"):
        st.subheader("Model description and equations")
        st.markdown(
            r"""
        Model used to simulate glucose-insulin system is the nonlinear compartment Hovorka model, 
        which consists of 5 subsystems (glucose and insulin absorption, insulin action, glucose subystem and CGM measruement dynamics) 
        of total 11 differential equations and 16 parameters. The model has two inputs: glucose intake 
        $d(t)$ [mmol/min] and insulin administration $v(t)$ [mU/min], and one output: glycemia $G_{CGM}(t)$ [mmol/l].
        """
        )
        st.markdown(
            r"Glucose absorption subsystem is described by two differential equations:"
        )
        st.latex(
            r"""
            \begin{array}{ll}
            \frac{dF(t)}{dt}&=A_{G}\frac{d(t)}{t_{G}}-\frac{F(t)}{t_{G}}\\
            \frac{dRa(t)}{dt}&=\frac{F(t)}{t_{G}}-\frac{Ra(t)}{t_{G}}
            \end{array}
        """
        )
        st.markdown(
            r"""
        Where $d(t)$ [mmol/min] is rate of glucose intake, modeled as $d(t)=D\cdot\delta(t-\tau_G)$, 
        where $\delta$ is Dirac impulse function approximation corresponding to sample rate 
        and $D$ [mmol] is glucose amount (1 mg=180 mmol for glucose molecule). Further,  $F(t)$ [mmol/min] 
        is rate of glucose absorption in first compartment, $Ra(t)$ is rate of appearance of 
        glucose in plasma, $A_G$ [-] is carbohydrate bioavailability and $t_G$ [min] is the time constant of this subsystem. 
        """
        )
        st.markdown(r"Insulin absorption dynamics is given by:")
        st.latex(
            r"""
            \begin{array}{ll}
            \frac{dS_{1}(t)}{dt}&=v(t)-\frac{S_{1}(t)}{t_{I}} \\
            \frac{dS_{2}(t)}{dt}&=\frac{S_{1}(t)}{t_{I}}-\frac{S_{2}(t)}{t_{I}} \\
            \frac{dI(t)}{dt}&=\frac{S_{2}(t)}{t_I V_I}-k_{I}I(t)
            \end{array}
        """
        )
        st.markdown(
            r"""
        Where $v(t)$ [mU/min] is rate of insulin intake and is sum of bolus and basal, 
        $v(t)=v_{bas}(t)+v_{bol}(t)$. Bolus insulin administration is modeled same way as 
        glucose intake (Dirac impulses). Basal is modeled as constant signal. 
        ignals $S_1$ and $S_2$ are state variables describing absorption of subcutaneously 
        administered insulin, $t_I$ [min] is time constant, $I(t)$ [mU/l] is the plasma insulin 
        concentration, $V_I$ [l] is the distribution volume and $k_I$ [min$^{-1}$] is the fractional elimination rate.
        """
        )
        st.markdown(
            r"Insulin action subsystem describes three actions of insulin on glucose kinetics:"
        )
        st.latex(
            r"""
            \begin{array}{ll}
            \frac{dx_{1}(t)}{dt}&=k_{b1}I(t)-k_{a1}x_{1}(t)\\
            \frac{dx_{2}(t)}{dt}&=k_{b2}I(t)-k_{a2}x_{2}(t)\\
            \frac{dx_{3}(t)}{dt}&=k_{b3}I(t)-k_{a3}x_{3}(t)
            \end{array}
        """
        )
        st.markdown(
            r"""
        Where $x_{1}(t)$ [min$^{-1}$] is rate of remote effect of insulin on glucose transport, 
        $x_{2}(t)$ [min$^{-1}$] elimination and $x_{3}(t)$ [-] endogenous glucose production. 
        Dynamics of these effects is given by constants: $k_{a1}$ [min$^{-1}$], $k_{a2}$ [min$^{-1}$], $k_{a3}$ [min$^{-1}$] 
        (deactivation rate constants) a $k_{b1}$ [min$^{-2}$mU$^{-1}$l], $k_{b2}$ [min$^{-2}$mU$^{-1}$l], 
        $k_{b3}$ [min$^{-1}$mU$^{-1}$l] (activation rate constants).
        """
        )
        st.markdown(
            r"Glucose subsystem describes insulin-glucose interaction with two nonlinear differential equations:"
        )
        st.latex(
            r"""
        \begin{array}{ll}
        \frac{dQ_{1}(t)}{dt}=&-(F^C_{01}+F_{R})-x_{1}(t)Q_{1}(t)+k_{12}Q_{2}(t)+Ra(t)\\
                            &+EGP_{0}[1-x_{3}(t)] \\
        \frac{dQ_{2}(t)}{dt}=&x_{1}(t)Q_{1}(t)-[k_{12}+x_{2}(t)]Q_{2}(t)										
        \end{array}
        """
        )
        st.markdown(
            r"""
        Where $Q_1$, $Q_2$ represent the masses of glucose in the accessible (where glycemia measurements are made) and 
        non-accessible compartments (for example muscle tissues), $k_{12}$ [min$^{-1}$] is the transfer rate constant from $Q_2$ to $Q_1$. 
        Glycemia is given by:
        """
        )
        st.latex(r"G(t)=\frac{Q_{1}(t)}{V_{G}} ")
        st.markdown(
            r"Where $V_G$ [l] is glucose distribution volume. $F^C_{01}$ [mmol/min] represents total non-insulin dependent glucose flux."
        )
        st.latex(
            r"""
        F^C_{01}=\left\{\begin{array}{ll}
        F_{01} & G(t)\geq4.5\textrm{ mmol/l} \\
        F_{01}G(t)/4.5 & \textrm{otherwise} 
        \end{array} \right.
        """
        )
        st.markdown(
            r"$F_{R}$ [mmol/min] represents renal glucose clearance above the glucose concentration threshold of 9~mmol/l:"
        )
        st.latex(
            r"""
        F_{R}=\left\{\begin{array}{ll}
        0.003(G(t)-9)V_{G} & G(t)\geq9\textrm{ mmol/l} \\
        0 & \textrm{otherwise} 
        \end{array} \right.
        """
        )
        st.markdown(
            r"The last equation represents dynamics or delay in glycemia measurement which is modeled by first order dynamics: "
        )
        st.latex(
            r"\frac{dG_{CGM}(t)}{dt}=\frac{G(t)}{t_{CGM}}-\frac{G_{CGM}(t)}{t_{CGM}}"
        )
        st.markdown(
            r"""
        Where $G_{CGM}(t)$ [mmol/l] is the final output of the system = measured glycemia and $t_{CGM}$ [min] 
        is time constant which governs the delay between actual and measured glycemia
        """
        )

    with st.expander("Display parameter values"):
        # st.write(model.parameters)
        """
        t_I,
        V_I,
        k_I,
        A_G,
        t_G,
        k_12,
        V_G,
        EGP_0,
        F_01,
        k_b1,
        k_b2,
        k_b3,
        k_a1,
        k_a2,
        k_a3,
        t_cgm,
        """
        st.latex(
            r"""
                    \begin{array}{lll}
                    t_I &= """
            + str(round(Hp[0], 2))
            + r"""& \textrm{[min]} \\
                    V_I &= """
            + format(Hp[1], ".2f")
            + r"""& \textrm{[l]} \\
                    k_I &= """
            + str(round(Hp[2], 2))
            + r"""& \textrm{[l/min]} \\
                    A_{G} &= """
            + str(round(Hp[3], 2))
            + r"""& \textrm{[-]} \\
                    t_{G} &= """
            + str(round(Hp[4], 2))
            + r"""& \textrm{[min]} \\
                    k_{12} &= """
            + format(Hp[5], ".2e")
            + r"""& \textrm{[min]} \\
                    V_{G} &= """
            + str(round(Hp[6], 2))
            + r"""& \textrm{[l]} \\
                    EGP_{0} &= """
            + str(round(Hp[7], 2))
            + r"""& \textrm{[mmol/min]} \\
                    F_{01} &= """
            + str(round(Hp[8], 2))
            + r"""& \textrm{[mmol/min]} \\
                    k_{b1} &= """
            + format(Hp[9], ".2e")
            + r"""& \textrm{[min$^{-2}$mU$^{-1}$l]} \\
                    k_{b2} &= """
            + format(Hp[10], ".2e")
            + r"""& \textrm{[min$^{-2}$mU$^{-1}$l]} \\
                    k_{b3} &= """
            + format(Hp[11], ".2e")
            + r"""& \textrm{[min$^{-1}$mU$^{-1}$l]} \\
                    k_{a1} &= """
            + format(Hp[12], ".2e")
            + r"""& \textrm{[1/min]} \\
                    k_{a2} &= """
            + format(Hp[13], ".2e")
            + r"""& \textrm{[1/min]} \\
                    k_{a3} &= """
            + format(Hp[14], ".2e")
            + r"""& \textrm{[1/min]} \\
                    t_{CGM} &= """
            + str(round(Hp[15], 2))
            + r"""& \textrm{[min]} \\
                    \end{array}
                    """
        )

    st.subheader("Reference model:")
    st.latex(r"W_m(s)=\frac{a_{0m}}{s^2 + a_{1m} s + a_{0m}}")
    col_ref_model = st.columns(2)
    a1m = col_ref_model[0].number_input(r"a1m", 0.01, 0.1, 0.05)
    a0m = col_ref_model[1].number_input(
        r"a0m", 0.0001, 0.001, 0.00035, step=0.00005, format="%.6f"
    )

    num = [a0m]
    den = [1, a1m, a0m]
    ref_model = lin_model([num, den])

    t = np.arange(0, 12 * 60, 1)
    u = np.ones_like(t)
    x0 = [0, 0]
    _, y = ref_model.simulation(x0, t, u)

    def plot_ref_model_step_response():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t / 60, y=np.squeeze(y), name="step response"))
        fig.update_xaxes(title_text="time [hours]")
        fig.update_yaxes(title_text="Reference model output [mmol/L]")
        st.plotly_chart(fig)

    with st.expander("Reference model step response"):
        plot_ref_model_step_response()
    st.subheader("Change shape, amplitude and period of reference signal")
    col_ref_signal_choice = st.columns(3)
    ref_signal_options = ["sinewave", "square", "sawtooth"]

    ra = col_ref_signal_choice[0].slider(
        "Amplitude [mmol/L]", min_value=0.1, max_value=1.0, value=0.5
    )
    rper = (
        col_ref_signal_choice[1].slider(
            "Period [hours]", min_value=12, max_value=36, value=24
        )
        * 60
    )
    ref_shape = col_ref_signal_choice[2].selectbox(
        "Reference signal shape", options=ref_signal_options, index=1
    )
    # reference signal
    # ra = 0.5
    # rper = 24 * 60
    if ref_shape == "square":
        r = -ra * signal.square(tt / rper * 2 * np.pi)
    elif ref_shape == "sawtooth":
        r = -ra * signal.sawtooth(tt / rper * 2 * np.pi)
    elif ref_shape == "sinewave":
        r = -ra * np.sin(tt / rper * 2 * np.pi)
    _, y = ref_model.simulation(x0, np.squeeze(tt), r)

    def plot_ref_model_response():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=np.squeeze(tt / 60), y=np.squeeze(r), name="reference signal")
        )
        fig.add_trace(
            go.Scatter(
                x=np.squeeze(tt / 60), y=np.squeeze(y), name="reference model response"
            )
        )
        fig.layout.update(
            title_text="Reference signal response", xaxis_rangeslider_visible=False
        )
        fig.update_xaxes(title_text="time [hours]")
        fig.update_yaxes(title_text="[mmol/L]")
        st.plotly_chart(fig)

    with st.expander("Reference model response to reference signal"):
        plot_ref_model_response()

    with st.expander("Control algorithm details"):
        st.text("Control algorithm:")
        st.latex(
            r"""
                \begin{array}{ll}
                \omega(t)&=\left[\frac{[s\quad 1]^T}{\Lambda(s)}u(t) \quad \frac{[s\quad 1]^T}{\Lambda(s)}y(t) \quad y(t) \quad r(t)\right]^T \\
                \omega_f(t)&=\frac{1}{s+\rho}\omega(t) \\
                e_1(t)&=y(t)-y_m(t)+W_m(s)(s+\rho)\left[\frac{1}{s+\rho}u(t)-\Theta(t)^T\omega_f(t)\right] \\
                u(t)&=\Theta(t)^T\omega(t)
                \end{array}
                """
        )
        st.markdown(
            r"here $y(t)$ represents deviation from equilibrium (basal glycemia) of the controlled system (glucose system) and thus:"
        )
        st.latex(r"y(t)=G(t)-G_b")
        st.text("Adaptation law:")
        st.latex(
            r"""
                \begin{array}{ll}
                \dot{\Theta}(t)&=\frac{\Gamma \omega_f(t) e_1(t)}{1+\omega^T_f(t)\omega_f(t)} - \sigma_s \Gamma \Theta(t)\\
                \sigma_s(t) &= \left\{ 
                \begin{array}{l l}
                0 & \quad \text{if } \left|\Theta(t)\right|\leq M_0  \\
                \left(\frac{\left|\Theta(t)\right|}{M_0}-1\right)^{q_0} \sigma_0 & \quad \text{if } M_0 < \left|\Theta(t)\right|\leq 2M_0  \\
                \sigma_0 & \quad \text{if } \left|\Theta(t)\right|> 2M_0
                \end{array} \right.
                \end{array}
                """
        )
        st.text("estimate of disturbance (carbohydrate intake) effect on system input:")
        st.latex(r"u_d(t)=\Theta_d(t)d(t)")

        st.markdown(
            r"""
                    Where $\Theta_d(t)$ is gain, which will be adapted so that the disturbance is rejected. 
                    Heuristic gradient-based adaptive law with switching $\sigma$-modification and normalization is used to estimate this gain:
                    """
        )
        st.latex(
            r"""
                \begin{array}{ll}
                \dot{\Theta}_d(t)&=-\frac{\gamma d(t)[y(t)-y_m(t)]}{1+d^2(t)}  - \sigma_d(t) \gamma \Theta_d(t)\\
                \sigma_d(t) &= \left\{ 
                \begin{array}{l l}
                0 & \quad \text{if } \left|\Theta_d(t)\right|\leq M_{0d}  \\
                \left(\frac{\left|\Theta_d(t)\right|}{M_{0d}}-1\right)^{q_0} \sigma_{0d} & \quad \text{if } M_{0d} < \left|\Theta_d(t)\right|\leq 2M_{0d}  \\
                \sigma_{0d} & \quad \text{if } \left|\Theta_d(t)\right|> 2M_{0d}
                \end{array} \right.
                \end{array}
                """
        )
        st.markdown(
            r"where $\gamma$, $M_{0d}$, $\sigma_{0d}$ are adaptive law parameters. Controller output together with basal administration will be:"
        )
        st.latex(r"v(t)=v_{bas}+u(t)-u_d(t)")
        st.markdown(
            r"Signal $u(t)$ represents basal and $-u_d(t)$ bolus administration, $v_{bas}$ is operating point (basal state of simulator)."
        )

        st.markdown(
            r"""
                    Notice that this method of disturbance rejection requires disturbance to be known, 
                    but precise carbohydrate content in meal does not have to be known. Knowledge of past carbohydrate content of specific person 
                    can be used to create three groups, each representing range of carbohydrate content. Diabetic person would then only had to 
                    choose between three options: low, medium or high carbohydrate content. 
                    """
        )
        st.markdown(
            r""" Let the maximum carbohydrate content be $CHO_{max}$, 
                    then the disturbance $d(t)$, which will be used in control algorithm can be switched as:"""
        )
        st.latex(
            r"""
                \begin{array}{ll}
                d(t) &= \left\{ 
                \begin{array}{l l}
                1 & \text{if } CHO>0 \textrm{ and } CHO\leq \frac{1}{3}CHO_{max}  \\
                2 &  \text{if } CHO>\frac{1}{3}CHO_{max} \textrm{ and } CHO\leq \frac{2}{3}CHO_{max}  \\
                3 &  \text{if } CHO>\frac{2}{3}CHO_{max}
                \end{array} \right.
                \end{array}
                """
        )
    dsigm = (
        dsig * 1000 / 180
    )  # carbohydrate intake as seen by model (insulin-glucose system)
    dsigc = dsig * Ts  # this signal is for disturbance rejection algorithm

    rm_list = [a1m, a0m]

    with st.expander("Control algorithm parameters"):
        st.subheader("Auxiliary filter denominator")
        st.latex(r"\Lambda(s)=s^2 + \lambda_{1} s + \lambda_{0}")
        col_aux_filter = st.columns(2)
        lambda1 = col_aux_filter[0].number_input(r"lambda1", 0.01, 0.5, 0.2)
        lambda0 = col_aux_filter[1].number_input(
            r"lambda0", 0.001, 0.1, 0.01, step=0.01  # , format="%.6f"
        )
        st.subheader(
            "Parameter for SPR (Strictly Positive Real) reference model in adaptation law:"
        )
        st.markdown(
            r"""
                 $\rho$ must be such that $W_m(s)(s+\rho)$
                 is strictly positive real function which is satisfied when 
                 $\rho<a_{1m}$ and so $\rho<$ """
            + str(a1m)
            + """
                 """
        )
        ro = st.number_input(r"rho", 0.001, a1m, 0.01)
        st.subheader("Sigma-modified Lyapunov adaptive law parameters:")
        st.text("Adaptation law")
        st.latex(
            r"""
        \begin{array}{ll}
        \dot{\Theta}(t)&=\frac{\Gamma \omega_f(t) e_1(t)}{1+\omega^T_f(t)\omega_f(t)} - \sigma_s \Gamma \Theta(t)\\
        \sigma_s(t) &= \left\{ 
        \begin{array}{l l}
        0 & \quad \text{if } \left|\Theta(t)\right|\leq M_0  \\
        \left(\frac{\left|\Theta(t)\right|}{M_0}-1\right)^{q_0} \sigma_0 & \quad \text{if } M_0 < \left|\Theta(t)\right|\leq 2M_0  \\
        \sigma_0 & \quad \text{if } \left|\Theta(t)\right|> 2M_0
        \end{array} \right.
        \end{array}
        """
        )
        st.latex(
            r"\text{Adaptation gain: }\Gamma = diag(\gamma_1, \gamma_2, \gamma_3, \gamma_4 ,\gamma_5, \gamma_6)"
        )
        col_adapt_gain1 = st.columns(3)
        col_adapt_gain1[0].latex(r"\gamma_1")
        gamma1 = col_adapt_gain1[0].number_input(r"", 0.01, 1.0, 0.1)
        col_adapt_gain1[1].latex(r"\gamma_2")
        gamma2 = col_adapt_gain1[1].number_input(r"  ", 0.001, 0.1, 0.01)
        col_adapt_gain1[2].latex(r"\gamma_3")
        gamma3 = col_adapt_gain1[2].number_input(r"   ", 0.01, 1.0, 0.1)
        col_adapt_gain2 = st.columns(3)
        col_adapt_gain2[0].latex(r"\gamma_4")
        gamma4 = col_adapt_gain2[0].number_input(r"", 0.001, 0.1, 0.01)
        col_adapt_gain2[1].latex(r"\gamma_5")
        gamma5 = col_adapt_gain2[1].number_input(r" ", 1, 100, 50)
        col_adapt_gain2[2].latex(r"\gamma_6")
        gamma6 = col_adapt_gain2[2].number_input(r"  ", 1e3, 10e3, 5e3)
        st.text("Sigma-modification parameters:")
        col_sigma_mod = st.columns(3)
        col_sigma_mod[0].latex(r"M_0")
        M0 = col_sigma_mod[0].number_input(" ", 50, 150, 100)
        col_sigma_mod[1].latex(r"q_0")
        q0 = col_sigma_mod[1].number_input("  ", 0.5, 1.5, 1.0)
        col_sigma_mod[2].latex(r"\sigma_0")
        sigma0 = col_sigma_mod[2].number_input("   ", 0.5, 1.5, 1.0)

        st.subheader(
            "Sigma-modified heuristic gradient-based adaptation law (disturbance rejection):"
        )
        st.text("Adaptation law")
        st.latex(
            r"""
            \begin{array}{ll}
            \dot{\Theta}_d(t)&=-\frac{\gamma d(t)[y(t)-y_m(t)]}{1+d^2(t)}  - \sigma_d(t) \gamma \Theta_d(t)\\
            \sigma_d(t) &= \left\{ 
            \begin{array}{l l}
            0 & \quad \text{if } \left|\Theta_d(t)\right|\leq M_{0d}  \\
            \left(\frac{\left|\Theta_d(t)\right|}{M_{0d}}-1\right)^{q_0} \sigma_{0d} & \quad \text{if } M_{0d} < \left|\Theta_d(t)\right|\leq 2M_{0d}  \\
            \sigma_{0d} & \quad \text{if } \left|\Theta_d(t)\right|> 2M_{0d}
            \end{array} \right.
            \end{array}
            """
        )
        st.latex(r"\text{Adaptation gain }\gamma")
        gamma = st.number_input(" ", 1, 10, 5)
        st.text("Sigma-modification parameters:")
        col_sigma_mod_dist = st.columns(2)
        col_sigma_mod_dist[0].latex(r"M_{0d}")
        M0d = col_sigma_mod_dist[0].number_input(" ", 500, 1500, 1000)
        col_sigma_mod_dist[1].latex(r"\sigma_d")
        sigma0d = col_sigma_mod_dist[1].number_input("  ", 500, 1500, 1000)
    Gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
    lambdapar = [lambda1, lambda0]

    @st.cache(suppress_st_warning=True)
    def sim_adaptive_control(
        tt,
        Hp,
        dsigm,
        dsigc,
        r,
        Gb,
        rm_list,
        lambdapar,
        ro,
        Gamma,
        M0,
        q0,
        sigma0,
        gamma,
        M0d,
        sigma0d,
    ):
        x, u, ud, vb = sim_MRAC(
            tt,
            Hp,
            dsigm,
            dsigc,
            r,
            Gb,
            rm_list,
            lambdapar,
            ro,
            Gamma,
            M0,
            q0,
            sigma0,
            gamma,
            M0d,
            sigma0d,
        )
        return x, u, ud, vb

    # data_load_state = st.text("Simulation in progress...")
    # start_time = time.time()
    with np.load("default_mrac_sim.npz") as data:
        x = data["x"]
        u = data["u"]
        ud = data["ud"]
        vb = data["vb"]
    """
    x, u, ud, vb = sim_adaptive_control(
        tt,
        Hp,
        dsigm,
        dsigc,
        -0.5 * signal.square(tt / 24 / 60 * 2 * np.pi),
        Gb,
        [0.05, 0.00035],
        [0.2, 0.01],
        0.01,
        np.diag([0.1, 0.01, 0.1, 0.01, 50, 5e3]),
        100,
        1,
        1,
        5,
        1e3,
        1e3,
    )
    """
    # data_load_state.text("Simulation in progress...done")
    # st.text("Simulation took %s seconds " % (time.time() - start_time))
    # adaptive control simulation
    col_rerun_sim = st.columns(2)
    if col_rerun_sim[0].button("Re-run simulation"):

        # start_time = time.time()
        data_load_state = st.text("Simulation in progress...")
        x, u, ud, vb = sim_adaptive_control(
            tt,
            Hp,
            dsigm,
            dsigc,
            r,
            Gb,
            rm_list,
            lambdapar,
            ro,
            Gamma,
            M0,
            q0,
            sigma0,
            gamma,
            M0d,
            sigma0d,
        )
        data_load_state.text("Simulation in progress...done")
        # st.text("Simulation took %s seconds " % (time.time() - start_time))

    col_rerun_sim[1].text(
        "Note: This may take more than 5 minutes \nwhen run from Heroku"
    )
    # signals for plotting
    Vbasdata = (u + vb) / 1e3 * 60  # basal insulin [U/h]
    Vboldata = -ud / 1e3 * Ts  # bolus insulin [U]
    Vboldata = np.abs(Vboldata)
    Gcgm = x[:, 10]

    def plot_adaptive_control_sim():
        fig = make_subplots(rows=4, cols=1, vertical_spacing=0.1)
        fig.append_trace(
            go.Scatter(
                x=np.squeeze(
                    tt / 60 / 24,
                ),
                y=np.squeeze(Gcgm),
                name="Glucose conc. [mmol/L]",
            ),
            row=1,
            col=1,
        )
        fig.append_trace(
            go.Scatter(
                x=np.squeeze(
                    tt / 60 / 24,
                ),
                y=np.ones_like(Gcgm) * 4,
                name="hypoglycemia",
                line_color="#FF0000",
            ),
            row=1,
            col=1,
        )
        fig.append_trace(
            go.Scatter(
                x=np.squeeze(
                    tt / 60 / 24,
                ),
                y=np.ones_like(Gcgm) * 10,
                name="hyperglycemia",
                line_color="#00FF00",
            ),
            row=1,
            col=1,
        )
        fig.append_trace(
            go.Bar(
                x=tt[Cdata > 0] / 60 / 24, y=Cdata[Cdata > 0], name="Carb intake [g]"
            ),
            row=2,
            col=1,
        )
        fig.append_trace(
            go.Scatter(
                x=np.squeeze(tt / 60 / 24),
                y=np.squeeze(Vbasdata),
                name="Basal insulin [U/h]",
            ),
            row=3,
            col=1,
        )
        fig.append_trace(
            go.Bar(
                x=tt[Vboldata > 0] / 60 / 24,
                y=Vboldata[Vboldata > 0],
                name="Bolus insulin [U]",
            ),
            row=4,
            col=1,
        )
        fig.layout.update(
            title_text="Adaptive control simulation",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            autosize=False,
            height=1000,
            width=800,
        )
        fig.update_xaxes(title_text="time [days]", row=4, col=1)
        fig.update_yaxes(title_text="Glucose conc. [mmol/L]", row=1, col=1)
        fig.update_yaxes(title_text="Carb amount [g]", row=2, col=1)
        fig.update_yaxes(title_text="Basal insulin [U/h]", row=3, col=1)
        fig.update_yaxes(title_text="Bolus insulin [U]", row=4, col=1)
        st.plotly_chart(fig)

    plot_adaptive_control_sim()

    def plot_controller_parameters():
        st.markdown(r"Evolution of $\Theta_d(t)$")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.squeeze(tt / 60 / 24),
                y=x[:, 26],
                name=r"$\Theta_d$",
            ),
        )
        fig.layout.update(
            # title_text="Time evolution of controller parameters",
            xaxis_rangeslider_visible=False,
            showlegend=False,
            autosize=False,
            height=400,
            width=800,
        )
        fig.update_xaxes(title_text="time [days]")
        fig.update_yaxes(title_text="parameter value")
        st.plotly_chart(fig)
        st.markdown(r"Evolution of $\Theta_{1-4}(t)$")
        fig = go.Figure()
        for k in range(4):
            fig.add_trace(
                go.Scatter(
                    x=np.squeeze(tt / 60 / 24),
                    y=x[:, 27 + k],
                    name="theta" + str(k + 1),
                ),
            )
        fig.layout.update(
            # title_text="Time evolution of controller parameters",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            autosize=False,
            height=400,
            width=800,
        )
        fig.update_xaxes(title_text="time [days]")
        fig.update_yaxes(title_text="parameter value")
        st.plotly_chart(fig)
        st.markdown(r"Evolution of $\Theta_{5-6}(t)$")
        fig = go.Figure()
        for k in range(2):
            fig.add_trace(
                go.Scatter(
                    x=np.squeeze(tt / 60 / 24),
                    y=x[:, 27 + k + 4],
                    name="theta" + str(k + 1 + 4),
                ),
            )
        fig.layout.update(
            # title_text="Time evolution of controller parameters",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            autosize=False,
            height=400,
            width=800,
        )
        fig.update_xaxes(title_text="time [days]")
        fig.update_yaxes(title_text="parameter value")
        st.plotly_chart(fig)

    with st.expander("Time evolution of adapting parameters"):
        plot_controller_parameters()
