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

    with st.beta_expander("Show data of carbohydrate intake"):
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

    st.subheader("Modify reference model:")
    st.latex(r"W_m(s)=\frac{a_{0m}}{s^2 + a_{1m} s + a_{0m}}")
    col_ref_model = st.beta_columns(2)
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
        fig.add_trace(go.Scatter(x=t, y=np.squeeze(y), name="step response"))
        fig.update_xaxes(title_text="time [hours]")
        fig.update_yaxes(title_text="Reference model output [mmol/L]")
        st.plotly_chart(fig)

    with st.beta_expander("Reference model step response"):
        plot_ref_model_step_response()
    st.subheader("Change shapem amplitude and period of reference signal")
    col_ref_signal_choice = st.beta_columns(3)
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

    with st.beta_expander("Reference model response to reference isgnal"):
        plot_ref_model_response()

    with st.beta_expander("Control algorithm details"):
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

    @st.cache
    def sim_adaptive_control(tt, Hp, dsigm, dsigc, r, Gb, rm_list):
        x, u, ud, vb = sim_MRAC(tt, Hp, dsigm, dsigc, r, Gb, rm_list)
        return x, u, ud, vb

    x, u, ud, vb = sim_adaptive_control(
        tt,
        Hp,
        dsigm,
        dsigc,
        -0.5 * signal.square(tt / 24 * 60 * 2 * np.pi),
        Gb,
        [0.05, 0.00035],
    )
    # adaptive control simulation
    if st.button("Re-run simulation"):
        data_load_state = st.text("Simulation in progress...")
        x, u, ud, vb = sim_adaptive_control(tt, Hp, dsigm, dsigc, r, Gb, rm_list)
        data_load_state.text("Simulation in progress...done")

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
                name="lowest desired glycemia",
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
                name="highest desired glycemia",
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
            showlegend=False,
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
