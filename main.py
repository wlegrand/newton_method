import streamlit as st
import numpy as np
from sympy import symbols, sympify, diff, lambdify
import plotly.graph_objs as go
import plotly.express as px

# Config de page
st.set_page_config(page_title="Newton", layout="centered")
st.title("🔍 Méthode de Newton")

# Entrées utilisateur
func_input = st.text_input("Entrez une fonction f(x):", value="x**3 - 2*x + sin(x)")
x0 = st.number_input("Valeur initiale x₀:", value=1.0)
iterations = st.slider("Nombre d'itérations:", 1, 20, value=7)

# Au clic sur le bouton
if st.button("Calculer"):
    x = symbols('x')
    try:
        # Analyse symbolique
        f_expr = sympify(func_input)
        f_prime_expr = diff(f_expr, x)
        f = lambdify(x, f_expr, 'numpy')
        f_prime = lambdify(x, f_prime_expr, 'numpy')

        # Newton-Raphson
        x_vals = [x0]
        for i in range(iterations):
            xi = x_vals[-1]
            dfx = f_prime(xi)
            if dfx == 0:
                st.error(f"f'({xi}) = 0 : division par zéro, arrêt.")
                break
            x_next = xi - f(xi) / dfx
            x_vals.append(x_next)

        st.success(f"✅ Racine approchée après {iterations} itérations : **x ≈ {x_vals[-1]}**")

        # Définir l'intervalle de tracé f(x) basé sur les x trouvés
        x_vals_arr = np.array(x_vals)
        x_min_data = np.min(x_vals_arr)
        x_max_data = np.max(x_vals_arr)
        x_margin = (x_max_data - x_min_data) * 0.5 if x_max_data != x_min_data else 5
        x_min = x_min_data - x_margin
        x_max = x_max_data + x_margin

        # Calcul f(x) sur tout l’intervalle utile
        x_range = np.linspace(x_min, x_max, 1000)
        y_range = f(x_range)

        fig = go.Figure()

        # Tracer f(x)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range, mode='lines',
            name='f(x)', line=dict(color='blue')
        ))

        # Axe horizontal (y = 0)
        fig.add_trace(go.Scatter(
            x=[x_min, x_max], y=[0, 0],
            mode='lines', line=dict(color='gray', dash='dash'),
            name='y = 0'
        ))

        # Tangentes colorées
        colors = px.colors.qualitative.Dark24
        for i in range(len(x_vals) - 1):
            xi = x_vals[i]
            yi = f(xi)
            x_next = x_vals[i + 1]

            fig.add_trace(go.Scatter(
                x=[xi, x_next],
                y=[yi, 0],
                mode='lines+markers',
                name=f'Tangente {i+1}',
                line=dict(dash='dash', color=colors[i % len(colors)]),
                marker=dict(color=colors[i % len(colors)])
            ))

        # Layout interactif
        fig.update_layout(
            title="Méthode de Newton-Raphson – Tracé interactif",
            xaxis_title="x",
            yaxis_title="f(x)",
            hovermode="closest",
            legend=dict(
                title="Légende (clicable)",
                itemclick="toggle",
                itemdoubleclick="toggleothers"
            ),
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(autorange=True)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur dans la fonction : {e}")
