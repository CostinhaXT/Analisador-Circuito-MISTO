import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cmath
from io import BytesIO
import requests
from PIL import Image


st.set_page_config(
    page_title="Analisador de Circuito RLC MISTO",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .title-section {
        flex: 1;
        min-width: 0;
    }
    .image-section {
        margin-left: 2rem;
    }
</style>
""", unsafe_allow_html=True)

header = st.container()
with header:
    title_col, img_col = st.columns([4, 1])
    
    with title_col:
        st.title("🔍 Analisador de Circuito MISTO (R+L) || (R+L+C)")
        st.markdown("""
        **Análise de circuito misto com potências fasoriais**  
        *Circuito: (R1 + L1) em paralelo com (R2 + L2 + C)*
        """)
    
    with img_col:
        try:
            img_url = "https://i.imgur.com/7YPAP6l.png" 
            st.image(img_url, width=180, use_container_width=False)
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {e}")

def format_fasor(z):
    return f"{abs(z):.2f} ∠ {np.degrees(cmath.phase(z)):.2f}°"

def format_retangular(z):
    return f"{z.real:.2f} + {z.imag:.2f}j"

def plot_fasores(fasores, labels, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    max_mag = max([abs(f) for f in fasores])
    
    for i, (fasor, label) in enumerate(zip(fasores, labels)):
        mag = abs(fasor) / max_mag * 0.8
        angle = cmath.phase(fasor)
        ax.arrow(angle, 0, 0, mag, alpha=0.7, width=0.015,
                 edgecolor='black', facecolor=plt.cm.tab10(i), lw=2,
                 label=f"{label} ({format_fasor(fasor)})")
        ax.text(angle, mag/2, label, ha='center', va='bottom')
    
    ax.set_rmax(1.0)
    ax.set_title(title, pad=20)
    ax.legend(bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    st.pyplot(fig)

with st.sidebar:
    st.header("🔧 Parâmetros do Circuito")
    f = st.number_input("Frequência (Hz)", min_value=1.0, value=60.0, step=1.0)
    
    st.subheader("Tensão de Referência")
    v_mag = st.number_input("Magnitude (V)", min_value=1.0, value=127.0)
    v_phase = st.number_input("Fase (graus)", min_value=-180.0, max_value=180.0, value=0.0)
    V_ref = cmath.rect(v_mag, np.radians(v_phase)) 
    
    st.subheader("Ramo 1 (R + L)")
    R1 = st.number_input("Resistência R1 (Ω)", min_value=0.1, value=20.0)
    L1 = st.number_input("Indutância L1 (H)", min_value=0.01, value=0.12)
    
    st.subheader("Ramo 2 (R + L + C)")
    R2 = st.number_input("Resistência R2 (Ω)", min_value=0.1, value=15.0)
    L2 = st.number_input("Indutância L2 (H)", min_value=0.001, value=0.08)
    C = st.number_input("Capacitância (µF)", min_value=0.1, value=70.0) * 1e-6

omega = 2 * np.pi * f

Z1 = R1 + 1j * omega * L1
Z2 = R2 + 1j * omega * L2 + 1/(1j * omega * C)
Z_total = 1 / (1/Z1 + 1/Z2)

I_total = V_ref / Z_total
I1 = V_ref / Z1
I2 = V_ref / Z2

V_R1 = I1 * R1
V_L1 = I1 * (1j * omega * L1)
V_R2 = I2 * R2
V_L2 = I2 * (1j * omega * L2)
V_C = I2 * (1/(1j * omega * C))

P1 = abs(I1)**2 * R1
Q1 = abs(I1)**2 * (omega * L1)
P2 = abs(I2)**2 * R2
Q2 = abs(I2)**2 * (omega * L2 - 1/(omega * C))

P_total = P1 + P2
Q_total = Q1 + Q2
S_total = cmath.sqrt(P_total**2 + Q_total**2)
FP = P_total / abs(S_total)


st.markdown("---")
st.header("📈 Resultados da Análise")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Impedâncias")
    st.metric("Z Total", format_fasor(Z_total))
    st.metric("Z1 (R1 + L1)", format_fasor(Z1))
    st.metric("Z2 (R2 + L2 + C)", format_fasor(Z2))

with col2:
    st.subheader("Correntes")
    st.metric("Total", format_fasor(I_total))
    st.metric("Ramo 1", format_fasor(I1))
    st.metric("Ramo 2", format_fasor(I2))

st.subheader("Tensões nos Componentes")
cols = st.columns(5)
with cols[0]: st.metric("R1", format_fasor(V_R1))
with cols[1]: st.metric("L1", format_fasor(V_L1))
with cols[2]: st.metric("R2", format_fasor(V_R2))
with cols[3]: st.metric("L2", format_fasor(V_L2))
with cols[4]: st.metric("C1", format_fasor(V_C))

st.markdown("---")
st.header("🔋 Triângulo de Potências")

p_col, q_col, s_col, fp_col = st.columns(4)
with p_col: st.metric("Potência Ativa (P)", f"{P_total:.2f} W")
with q_col: st.metric("Potência Reativa (Q)", f"{Q_total:.2f} VAR")
with s_col: st.metric("Potência Aparente (S)", f"{abs(S_total):.2f} VA")
with fp_col: st.metric("Fator de Potência (FP)", f"{FP:.2f}")

st.markdown("---")
st.header("📊 Diagramas Fasoriais")

plot_fasores([V_ref, V_R1, V_L1, V_R2, V_L2, V_C], 
            ["Fonte", "R1", "L1", "R2", "L2", "C1"],
            "Tensões Fasoriais")

plot_fasores([I_total, I1, I2],
            ["It", "I1", "I2"],
            "Correntes Fasoriais")

def plot_triangulo_potencia(P, Q, S):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Desenha o triângulo
    ax.plot([0, P], [0, 0], 'r-', linewidth=3)
    ax.plot([P, P], [0, Q], 'b-', linewidth=3) 
    ax.plot([0, P], [0, Q], 'g-', linewidth=3)

    # Anotações reposicionadas
    ax.text(P/2, -0.05*abs(Q), f'{P:.2f} W', ha='center', va='top', color='red')  
    ax.text(P*1.02, Q/2, f'{Q:.2f} VAR', ha='left', va='center', color='blue')  
    ax.text(P/3, Q/1.8, f'{abs(S):.2f} VA', ha='center', va='center', color='green')

    fasoriais = [complex(P, 0), complex(0, Q), complex(P, Q)]
    labels = ['P', 'Q', 'S']

    retangulares = [f'{np.real(f):.2f} + {np.imag(f):.2f}j' for f in fasoriais]

    sc = ax.scatter([np.real(f) for f in fasoriais], [np.imag(f) for f in fasoriais], s=100)
   
    ax.set_xlim(0, max(P, abs(S)) * 1.1)
    ax.set_ylim(min(0, Q) - 5, max(0, Q) + 5)
    ax.set_xlabel('Potência Ativa (W)')
    ax.set_ylabel('Potência Reativa (VAR)')
    ax.set_title('Triângulo de Potências')
    ax.grid(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label=f'P = {P:.2f} W'),
        Line2D([0], [0], color='blue', lw=3, label=f'Q = {Q:.2f} VAR'),
        Line2D([0], [0], color='green', lw=3, label=f'S = {abs(S):.2f} VA')
    ]

    if Q >= 0:
        legenda_pos = 'upper left'
    else:
        legenda_pos = 'lower left'

    ax.legend(handles=legend_elements, loc=legenda_pos)

    st.pyplot(fig)

plot_triangulo_potencia(P_total, Q_total, S_total)

st.markdown(f"""
**Análise do Fator de Potência:**
- FP = {FP:.2f} ({'indutivo' if Q_total > 0 else 'capacitivo'})
- Ângulo θ = {np.degrees(cmath.phase(complex(P_total, Q_total))):.2f}°
""")

with st.expander("🔍 Fórmulas Utilizadas"):
    st.latex(r'''
    \begin{align*}
    S &= \sqrt{P^2 + Q^2} \\
    P &= P_1 + P_2 = |I_1|^2 R_1 + |I_2|^2 R_2 \\
    Q &= Q_1 + Q_2 = |I_1|^2 X_{L1} + |I_2|^2 (X_{L2} - X_{C2}) \\
    \text{FP} &= \cos(\theta) = \frac{P}{|S|}
    \end{align*}
    ''')

st.markdown("---")
st.caption("João Guilherme | Flávio H. | Mikhaelly M. | Gustavo H. \\\n Circuitos II - Engenharia Elétrica | 2025-1")
