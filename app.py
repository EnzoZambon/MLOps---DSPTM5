import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import requests

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Monitor de Estabilidad y Predicción", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetric"] { 
        background-color: #ffffff !important; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #d1d5db;
    }
    [data-testid="stMetricValue"] * {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 2.8rem !important;
    }
    [data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] * {
        color: #b91c1c !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIÓN TÉCNICA: CÁLCULO DE PSI ---
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.histogram(expected, bins=buckets)[1]
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)
    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Configuración")
    p_val_limit = st.number_input("Nivel de Confianza (p-value)", 0.001, 0.100, 0.05, step=0.01)
    psi_limit = st.number_input("Umbral PSI (Alerta)", 0.10, 0.50, 0.20, step=0.05)
    st.info("Comparativa: Entrenamiento (Referencia) vs Recientes (Actual).")

# --- CARGA DE DATOS (MOCK ORIGINAL) ---
def load_data():
    np.random.seed(42)
    ref = pd.DataFrame({
        'ingresos': np.random.normal(3000, 500, 1000),
        'edad': np.random.randint(18, 70, 1000),
        'score_crediticio': np.random.beta(2, 5, 1000) * 1000,
        'antiguedad': np.random.poisson(5, 1000),
        'tipo_laboral': np.random.choice(['Formal', 'Informal', 'Independiente'], 1000),
        'Pago_atiempo': np.random.choice([0, 1], 1000, p=[0.2, 0.8])
    })
    cur = ref.copy()
    cur['ingresos'] = np.random.normal(2750, 550, 1000)
    cur['score_crediticio'] = np.random.beta(2.5, 5, 1000) * 1000
    cur['Pago_atiempo'] = np.random.choice([0, 1], 1000, p=[0.28, 0.72])
    return ref, cur

try:
    df_ref, df_cur = load_data()
    target_col = 'Pago_atiempo'
    
    st.title("🛡️ Dashboard MLOps: Monitoreo y Predicción")
    
    tab_monitor, tab_predict = st.tabs(["📊 Monitoreo de Data Drift", "🚀 Predicción en Tiempo Real"])

    # --- PESTAÑA 1: MONITOREO (ESTRUCTURA ORIGINAL) ---
    with tab_monitor:
        st.subheader("1. Estado de Salud del Modelo")
        
        num_cols = df_ref.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols: num_cols.remove(target_col)
        cat_cols = df_ref.select_dtypes(exclude=[np.number]).columns.tolist()
        
        drift_vars = [col for col in num_cols if calculate_psi(df_ref[col], df_cur[col]) > psi_limit]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Variables Numéricas", len(num_cols))
        col2.metric("Variables Categóricas", len(cat_cols))
        col3.metric("Variables con Drift", len(drift_vars))
        with col4:
            if len(drift_vars) == 0: st.success("Saludable 🟢")
            else: st.warning("Revisar ⚠️")

        # Target Drift
        st.divider()
        st.subheader("1.1 Análisis del Objetivo (Target Drift)")
        target_diff = abs(df_ref[target_col].mean() - df_cur[target_col].mean()) * 100

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #dc2626; border: 1px solid #d1d5db;">
                    <p style="color: #475569; margin:0; font-weight: bold;">MÁXIMA DESVIACIÓN</p>
                    <h2 style="color: #dc2626; margin:0; font-size: 2.5rem;">{target_diff:.2f}%</h2>
                    <p style="margin:0; font-weight: bold;">{'✅ Estable' if target_diff < 5 else '❌ Inestable'}</p>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            dist_ref = df_ref[target_col].value_counts(normalize=True).sort_index()
            dist_cur = df_cur[target_col].value_counts(normalize=True).sort_index()
            fig_target = go.Figure(data=[
                go.Bar(name='Referencia', x=dist_ref.index.astype(str), y=dist_ref.values, marker_color='#1e3a8a'),
                go.Bar(name='Actual', x=dist_cur.index.astype(str), y=dist_cur.values, marker_color='#3b82f6')
            ])
            fig_target.update_layout(barmode='group', height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_target, use_container_width=True)

        # Análisis Detallado
        st.divider()
        st.subheader("2. Análisis Detallado por Tipo de Variable")
        tab_num, tab_cat = st.tabs(["📊 Variables Numéricas", "📋 Variables Categóricas"])

        with tab_num:
            res_data = []
            for col in num_cols:
                stat, p = ks_2samp(df_ref[col], df_cur[col])
                psi = calculate_psi(df_ref[col], df_cur[col])
                res_data.append({"Variable": col, "p-value": f"{p:.4f}", "PSI": f"{psi:.4f}", "Drift": "🔴" if psi > psi_limit else "🟢"})
            st.table(pd.DataFrame(res_data))

        with tab_cat:
            if len(cat_cols) > 0:
                cat_res = []
                for col in cat_cols:
                    ref_dist = df_ref[col].value_counts(normalize=True)
                    cur_dist = df_cur[col].value_counts(normalize=True)
                    diff = (ref_dist - cur_dist).abs().sum() / 2
                    cat_res.append({"Variable": col, "Categorías": len(ref_dist), "Desviación": f"{diff:.2%}", "Estado": "🟢 Estable" if diff < 0.1 else "🟡 Revisar"})
                st.table(pd.DataFrame(cat_res))

        # Galería de Variables
        st.divider()
        st.subheader("3. Galería de Variables (Vista Rápida)")
        var_to_plot = st.selectbox("Selecciona una variable para inspeccionar:", df_ref.columns)
        
        fig_var = go.Figure()
        if not pd.api.types.is_numeric_dtype(df_ref[var_to_plot]):
            r = df_ref[var_to_plot].value_counts(normalize=True)
            a = df_cur[var_to_plot].value_counts(normalize=True)
            fig_var.add_trace(go.Bar(x=r.index, y=r.values, name='Referencia', marker_color='#1e3a8a'))
            fig_var.add_trace(go.Bar(x=a.index, y=a.values, name='Actual', marker_color='#3b82f6'))
        else:
            fig_var.add_trace(go.Histogram(x=df_ref[var_to_plot], name='Referencia', histnorm='probability density', marker_color='#1e3a8a', opacity=0.6))
            fig_var.add_trace(go.Histogram(x=df_cur[var_to_plot], name='Actual', histnorm='probability density', marker_color='#3b82f6', opacity=0.6))
            fig_var.update_layout(barmode='overlay')
        
        fig_var.update_layout(height=350)
        st.plotly_chart(fig_var, use_container_width=True)

        # Correlaciones
        st.divider()
        st.subheader("4. Análisis de Correlaciones")
        c_corr1, c_corr2 = st.columns(2)
        with c_corr1:
            st.write("**Matriz Referencia**")
            st.plotly_chart(px.imshow(df_ref.corr(numeric_only=True), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
        with c_corr2:
            st.write("**Matriz Actual**")
            st.plotly_chart(px.imshow(df_cur.corr(numeric_only=True), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

    # --- PESTAÑA 2: PREDICCIÓN (CONEXIÓN API DOCKER) ---
    with tab_predict:
        st.subheader("🚀 Interfaz de Predicción (API Docker)")
        col_in, col_out = st.columns([1, 1])
        
        with col_in:
            st.info("Complete los datos esenciales para la predicción.")
            val_edad = st.number_input("Edad del Cliente", 18, 90, 35)
            val_ingresos = st.number_input("Salario Mensual ($)", 0, 100000, 5000)
            val_score = st.slider("Puntaje / Score Crédito", 300, 850, 600)
            val_antiguedad = st.number_input("Antigüedad Laboral (Años)", 0, 50, 5)
            val_mora = st.number_input("Saldo en Mora ($)", 0, 50000, 0)
            val_laboral = st.selectbox("Situación Laboral", ["Formal", "Informal", "Independiente"])
            
            if st.button("Ejecutar Predicción"):
                # Payload con las 21 variables requeridas por el modelo
                payload = {
                    "data": [{
                        "edad_cliente": val_edad,
                        "salario_cliente": val_ingresos,
                        "puntaje": val_score,
                        "tipo_laboral": val_laboral,
                        "antiguedad": val_antiguedad,
                        "saldo_mora": float(val_mora),
                        "creditos_sectorFinanciero": 1.0,
                        "saldo_total": float(val_mora),
                        "plazo_meses": 24,
                        "creditos_sectorReal": 0.0,
                        "creditos_sectorCooperativo": 0.0,
                        "capital_prestado": 2000.0,
                        "huella_consulta": 1,
                        "puntaje_datacredito": float(val_score),
                        "cuota_pactada": 100.0,
                        "saldo_principal": 0.0,
                        "promedio_ingresos_datacredito": float(val_ingresos),
                        "tipo_credito": "Consumo",
                        "tendencia_ingresos": "Estable",
                        "saldo_mora_codeudor": 0.0,
                        "total_otros_prestamos": 0.0,
                        "cant_creditosvigentes": 1
                    }]
                }
                
                try:
                    # Usamos 127.0.0.1 para evitar problemas de resolución de nombres en Windows
                    response = requests.post("http://api:8000/predict", json=payload, timeout=5)
                    if response.status_code == 200:
                        res = response.json()["predictions"][0]
                        with col_out:
                            st.subheader("Resultado:")
                            label = "APROBADO" if res == 1 else "RECHAZADO"
                            color = "#16a34a" if res == 1 else "#dc2626"
                            st.markdown(f"""
                                <div style="background-color: {color}; padding: 40px; border-radius: 20px; text-align: center;">
                                    <h1 style="color: white; margin:0;">{label}</h1>
                                </div>
                            """, unsafe_allow_html=True)
                            if res == 1: st.balloons()
                    else:
                        st.error(f"Error API: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error de conexión: {e}")

        with col_out:
            if 'response' not in locals():
                st.write("### Estado de Conexión")
                st.write("Asegúrate de que el contenedor Docker esté en 'Running'.")
                st.image("https://cdn-icons-png.flaticon.com/512/564/564390.png", width=100)

except Exception as e:
    st.error(f"⚠️ Error: {e}")