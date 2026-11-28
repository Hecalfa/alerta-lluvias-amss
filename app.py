
# Para correr el app.py ejecutar en la terminal:   python -m streamlit run app.py

"""Aplicación Streamlit: predicción de lluvia y alerta experimental por acumulación."""
from __future__ import annotations   #mantener la compatibilidad y legibilidad de tipos de datos

import base64  #incrustar el logo en el encabezado
from datetime import date, datetime, timedelta  #manejo de fechas
from zoneinfo import ZoneInfo
from pathlib import Path  #Rutas

import joblib  #modelo
import numpy as np
import pandas as pd
import requests
import streamlit as st

#Busca la ruta del modelo entrenado
DIRECTORIO_BASE = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO_BASE / "models" / "modelo_lluvia.joblib"
RUTA_LOGO = DIRECTORIO_BASE / "logo-uped.png"
ZONA_HORARIA = "America/El_Salvador"

#Enlace con los datos
URL_API = "https://api.open-meteo.com/v1/forecast"

#Diccionario de coordenadas exactas por municipio para capturar microclimas del AMSS
COORDENADAS_MUNICIPIOS = {
    "San Salvador Centro (Histórico / Escalón)": {"lat": 13.6929, "lon": -89.2182},
    "Soyapango (Zonas urbanas y cauces)": {"lat": 13.7142, "lon": -89.1447},
    "Ilopango (Zonas cercanas al lago / Planadas)": {"lat": 13.7031, "lon": -89.1119},
    "Mejicanos (Comunidades en laderas)": {"lat": 13.7333, "lon": -89.2167},
    "Cuscatancingo (Zonas vulnerables a quebradas)": {"lat": 13.7383, "lon": -89.1872},
    "Ciudad Delgado (Riesgo por taludes y quebradas)": {"lat": 13.7225, "lon": -89.1853}
}

#Diseño de pagina 
st.set_page_config(page_title="Alerta de Lluvias AMSS", page_icon=":material/water_drop:", layout="wide")

#Estilos visuales
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .encabezado {
        background: linear-gradient(135deg, #0F3D7A 0%, #1E5AA8 100%);
        color: #FFFFFF; padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.2rem;
    }
    .encabezado {display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap;}
    .encabezado .logo {
        background: #FFFFFF; border-radius: 10px; padding: .5rem .8rem;
        display: flex; align-items: center;
    }
    .encabezado .logo img {height: 64px; width: auto; display: block;}
    .encabezado h1 {color: #FFFFFF; font-size: 1.9rem; margin: 0; padding: 0; font-weight: 700;}
    .encabezado p {color: #D6E4F5; margin: .4rem 0 0 0; font-size: .95rem;}
    .nivel {
        display: inline-block; padding: .35rem .9rem; border-radius: 999px;
        font-weight: 600; font-size: .95rem; color: #FFFFFF;
    }
    [data-testid="stSidebar"] {border-right: 1px solid #E3E8EF;}
    [data-testid="stMetric"] {background: #FFFFFF;}
    </style>
    """,
    unsafe_allow_html=True,
)

#Logo institucional (se incrusta en base64 para mostrarlo dentro del encabezado HTML)
logo_html = ""
if RUTA_LOGO.exists():
    logo_b64 = base64.b64encode(RUTA_LOGO.read_bytes()).decode()
    logo_html = f'<div class="logo"><img src="data:image/png;base64,{logo_b64}" alt="UPED"></div>'

st.markdown(
    f"""
    <div class="encabezado">
        {logo_html}
        <div>
            <h1>Predicción de lluvia y alerta de riesgos</h1>
            <p>Área Metropolitana de San Salvador (AMSS) · Modelo de Machine Learning con coordenadas específicas por municipio</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Este prototipo es un sistema de alerta temprana basado en precipitación reciente, "
    "pronosticada y modelos de Machine Learning (Random Forest) orientados a la prevención en el AMSS.",
    icon=":material/info:",
)

#Menu lateral izquierdo
with st.sidebar:
    st.header(":material/location_on: Consulta ciudadana")
    municipio = st.selectbox(
        "Seleccione su Municipio / Zona del AMSS:",
        list(COORDENADAS_MUNICIPIOS.keys())
    )
    
    coords_actuales = COORDENADAS_MUNICIPIOS[municipio]
    lat_actual = coords_actuales["lat"]
    lon_actual = coords_actuales["lon"]
    
    st.divider()
    st.header(":material/calendar_month: Fecha a predecir")
    opcion_dia = st.radio(
        "Seleccione el día a predecir:",
        ["Hoy", "Mañana"],
        index=1
    )

    st.divider()
    st.caption(":material/cloud: Datos en vivo: Open-Meteo")
    st.caption(":material/model_training: Modelo: Random Forest")


#Carga el modelo ya entrenado
@st.cache_resource
def cargar_modelo():
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)

#Adquisición de datos y definición de parametros
@st.cache_data(ttl=900)
def obtener_clima_reciente(lat: float, lon: float):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "past_days": 10,
        "forecast_days": 4,
        "timezone": ZONA_HORARIA,
    }

    #Petición HTTP a la API
    respuesta = requests.get(URL_API, params=params, timeout=25)
    respuesta.raise_for_status()
    payload = respuesta.json()
    if "daily" not in payload or not payload["daily"].get("time"):
        raise ValueError("La API no devolvió datos diarios.")
    diario = payload["daily"]
    #Procesamiento de datos
    df = pd.DataFrame({
        "fecha": pd.to_datetime(diario["time"], errors="coerce"),
        "precipitacion_mm": pd.to_numeric(diario["precipitation_sum"], errors="coerce"),
        "temp_max_c": pd.to_numeric(diario["temperature_2m_max"], errors="coerce"),
        "temp_min_c": pd.to_numeric(diario["temperature_2m_min"], errors="coerce"),
        "prob_precipitacion_pct": pd.to_numeric(diario.get("precipitation_probability_max"), errors="coerce"),
    })
    df = df.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha")
    return df.reset_index(drop=True)

#Transformación de datos para la predicción
def construir_fila_caracteristicas(historial: pd.DataFrame, fecha_objetivo: pd.Timestamp) -> pd.DataFrame:
    historial = historial.sort_values("fecha").copy()
    historial = historial.dropna(subset=["precipitacion_mm", "temp_max_c", "temp_min_c"])
    #Minimo requerido de dias
    if len(historial) < 5:
        raise ValueError("Se necesitan al menos 5 días recientes completos para crear las características.")
    #Extracción de series numericas
    lluvia = historial["precipitacion_mm"].astype(float)
    tmax = historial["temp_max_c"].astype(float)
    tmin = historial["temp_min_c"].astype(float)
    doy = pd.Timestamp(fecha_objetivo).dayofyear
    fila = {
        "precip_dia_anterior_mm": float(lluvia.iloc[-1]),
        "precip_previo3d_mm": float(lluvia.iloc[-3:].sum()),
        "precip_previo5d_mm": float(lluvia.iloc[-5:].sum()),  #Acumulados
        "temp_max_dia_anterior_c": float(tmax.iloc[-1]),
        "temp_min_dia_anterior": float(tmin.iloc[-1]),
        "prome_temp_max_3d": float(tmax.iloc[-3:].mean()), #Promedios de temperatura
        "prome_temp_min_3d": float(tmin.iloc[-3:].mean()),
        "dia_del_year_sin": float(np.sin(2 * np.pi * doy / 365.25)), #Estacionalidad
        "dia_del_year_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([fila])

#Niveles de alerta
def nivel_exposicion_riesgo(mm: float) -> tuple[str, str]:
    if mm < 20:
        return "Verde (Bajo Riesgo)", "Suelo con humedad estable. Sin amenazas críticas reportadas en la zona."
    if mm < 50:
        return "Amarilla (Precaución)", "Humedad moderada acumulada. Conviene vigilar los pronósticos locales."
    if mm < 90:
        return "Naranja (Alto Riesgo)", "Saturación importante acumulada por lluvias recientes. Atención preventiva."
    return "Roja (Crítico / Emergencia)", "Saturación extrema. Peligro inminente de anegamientos o deslizamientos."

#Verificacion del modelo entrenado
paquete_modelo = cargar_modelo()
if paquete_modelo is None:
    st.error("No se encontró el modelo entrenado.", icon=":material/error:")
    st.markdown("Primero ejecuta en tu terminal: `python entreno_modelo.py` y luego vuelve a abrir la app.")
    st.stop()

#Consulta de clima 
try:
    with st.spinner(f"Consultando microclima en vivo para {municipio}..."):
        clima = obtener_clima_reciente(lat_actual, lon_actual)
except (requests.RequestException, ValueError, KeyError) as exc:
    st.error(f"No fue posible consultar Open-Meteo: {exc}", icon=":material/cloud_off:")
    st.stop()

#Saca la fecha ya sea hoy o mañana
hoy_el_salvador = datetime.now(ZoneInfo(ZONA_HORARIA)).date()
clima["fecha_date"] = clima["fecha"].dt.date

if opcion_dia == "Hoy":
    fecha_objetivo = pd.Timestamp(hoy_el_salvador)
    historial = clima[clima["fecha_date"] < hoy_el_salvador].copy()
    pronostico = clima[clima["fecha_date"] >= hoy_el_salvador].copy()
else:
    fecha_objetivo = pd.Timestamp(hoy_el_salvador + timedelta(days=1))
    historial = clima[clima["fecha_date"] <= hoy_el_salvador].copy()
    pronostico = clima[clima["fecha_date"] > hoy_el_salvador].copy()

if len(historial) < 5:
    st.error("La API no entregó suficientes días recientes. Intenta más tarde.", icon=":material/error:")
    st.stop()

#Ejecucion del modelo
try:
    x_siguiente = construir_fila_caracteristicas(historial, fecha_objetivo)
    x_siguiente = x_siguiente[paquete_modelo["features"]]
    modelo = paquete_modelo["model"]
    probabilidad = float(modelo.predict_proba(x_siguiente)[0][list(modelo.classes_).index(1)]) #Saca la probabilidad de que llueva mas de 1mm
    clase_lluvia = int(probabilidad >= 0.5)   #Umbral de desicion del 50%
except (ValueError, KeyError, IndexError) as exc:
    st.error(f"No se pudo construir la predicción: {exc}", icon=":material/error:")
    st.stop()

#Limpieza y calculo de lluvia de 3 y 5 dias
reciente_sin_nulos = historial.dropna(subset=["precipitacion_mm"]).sort_values("fecha")
reciente_3d = float(reciente_sin_nulos.tail(3)["precipitacion_mm"].sum())
reciente_5d = float(reciente_sin_nulos.tail(5)["precipitacion_mm"].sum())

#Calculo de lluvia
pronostico_3d = float(pronostico.head(3)["precipitacion_mm"].sum()) if pronostico["precipitacion_mm"].notna().sum() >= 1 else 0.0
#definición de precipitacion y nivel de riesgo
base_precipitacion = max(reciente_3d, pronostico_3d)
nivel, recomendacion = nivel_exposicion_riesgo(base_precipitacion)

fecha_str = fecha_objetivo.strftime("%d/%m/%Y")
etiqueta_dia = f"Hoy ({fecha_str})" if opcion_dia == "Hoy" else f"Mañana ({fecha_str})"

# Botón principal de evaluación
COLORES_NIVEL = {"Verde": "#2E7D32", "Amarilla": "#B7950B", "Naranja": "#D35400", "Roja": "#C0392B"}

if st.button(f"Evaluar riesgo y pronóstico para {opcion_dia.lower()}", type="primary", icon=":material/search:"):

    # Pronóstico Meteorológico
    st.subheader(f":material/partly_cloudy_day: Pronóstico meteorológico — {etiqueta_dia}")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric(label="Zona seleccionada", value=municipio.split(" (")[0], border=True)
    with mcol2:
        st.metric(label="Probabilidad de lluvia (ML)", value=f"{probabilidad * 100:.1f}%", border=True)
    with mcol3:
        st.metric(label="Lluvia reciente (3 días)", value=f"{reciente_3d:.1f} mm", border=True)

    if clase_lluvia == 1:
        st.error(f"**Alta probabilidad de lluvia para {opcion_dia.lower()}:** el modelo predice precipitaciones significativas (> 1 mm).", icon=":material/rainy:")
    else:
        st.success(f"**Condiciones estables para {opcion_dia.lower()}:** el modelo indica baja probabilidad de precipitaciones.", icon=":material/sunny:")

    st.divider()

    # Alerta de Riesgo Territorial
    st.subheader(":material/shield: Alerta de riesgo territorial")
    color_nivel = COLORES_NIVEL.get(nivel.split(" ")[0], "#1E5AA8")
    st.markdown(f'<span class="nivel" style="background:{color_nivel};">Nivel {nivel}</span>', unsafe_allow_html=True)
    st.write("")

    contexto_riesgo = {
        "San Salvador Centro (Histórico / Escalón)": "Riesgo de inundaciones urbanas repentinas en arterias principales y pasos a desnivel.",
        "Soyapango (Zonas urbanas y cauces)": "Monitoreo preventivo en comunidades aledañas a la cuenca del río Las Cañas e inundaciones urbanas repentinas en arterias principales.",
        "Ilopango (Zonas cercanas al lago / Planadas)": "Bajo riesgo de deslaves directos, atención a anegamientos temporales en vías.",
        "Mejicanos (Comunidades en laderas)": "Atención especial en taludes, muros y viviendas ubicadas en pendientes.",
        "Cuscatancingo (Zonas vulnerables a quebradas)": "Vulnerabilidad ante crecidas repentinas de quebradas locales.",
        "Ciudad Delgado (Riesgo por taludes y quebradas)": "Alto monitoreo en laderas inestables y cercanías del río Acelhuate."
    }
    desc_municipio = contexto_riesgo.get(municipio, "Evaluación de riesgo territorial en el AMSS.")
    
    st.info(f"**Recomendación de Protección Civil:**\n\n{recomendacion}\n\n**Contexto de la zona:**\n{desc_municipio}", icon=":material/health_and_safety:")

st.divider()
st.subheader(f":material/analytics: Análisis gráfico del comportamiento climático — {municipio.split(' (')[0]}")

#Graficos
if not clima.empty:
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        with st.container(border=True):
            st.markdown("**:material/water_drop: Precipitación diaria (mm)**")
            grafico_lluvia = clima.set_index('fecha')[['precipitacion_mm']].rename(columns={"precipitacion_mm": "Precipitación (mm)"})
            st.bar_chart(grafico_lluvia, color="#1E5AA8")
    with gcol2:
        with st.container(border=True):
            st.markdown("**:material/device_thermostat: Tendencia de temperaturas (°C)**")
            grafico_temp = clima.set_index('fecha')[['temp_max_c', 'temp_min_c']].rename(
                columns={"temp_max_c": "Máxima (°C)", "temp_min_c": "Mínima (°C)"}
            )
            st.line_chart(grafico_temp, color=["#C0392B", "#1E5AA8"])

# Para correr el app.py ejecutar en la terminal:   python -m streamlit run app.py