
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

# Estilos visuales - Dashboard meteorológico AMSS
st.markdown(
    """
    <style>
    :root {
        --fondo: #0B1220;
        --panel: #142238;
        --panel2: #192B43;
        --cian: #16C6D9;
        --texto: #F1F5F9;
        --secundario: #A8B8CC;
    }

    /* Fondo y área principal */
    .stApp {
        background: linear-gradient(145deg, #0B1220, #101D31 55%, #0B1728);
        color: var(--texto);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.9);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Títulos y textos */
    h1, h2, h3, h4 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--secundario);
    }

    /* Encabezado institucional */
    .encabezado {
        background: linear-gradient(115deg, #0D1B30 0%, #123D65 48%, #087F9C 100%);
        border: 1px solid rgba(22, 198, 217, 0.38);
        color: white;
        padding: 2.3rem 2.5rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }

    .encabezado .logo {
        background: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.75);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.20);
        flex-shrink: 0;
    }

    .encabezado .logo img {
        height: 72px;
        width: auto;
        max-width: 190px;
        object-fit: contain;
        display: block;
    }

    .encabezado h1 {
        color: #FFFFFF !important;
        font-size: 2.25rem;
        line-height: 1.2;
        letter-spacing: -0.7px;
        margin: 0;
        padding: 0;
        font-weight: 850;
        text-shadow: 0 2px 12px rgba(0,0,0,0.18);
    }

    .encabezado p {
        color: #D5EAF7 !important;
        margin: .75rem 0 0;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: #0D192B;
        border-right: 1px solid rgba(148, 163, 184, .18);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #EAF6FF !important;
    }

    /* Selectores */
    [data-baseweb="select"] > div {
        background-color: #17263D;
        border-color: #344B68;
        color: white;
        border-radius: 10px;
    }

    /* Tarjetas de indicadores */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #192D48, #122238);
        border: 1px solid rgba(22, 198, 217, .2);
        padding: 1.2rem 1.3rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, .16);
    }

    [data-testid="stMetricLabel"] {
        color: #AFC4DA !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 750;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(100deg, #087FC4, #13AFC5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: .65rem 1.2rem;
        font-weight: 700;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(100deg, #1199DD, #21C9D5);
        color: white;
        border: none;
        transform: translateY(-2px);
    }

    /* Contenedores */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 29, 48, .82);
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 16px;
        padding: .8rem;
    }

    /* Alertas */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .nivel {
        display: inline-block;
        padding: .55rem 1.1rem;
        border-radius: 999px;
        font-weight: 750;
        font-size: 1rem;
        color: white;
    }

    /* Separadores */
    hr {
        border-color: rgba(148, 163, 184, .18);
    }

    /* Pantallas pequeñas */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .encabezado {
            padding: 1.3rem;
        }

        .encabezado h1 {
            font-size: 1.5rem;
        }

        .encabezado .logo img {
            height: 48px;
        }
    }
    
    /* Panel lateral de configuración */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D192B, #101F35) !important;
        border-right: 1px solid rgba(22, 198, 217, 0.25);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #16C6D9 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.3px;
    }

    [data-testid="stSidebar"] label {
        color: #DCE8F5 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #192B43 !important;
        border: 1px solid #16C6D9 !important;
        border-radius: 12px !important;
        min-height: 45px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        background: #142238;
        border: 1px solid rgba(22, 198, 217, 0.2);
        border-radius: 12px;
        padding: 12px;
        gap: 10px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #F1F5F9 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.2);
        margin: 1.2rem 0;
    }

    /* Botón principal de evaluación */
    [data-testid="stMainBlockContainer"] .stButton > button {
        width: 100%;
        min-height: 48px;
        font-size: 1rem;
        letter-spacing: 0.2px;
        box-shadow: 0 5px 18px rgba(8, 127, 196, 0.22);
    }


/* EFECTO VISUAL TARJETAS METRICAS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #192B43, #142238) !important;
    border: 1px solid rgba(22, 198, 217, 0.22) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(22, 198, 217, 0.75) !important;
    box-shadow: 0 8px 22px rgba(22, 198, 217, 0.12);
}

[data-testid="stMetricLabel"] {
    color: #A8B8CC !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-weight: 800 !important;
}


/* ESTILO VISUAL ALERTAS Y RECOMENDACIONES */
[data-testid="stAlert"] {
    background: linear-gradient(145deg, #192B43, #142238) !important;
    border: 1px solid rgba(22, 198, 217, 0.28) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #E2EAF4 !important;
    line-height: 1.7;
}

[data-testid="stAlert"] strong {
    color: #16C6D9 !important;
    font-weight: 700;
}

[data-testid="stAlert"] svg {
    color: #16C6D9 !important;
}

[data-testid="stAlert"]:hover {
    border-color: rgba(22, 198, 217, 0.55) !important;
    transition: border-color 0.2s ease;
}


/* JERARQUIA VISUAL DE SECCIONES */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2 {
    color: #F1F5F9 !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.35px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(22, 198, 217, 0.38);
    margin-top: 1.4rem;
    margin-bottom: 1rem;
}

[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3 {
    color: #DCEAF7 !important;
    font-size: 1.12rem !important;
    font-weight: 750 !important;
    margin-top: 1rem;
}

[data-testid="stMain"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, rgba(22,198,217,0.55), rgba(22,198,217,0.03)) !important;
    margin: 1.8rem 0 !important;
}


/* PANELES VISUALES DE GRAFICAS */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #172941, #111F33) !important;
    border: 1px solid rgba(22, 198, 217, 0.22) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    box-shadow: 0 7px 22px rgba(0, 0, 0, 0.15);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(22, 198, 217, 0.55) !important;
    box-shadow: 0 9px 25px rgba(22, 198, 217, 0.08);
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] strong {
    color: #EAF6FF !important;
    font-size: 1rem;
}


/* CORRECCION DE FONDOS DE ALERTAS */
[data-testid="stAlert"],
div[role="alert"] {
    background: #142238 !important;
    background-color: #142238 !important;
    color: #F1F5F9 !important;
    border: 1px solid rgba(22, 198, 217, 0.30) !important;
    border-radius: 14px !important;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15) !important;
}

[data-testid="stAlert"] > div,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stAlertContent"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #E2EAF4 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li {
    color: #E2EAF4 !important;
    line-height: 1.7;
}

[data-testid="stAlert"] strong {
    color: #16C6D9 !important;
    font-weight: 750 !important;
}

[data-testid="stAlert"] svg {
    color: #16C6D9 !important;
    fill: currentColor;
}

[data-testid="stAlert"]:hover {
    border-color: rgba(22, 198, 217, 0.65) !important;
    transition: border-color 0.2s ease;
}


/* BOTON PRINCIPAL EVALUAR RIESGO */
[data-testid="stMainBlockContainer"] .stButton > button {
    width: 100% !important;
    min-height: 58px !important;
    background: linear-gradient(110deg, #087FC4 0%, #0B9FC2 55%, #16C6D9 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(100, 230, 245, 0.65) !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.35px !important;
    box-shadow: 0 7px 22px rgba(8, 127, 196, 0.30) !important;
    transition: all 0.22s ease !important;
}

[data-testid="stMainBlockContainer"] .stButton > button:hover {
    background: linear-gradient(110deg, #0795D8 0%, #0DB8D0 55%, #36D7E5 100%) !important;
    border-color: #75F0F5 !important;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(22, 198, 217, 0.32) !important;
}

[data-testid="stMainBlockContainer"] .stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 3px 12px rgba(22, 198, 217, 0.22) !important;
}

[data-testid="stMainBlockContainer"] .stButton > button p {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

[data-testid="stMainBlockContainer"] .stButton > button svg {
    color: #FFFFFF !important;
    fill: currentColor;
}


    /* FIX DELTA PROBABILIDAD ESTIMADA */
    [data-testid="stMetricDelta"] {
        background: #192B43 !important;
        background-color: #192B43 !important;
        color: #16C6D9 !important;
        padding: 5px 9px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(22, 198, 217, 0.25) !important;
    }

    [data-testid="stMetricDelta"] * {
        color: #16C6D9 !important;
        background: transparent !important;
    }

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



# MAPA INTERACTIVO AMSS
st.divider()
st.subheader(":material/map: Mapa territorial del AMSS")
st.caption("Ubicación de los seis municipios incluidos en el sistema. El municipio seleccionado aparece destacado.")

import pydeck as pdk

datos_mapa = []

for nombre, coordenadas in COORDENADAS_MUNICIPIOS.items():
    seleccionado = nombre == municipio

    datos_mapa.append({
        "municipio": nombre.split(" (")[0],
        "zona": nombre,
        "lat": coordenadas["lat"],
        "lon": coordenadas["lon"],
        "color": [22, 198, 217, 230] if seleccionado else [80, 140, 180, 190],
        "radio": 1100 if seleccionado else 750
    })

df_mapa = pd.DataFrame(datos_mapa)

capa_municipios = pdk.Layer(
    "ScatterplotLayer",
    data=df_mapa,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius="radio",
    pickable=True,
    auto_highlight=True,
    stroked=True,
    get_line_color=[241, 245, 249, 220],
    line_width_min_pixels=2
)

vista_mapa = pdk.ViewState(
    latitude=13.715,
    longitude=-89.18,
    zoom=10.3,
    pitch=35
)

mapa_amss = pdk.Deck(
    layers=[capa_municipios],
    initial_view_state=vista_mapa,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    tooltip={
        "html": "<b>{municipio}</b><br/>Municipio del AMSS",
        "style": {
            "backgroundColor": "#142238",
            "color": "#F1F5F9"
        }
    }
)

st.pydeck_chart(mapa_amss, use_container_width=True, height=450)


# LEYENDA VISUAL DEL MAPA
st.markdown(
    """
    <div style="display:flex;flex-wrap:wrap;gap:22px;align-items:center;background:#142238;border:1px solid rgba(22,198,217,0.20);border-radius:12px;padding:14px 18px;margin-top:10px;">
        <div style="display:flex;align-items:center;gap:9px;">
            <span style="width:13px;height:13px;border-radius:50%;background:#16C6D9;display:inline-block;"></span>
            <span style="color:#F1F5F9;font-size:0.9rem;">Municipio seleccionado</span>
        </div>
        <div style="display:flex;align-items:center;gap:9px;">
            <span style="width:13px;height:13px;border-radius:50%;background:#508CB4;display:inline-block;"></span>
            <span style="color:#F1F5F9;font-size:0.9rem;">Otros municipios</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


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

    # Cuatro indicadores principales
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.metric(
            label="Municipio seleccionado",
            value=municipio.split(" (")[0],
            border=True,
            help="Municipio seleccionado para el análisis meteorológico."
        )

    with mcol2:
        st.metric(
            label="Probabilidad de lluvia (ML)",
            value=f"{probabilidad * 100:.1f}%",
            delta="Probabilidad estimada",
            border=True
        )

    with mcol3:
        st.metric(
            label="Lluvia reciente (3 días)",
            value=f"{reciente_3d:.1f} mm",
            border=True,
            help="Precipitación acumulada durante los últimos tres días."
        )


    if clase_lluvia == 1:
        st.error(f"**Alta probabilidad de lluvia para {opcion_dia.lower()}:** el modelo predice precipitaciones significativas (> 1 mm).", icon=":material/rainy:")
    else:
        st.success(f"**Condiciones estables para {opcion_dia.lower()}:** el modelo indica baja probabilidad de precipitaciones.", icon=":material/sunny:")

    
    st.divider()

    # Contexto de riesgo por municipio
    contexto_riesgo = {
        "San Salvador Centro (Histórico / Escalón)": "Riesgo de inundaciones urbanas repentinas en arterias principales y pasos a desnivel.",
        "Soyapango (Zonas urbanas y cauces)": "Monitoreo preventivo en comunidades aledañas a la cuenca del río Las Cañas e inundaciones urbanas repentinas en arterias principales.",
        "Ilopango (Zonas cercanas al lago / Planadas)": "Bajo riesgo de deslaves directos, atención a anegamientos temporales en vías.",
        "Mejicanos (Comunidades en laderas)": "Atención especial en taludes, muros y viviendas ubicadas en pendientes.",
        "Cuscatancingo (Zonas vulnerables a quebradas)": "Vulnerabilidad ante crecidas repentinas de quebradas locales.",
        "Ciudad Delgado (Riesgo por taludes y quebradas)": "Alto monitoreo en laderas inestables y cercanías del río Acelhuate."
    }

    # Obtener contexto antes de mostrarlo
    desc_municipio = contexto_riesgo.get(
        municipio,
        "Evaluación de riesgo territorial en el AMSS."
    )

    # Alerta de Riesgo Territorial
    st.subheader(":material/shield: Alerta de riesgo territorial")

    color_nivel = COLORES_NIVEL.get(nivel.split(" ")[0], "#16C6D9")

    st.markdown(
        f"""
        <div style="
            background: #142238;
            border-left: 6px solid {color_nivel};
            border-radius: 12px;
            padding: 20px 24px;
            margin: 12px 0 20px 0;
        ">
            <div style="
                color: #A8B8CC;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 8px;
            ">
                NIVEL DE RIESGO DETECTADO
            </div>
            <div style="
                color: {color_nivel};
                font-size: 1.5rem;
                font-weight: 800;
                line-height: 1.4;
                overflow-wrap: anywhere;
            ">
                {nivel}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### :material/health_and_safety: Recomendaciones y contexto")

    st.info(
        f"**Recomendación de Protección Civil:**\n\n"
        f"{recomendacion}\n\n"
        f"**Contexto de la zona:**\n\n"
        f"{desc_municipio}",
        icon=":material/health_and_safety:"
    )


    desc_municipio = contexto_riesgo.get(municipio, "Evaluación de riesgo territorial en el AMSS.")


st.divider()
st.subheader(f":material/analytics: Análisis gráfico del comportamiento climático — {municipio.split(' (')[0]}")


# Gráficas meteorológicas interactivas con Plotly
if not clima.empty:
    import plotly.graph_objects as go

    gcol1, gcol2 = st.columns(2)
    fechas = pd.to_datetime(clima["fecha"])

    with gcol1:
        with st.container(border=True):
            st.markdown("**:material/water_drop: Precipitación diaria (mm)**")

            fig_lluvia = go.Figure()
            fig_lluvia.add_trace(go.Bar(
                x=fechas,
                y=clima["precipitacion_mm"].fillna(0),
                name="Precipitación",
                marker_color="#16C6D9",
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Precipitación: %{y:.1f} mm<extra></extra>"
            ))

            fig_lluvia.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=25, b=35),
                paper_bgcolor="#142238",
                plot_bgcolor="#142238",
                font=dict(color="#A8B8CC"),
                hovermode="x",
                showlegend=False,
                yaxis_title="Precipitación (mm)",
                bargap=0.30
            )
            fig_lluvia.update_xaxes(tickformat="%d/%m", showgrid=False, color="#A8B8CC", linecolor="#52647A")
            fig_lluvia.update_yaxes(gridcolor="rgba(82, 100, 122, 0.30)", zerolinecolor="#52647A")

            st.plotly_chart(fig_lluvia, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True})

    with gcol2:
        with st.container(border=True):
            st.markdown("**:material/device_thermostat: Tendencia de temperaturas (°C)**")

            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=fechas,
                y=clima["temp_max_c"],
                mode="lines+markers",
                name="Máxima",
                line=dict(color="#FF8066", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Máxima: %{y:.1f} °C<extra></extra>"
            ))
            fig_temp.add_trace(go.Scatter(
                x=fechas,
                y=clima["temp_min_c"],
                mode="lines+markers",
                name="Mínima",
                line=dict(color="#16C6D9", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Mínima: %{y:.1f} °C<extra></extra>"
            ))

            fig_temp.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=45, b=35),
                paper_bgcolor="#142238",
                plot_bgcolor="#142238",
                font=dict(color="#A8B8CC"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.15, x=0, font=dict(color="#F1F5F9")),
                yaxis_title="Temperatura (°C)"
            )
            fig_temp.update_xaxes(tickformat="%d/%m", showgrid=False, color="#A8B8CC", linecolor="#52647A")
            fig_temp.update_yaxes(gridcolor="rgba(82, 100, 122, 0.30)", zerolinecolor="#52647A")

            st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True})

# Para correr el app.py ejecutar en la terminal:   python -m streamlit run app.py