
"""
Entrenamiento y evaluación del modelo de lluvia para el AMSS.
Ejecutar: python entreno_modelo.py


El objetivo es predecir si la precipitación diaria será mayor a 1 mm para la fecha objetivo,
utilizando únicamente variables disponibles en días anteriores y estacionalidad.
La evaluación usa separación cronológica para evitar mezclar el futuro con el pasado.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Creación de rutas y directorios
DIRECTORIO_BASE = Path(__file__).resolve().parent
DIRECTORIO_DATOS = DIRECTORIO_BASE / "data"
DIRECTORIO_SALIDAS = DIRECTORIO_BASE / "outputs"
DIRECTORIO_MODELOS = DIRECTORIO_BASE / "models"

for carpeta in (DIRECTORIO_DATOS, DIRECTORIO_SALIDAS, DIRECTORIO_MODELOS):
    carpeta.mkdir(exist_ok=True)

# Coordenadas geográficas de referencia
LATITUD = 13.6929
LONGITUD = -89.2182
FECHA_INICIO = "2020-01-01"
# El archivo histórico puede tener retraso; se deja margen de 5 días.
FECHA_FIN = (date.today() - timedelta(days=5)).isoformat()
UMBRAL_LLUVIA_MM = 1.0

# Características que usa el modelo para aprender (en español)
CARACTERISTICAS = [
    "precip_dia_anterior_mm",    # Precipitación del día anterior
    "precip_previo3d_mm",        # Acumulación últimos 3 días
    "precip_previo5d_mm",        # Acumulación últimos 5 días
    "temp_max_dia_anterior_c",   # Temperatura máxima del día anterior
    "temp_min_dia_anterior",     # Temperatura mínima del día anterior
    "prome_temp_max_3d",         # Promedio de temperaturas máximas de los últimos 3 días
    "prome_temp_min_3d",         # Promedio de temperaturas mínimas de los últimos 3 días
    "dia_del_year_sin",          # Seno del día del año
    "dia_del_year_cos",          # Coseno del día del año
]

def descargar_datos_historicos() -> pd.DataFrame:
    """Descarga los datos diarios históricos desde Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    parametros = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": FECHA_INICIO,
        "end_date": FECHA_FIN,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "America/El_Salvador",
    }
    #Peticion HTTP
    respuesta = requests.get(url, params=parametros, timeout=30)
    respuesta.raise_for_status()
    payload = respuesta.json()

    #Validacion de datos
    if "daily" not in payload or not payload["daily"].get("time"):
        raise ValueError(f"La API no devolvió datos diarios utilizables: {payload}")

    #Dataframe con pandas (Una tabla practicamente)
    diario = payload["daily"]
    df = pd.DataFrame({
        "fecha": diario["time"],
        "precipitacion_mm": diario["precipitation_sum"],
        "temp_max_c": diario["temperature_2m_max"],
        "temp_min_c": diario["temperature_2m_min"],
    })
    #Conversion del tipo de datos
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in ["precipitacion_mm", "temp_max_c", "temp_min_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #Limpieza y orden
    df = df.dropna(subset=["fecha"]).sort_values("fecha")  #Dropna elimina filas nulas
    df = df.drop_duplicates(subset=["fecha"], keep="last").reset_index(drop=True)

    df.loc[df["precipitacion_mm"] < 0, "precipitacion_mm"] = np.nan
    df.loc[df["temp_max_c"] < -10, "temp_max_c"] = np.nan
    df.loc[df["temp_min_c"] < -10, "temp_min_c"] = np.nan   #.loc busca colum especif
    return df

#Crea una coipa de los datos historicos
def construir_dataset_supervisado(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("fecha").reset_index(drop=True)
    lluvia = df["precipitacion_mm"]
    tmax = df["temp_max_c"]
    tmin = df["temp_min_c"]

    df["precip_dia_anterior_mm"] = lluvia.shift(1)
    df["precip_previo3d_mm"] = lluvia.shift(1).rolling(3, min_periods=3).sum()
    df["precip_previo5d_mm"] = lluvia.shift(1).rolling(5, min_periods=5).sum()
    df["temp_max_dia_anterior_c"] = tmax.shift(1)
    df["temp_min_dia_anterior"] = tmin.shift(1)
    df["prome_temp_max_3d"] = tmax.shift(1).rolling(3, min_periods=3).mean()
    df["prome_temp_min_3d"] = tmin.shift(1).rolling(3, min_periods=3).mean()
    
    doy = df["fecha"].dt.dayofyear
    df["dia_del_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["dia_del_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["target_llueve"] = np.where(
        df["precipitacion_mm"].notna(),
        (df["precipitacion_mm"] > UMBRAL_LLUVIA_MM).astype(float),
        np.nan,
    )
    #Limpieza final
    df_modelo = df.dropna(subset=CARACTERISTICAS + ["target_llueve"]).copy()
    df_modelo["target_llueve"] = df_modelo["target_llueve"].astype(int)
    return df_modelo

#MEdicion del rendimiento del modelo
def evaluar_modelo(y_real, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_pred)),
        "precision_lluvia": float(precision_score(y_real, y_pred, pos_label=1, zero_division=0)),
        "recall_lluvia": float(recall_score(y_real, y_pred, pos_label=1, zero_division=0)),
        "f1_lluvia": float(f1_score(y_real, y_pred, pos_label=1, zero_division=0)),
    }


def main():
    print("Descargando datos históricos diarios para el punto de referencia del AMSS...")
    datos_raw = descargar_datos_historicos()
    datos_raw.to_csv(DIRECTORIO_DATOS / "clima_historico.csv", index=False)
    
    dataset = construir_dataset_supervisado(datos_raw)
    dataset.to_csv(DIRECTORIO_DATOS / "dataset_modelado.csv", index=False)

    if len(dataset) < 500:
        raise ValueError(f"Solo hay {len(dataset)} filas modelables. Se necesitan más datos.")

    #Division de datos 80 entrenamiento
    indice_division = int(len(dataset) * 0.80)
    df_entrenamiento = dataset.iloc[:indice_division].copy()
    df_prueba = dataset.iloc[indice_division:].copy()
    
    X_train, y_train = df_entrenamiento[CARACTERISTICAS], df_entrenamiento["target_llueve"]
    X_test, y_test = df_prueba[CARACTERISTICAS], df_prueba["target_llueve"]
    #Configuracion modelo
    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    baseline = DummyClassifier(strategy="most_frequent")

    print(f"Datos utilizables: {len(dataset)}")
    print(f"Entrenamiento: {len(df_entrenamiento)} filas ({df_entrenamiento['fecha'].min().date()} a {df_entrenamiento['fecha'].max().date()})")
    print(f"Prueba temporal: {len(df_prueba)} filas ({df_prueba['fecha'].min().date()} a {df_prueba['fecha'].max().date()})")

    #Entrenamiento y evaluacion 
    modelo.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    predicciones_baseline = baseline.predict(X_test)

    metricas = evaluar_modelo(y_test, predicciones)
    metricas_baseline = evaluar_modelo(y_test, predicciones_baseline)
    metricas["baseline_accuracy"] = metricas_baseline["accuracy"]
    metricas["baseline_f1_lluvia"] = metricas_baseline["f1_lluvia"]
    metricas["train_start"] = str(df_entrenamiento["fecha"].min().date())
    metricas["train_end"] = str(df_entrenamiento["fecha"].max().date())
    metricas["test_start"] = str(df_prueba["fecha"].min().date())
    metricas["test_end"] = str(df_prueba["fecha"].max().date())
    metricas["n_train"] = int(len(df_entrenamiento))
    metricas["n_test"] = int(len(df_prueba))
    metricas["rain_threshold_mm"] = UMBRAL_LLUVIA_MM

    print("\n=== Métricas del modelo Random Forest ===")
    for clave, valor in metricas.items():
        print(f"{clave}: {valor}")

    reporte = classification_report(
        y_test, predicciones, labels=[0, 1], target_names=["No lluvia", "Lluvia"],
        zero_division=0, output_dict=True
    )
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_test, predicciones, labels=[0, 1], target_names=["No lluvia", "Lluvia"], zero_division=0))

    #exportacion de metricas
    with open(DIRECTORIO_SALIDAS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": metricas, "reporte_clasificacion": reporte}, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicciones, labels=[0, 1], display_labels=["No lluvia", "Lluvia"],
        ax=ax
    )
    ax.set_title("Matriz de confusión — prueba cronológica")
    fig.tight_layout()
    fig.savefig(DIRECTORIO_SALIDAS / "matriz_confusion.png", dpi=160)
    plt.close(fig)

    #Reentrenamiento y empaquetado de modelo Joblib
    modelo_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    modelo_final.fit(dataset[CARACTERISTICAS], dataset["target_llueve"])
    
    paquete_modelo = {
        "model": modelo_final,
        "features": CARACTERISTICAS,
        "rain_threshold_mm": UMBRAL_LLUVIA_MM,
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "trained_through": str(dataset["fecha"].max().date()),
        "evaluation_metrics": metricas,
        "target_definition": "Precipitación diaria > 1 mm en la fecha objetivo",
    }
    
    joblib.dump(paquete_modelo, DIRECTORIO_MODELOS / "modelo_lluvia.joblib")
    print(f"\nModelo guardado en: {DIRECTORIO_MODELOS / 'modelo_lluvia.joblib'}")


if __name__ == "__main__":
    main()