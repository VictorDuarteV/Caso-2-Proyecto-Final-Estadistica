"""Dashboard del Caso 2 del Proyecto Final de Estadística Aplicada a la Investigación.

Predicción del valor de propiedades inmobiliarias (King County, Washington) con
análisis estadístico (descriptivos, correlación, ANOVA, regresión lineal múltiple)
y comparación de modelos de Machine Learning.

Ejecución local:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Predicción de precios de vivienda', layout='wide')

CARPETA = Path(__file__).parent
CARPETA_MODELOS = CARPETA / 'modelos'

# ---------------------------------------------------------------- colores
FONDO, SUPERFICIE, BORDE = '#F5F3EE', '#FFFFFF', '#E2DDD3'
TEXTO, TEXTO_SEC = '#1C2B33', '#56656D'
PRIMARIO = '#0E4F5C'
ACENTO = '#C4572F'
COLOR_MODELO = {'Gradient Boosting optimizado': '#0E4F5C', 'Random Forest optimizado': '#7FA9B3',
                'Regresión lineal': '#D49A3A'}
NOMBRE_CORTO = {'Gradient Boosting optimizado': 'Gradient Boosting', 'Random Forest optimizado': 'Random Forest',
                'Regresión lineal': 'Regresión lineal'}
# Modelos que se presentan en la app (el notebook compara más; aquí se dejan los tres principales)
MODELOS_APP = list(COLOR_MODELO)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, select, textarea {font-family: 'Montserrat', sans-serif;}
.stApp * {font-variant-numeric: tabular-nums;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none !important;}
[data-testid="stMainBlockContainer"] {max-width: 1184px; padding: 0 32px 80px;}

/* Encabezado con navegación */
.st-key-encabezado {background: #FFFFFF; box-shadow: 0 0 0 100vmax #FFFFFF; clip-path: inset(0 -100vmax);
                    position: relative; margin-bottom: 16px;}
.st-key-encabezado::after {content: ""; position: absolute; left: -100vmax; right: -100vmax; bottom: 0;
                           height: 1px; background: #E2DDD3;}
.marca {height: 62px; transform: translateY(-8px); display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 17px; color: #1C2B33;}
.marca .logo {width: 30px; height: 30px; border-radius: 8px; background: #0E4F5C; color: #FFFFFF;
              display: inline-flex; align-items: center; justify-content: center; font-size: 16px;}
.st-key-encabezado [data-testid="stElementContainer"], .st-key-encabezado [data-testid="stRadio"] {width: 100%;}
.st-key-encabezado div[role="radiogroup"] {gap: 2px; justify-content: flex-end; flex-wrap: wrap; width: 100%;}
.st-key-encabezado [data-testid="stRadioOption"] {margin: 0; padding: 20px 12px 18px; cursor: pointer;
                                                 border-bottom: 2px solid transparent;}
.st-key-encabezado [data-testid="stRadioOption"] > div > div:first-child {display: none;}
.st-key-encabezado [data-testid="stRadioOption"] p {font-size: 14px; font-weight: 500; color: #56656D;}
.st-key-encabezado [data-testid="stRadioOption"][data-selected="true"] {border-bottom-color: #0E4F5C;}
.st-key-encabezado [data-testid="stRadioOption"][data-selected="true"] p {color: #0E4F5C; font-weight: 600;}
div[data-testid="stElementContainer"]:has(style) {display: none;}

/* Tema claro forzado: no depende de .streamlit/config.toml ni del modo oscuro del navegador */
:root, .stApp {color-scheme: light !important;}
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stHeader"], section.main {background-color: #F5F3EE !important;}
.stApp, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"], .stApp label,
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stRadio"] p, .stApp [data-testid="stSelectbox"] div, .stApp [data-testid="stMultiSelect"] div {color: #1C2B33;}
.campo span:last-child {color: #1C2B33 !important;}
/* Campos de captura y listas desplegables */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="base-input"],
.stApp input, .stApp textarea, [data-testid="stNumberInputContainer"] {
    background-color: #FFFFFF !important; color: #1C2B33 !important; -webkit-text-fill-color: #1C2B33 !important;
    border-color: #D6D0C4 !important;}
[data-baseweb="select"] div, .stSelectbox [role="group"], .stMultiSelect [role="group"],
.stSelectbox [role="group"] button, .stMultiSelect [role="group"] button {background-color: #FFFFFF !important;
    color: #1C2B33 !important; border-color: #D6D0C4 !important;}
.stSelectbox [role="group"] svg, .stMultiSelect [role="group"] svg {color: #56656D !important; fill: #56656D !important;}
[data-baseweb="select"] svg {fill: #56656D !important; color: #56656D !important;}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul,
[data-baseweb="popover"] li, [role="listbox"], [role="option"] {background-color: #FFFFFF !important; color: #1C2B33 !important;}
[role="option"]:hover, [role="option"][aria-selected="true"] {background-color: #EEF4F4 !important;}
[data-baseweb="select"] [data-baseweb="tag"], [data-baseweb="tag"] {background-color: #0E4F5C !important;}
[data-baseweb="tag"] div, [data-baseweb="tag"] span {background-color: transparent !important; color: #FFFFFF !important;}
[data-baseweb="tag"] span {color: #FFFFFF !important;}
/* Sliders y radios */
[data-testid="stSlider"] [role="slider"] {background-color: #0E4F5C !important; border-color: #0E4F5C !important;}
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color: #1C2B33 !important;}
[data-testid="stRadio"] label div:first-child {border-color: #56656D;}
/* Botones */
.stButton button, .stDownloadButton button {background-color: #FFFFFF !important; color: #1C2B33 !important;
    border: 1px solid #C9C2B4 !important;}
.stButton button p, .stDownloadButton button p {color: #1C2B33 !important;}
.stFormSubmitButton button {background-color: #0E4F5C !important; border-color: #0E4F5C !important;}
.stFormSubmitButton button, .stFormSubmitButton button p {color: #FFFFFF !important;}
[data-testid="stForm"] {background-color: #FFFFFF !important;}
/* Elementos con fondo oscuro propio: texto claro */
.cifra.destacada .valor, .marca .logo {color: #FFFFFF !important;}
.cifra.destacada .rotulo, .cifra.destacada .detalle, .cifra.destacada span {color: #CFE2E5 !important;}
.cifra.destacada .valor {color: #FFFFFF !important;}

/* Texto */
div[data-testid="stMarkdownContainer"] .titulo {padding: 0; font-size: 34px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2; color: #1C2B33;
         margin: 16px 0 6px;}
div[data-testid="stMarkdownContainer"] .seccion {padding: 0; font-size: 24px; font-weight: 600; line-height: 1.25; color: #1C2B33; margin: 28px 0 10px;}
div[data-testid="stMarkdownContainer"] .subseccion {padding: 0; font-size: 20px; font-weight: 600; color: #1C2B33; margin: 0 0 6px;}
div[data-testid="stMarkdownContainer"] .texto {color: #36454D; font-size: 15px; line-height: 1.6; max-width: 860px; margin: 0 0 14px;}
div[data-testid="stMarkdownContainer"] .nota {color: #56656D; font-size: 13px; line-height: 1.5; margin: 8px 0 0;}
.etiqueta {font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #0E4F5C;}
.fila-titulo {display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
              gap: 6px 24px; margin: 28px 0 14px;}
div[data-testid="stMarkdownContainer"] .fila-titulo .seccion {margin: 0;}
.fila-titulo > span {font-size: 14px; color: #56656D;}

/* Tarjetas */
.tarjeta {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 32px 36px;}
div[data-testid="stMarkdownContainer"] .tarjeta h1 {font-size: 36px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2; margin: 6px 0 0;
             padding: 0; color: #1C2B33;}
.campos {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px 32px;
         border-top: 1px solid #ECE8E0; padding-top: 22px; margin-top: 24px;}
.campo {display: flex; flex-direction: column; gap: 2px; font-weight: 500;}
.campo span:first-child {font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: #56656D;}
.dos-columnas {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 32px 48px;}
div[data-testid="stMarkdownContainer"] .dos-columnas .seccion {margin-top: 8px;}
.cifras {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px;}
.cifras.tres-c {grid-template-columns: repeat(3, minmax(0, 1fr));}
.cifra {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 14px; padding: 20px 22px;
        display: flex; flex-direction: column; gap: 4px;}
.cifra .rotulo, .cifra .detalle {font-size: 13px; color: #56656D;}
.cifra .rotulo {font-weight: 500; display: flex; align-items: center; gap: 8px;}
.cifra .valor {font-size: 30px; font-weight: 600; line-height: 1.15; color: #1C2B33;}
.cifra.destacada {background: #0E4F5C; border-color: #0E4F5C;}
.cifra.destacada .rotulo, .cifra.destacada .detalle {color: #CFE2E5;}
.cifra.destacada .valor {color: #FFFFFF;}
.lista-modelos article {display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 4px 16px;
                        padding: 20px 0; border-top: 1px solid #E2DDD3;}
.lista-modelos article:last-child {border-bottom: 1px solid #E2DDD3;}
.lista-modelos .num {font-size: 14px; font-weight: 600; color: #0E4F5C; padding-top: 4px;}
div[data-testid="stMarkdownContainer"] .lista-modelos h3 {font-size: 20px; font-weight: 600; margin: 0 0 4px; padding: 0; color: #1C2B33;}
.lista-modelos .sub {font-size: 13px; color: #56656D; margin-bottom: 6px;}
div[data-testid="stMarkdownContainer"] .lista-modelos p {margin: 0; color: #36454D; max-width: 860px; line-height: 1.6;}
.tres {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px;}
div[data-testid="stMarkdownContainer"] .tres .cifra h3, div[data-testid="stMarkdownContainer"] .dos-tarjetas .cifra h3 {font-size: 19px; font-weight: 600; margin: 0; padding: 0; color: #1C2B33;}
div[data-testid="stMarkdownContainer"] .tres .cifra p, div[data-testid="stMarkdownContainer"] .dos-tarjetas .cifra p {margin: 0; font-size: 14px; color: #36454D; line-height: 1.6;}
.dos-tarjetas {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px;}
div[data-testid="stMarkdownContainer"] .lista {margin: 0; padding-left: 22px; color: #36454D; max-width: 900px; line-height: 1.6;}
.lista li {margin-bottom: 8px;}
.ecuacion {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 12px; padding: 16px 22px; font-size: 15px;
           color: #1C2B33; overflow-x: auto; line-height: 1.7;}

/* Tablas */
.marco {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; overflow-x: auto;}
div[data-testid="stMarkdownContainer"] table.tabla {width: 100%; border-collapse: collapse; font-size: 15px;
                                                    margin: 0; border: 0;}
div[data-testid="stMarkdownContainer"] table.tabla th {text-align: left; padding: 14px 20px; font-size: 12px;
    font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #56656D; background: #FAF8F4;
    border: 0; border-bottom: 1px solid #E2DDD3;}
div[data-testid="stMarkdownContainer"] table.tabla td {padding: 12px 20px; border: 0;
                                                       border-bottom: 1px solid #ECE8E0; color: #1C2B33;}
div[data-testid="stMarkdownContainer"] table.tabla tr:last-child td {border-bottom: 0;}
div[data-testid="stMarkdownContainer"] table.tabla .der {text-align: right;}
div[data-testid="stMarkdownContainer"] table.tabla .fuerte {font-weight: 600;}
div[data-testid="stMarkdownContainer"] code {background: #ECE8E0; color: #1C2B33; padding: 1px 6px; border-radius: 4px;}
div[data-testid="stMarkdownContainer"] table.tabla .sec {color: #56656D;}
div[data-testid="stMarkdownContainer"] table.tabla .mejor {font-weight: 700; color: #0E4F5C;}
div[data-testid="stMarkdownContainer"] table.tabla tr.resaltada td {background: #EEF4F4;}
div[data-testid="stMarkdownContainer"] table.compacta td, div[data-testid="stMarkdownContainer"] table.compacta th {
    padding-left: 14px; padding-right: 14px; font-size: 14px;}
div[data-testid="stMarkdownContainer"] table.compacta th {font-size: 12px;}
.muestra {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px;}
.si {color: #2E6B5A; font-weight: 600;} .no {color: #A8461F; font-weight: 600;}

/* Avisos */
.aviso {display: flex; align-items: flex-start; gap: 12px; padding: 14px 18px; border-radius: 12px;
        border: 1px solid; margin: 0 0 16px; line-height: 1.5;}
.aviso svg {flex: none; margin-top: 1px;}
.aviso.ok {background: #E6F1EE; border-color: #B9D8CF; color: #1E4D41;}
.aviso.alerta {background: #FBF1E3; border-color: #EBCB9E; color: #6B4410;}
.aviso.error {background: #FBEAE4; border-color: #E9B7A4; color: #7A2E14;}

/* Barras horizontales */
.figura {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 24px 28px; margin-top: 16px;}
.figura .cabeza {display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
                 gap: 8px 24px; margin-bottom: 18px;}
.figura .cabeza b {font-size: 18px; font-weight: 600;}
.leyenda {display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #56656D;}
.leyenda span {display: inline-flex; align-items: center; gap: 6px;}
.leyenda i {display: inline-block; width: 12px; height: 12px; border-radius: 3px;}
.barras {display: flex; flex-direction: column; gap: 14px;}
.barras .fila {display: grid; grid-template-columns: 170px minmax(0, 1fr) 130px; gap: 14px; align-items: center;
               font-size: 14px;}
.barras .fila > span:first-child {font-weight: 500;}
.barras .fila > span:last-child {text-align: right; font-weight: 600;}
.pista {position: relative; height: 24px; background: #EFEBE3; border-radius: 6px;}
.pista .barra {height: 100%; border-radius: 6px;}
.pista .rango {position: absolute; top: 9px; height: 6px; background: rgba(28,43,51,0.35); border-radius: 3px;}

/* Widgets */
.st-key-filtros, [data-testid="stForm"], .st-key-grafica, .st-key-grafica2, .st-key-grafica3, .st-key-grafica4 {
    background: #FFFFFF; border: 1px solid #E2DDD3 !important; border-radius: 16px; padding: 24px 28px;}
[data-testid="stForm"] {padding: 28px;}
[data-testid="stWidgetLabel"] p {font-size: 13px; font-weight: 500; color: #36454D;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display: none;}
.stButton button, .stFormSubmitButton button {font-weight: 600; min-height: 44px;}
.stFormSubmitButton button {min-height: 50px; font-size: 16px;}
.stFormSubmitButton button:hover {background: #0A3D47; border-color: #0A3D47; color: #FFFFFF;}
[data-testid="InputInstructions"] {display: none;}
@media (max-width: 800px) {
  .cifras, .tres, .campos, .dos-columnas, .dos-tarjetas, .cifras.tres-c {grid-template-columns: minmax(0, 1fr);}
  .barras .fila {grid-template-columns: 110px minmax(0, 1fr) 100px;}
}
</style>
""",
    unsafe_allow_html=True,
)


def html(texto):
    """Muestra HTML. Se quitan la sangría y las líneas vacías para que Markdown no lo trate como código."""
    st.markdown('\n'.join(linea.strip() for linea in texto.splitlines() if linea.strip()), unsafe_allow_html=True)


ICONO_OK = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2E6B5A" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
            '<path d="M8 12.5l2.6 2.6L16 9.5"/></svg>')
ICONO_ALERTA = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#A86B12" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3.5l9 16H3z"/>'
                '<path d="M12 10v4.5"/><path d="M12 17.5h.01"/></svg>')


def aviso(texto, tipo='alerta'):
    icono = ICONO_OK if tipo == 'ok' else ICONO_ALERTA
    html(f'<div class="aviso {tipo}" role="status">{icono}<span>{texto}</span></div>')


def usd(x, dec=0):
    return f'{x:,.{dec}f} USD'


def num(x, dec=2):
    return f'{x:,.{dec}f}'


def p_txt(p):
    """Valor-p legible: muy pequeños en notación científica."""
    if p < 0.0001:
        return '< 0.0001'
    return f'{p:.4f}'


def si_no(cond):
    return '<span class="si">Sí</span>' if cond else '<span class="no">No</span>'


def tabla_html(df, formatos=None, der=None, fuerte=None, resaltar=None, compacta=True):
    """Convierte un DataFrame en tabla HTML con el estilo de la aplicación."""
    formatos = formatos or {}
    der = set(der or [])
    fuerte = set(fuerte or [])
    clase = 'tabla compacta' if compacta else 'tabla'
    cab = ''.join(f'<th class="{"der" if c in der else ""}">{c}</th>' for c in df.columns)
    filas = ''
    for i, fila in df.iterrows():
        celdas = ''
        for c in df.columns:
            v = fila[c]
            texto = formatos[c](v) if c in formatos else v
            clases = ' '.join(k for k, cond in (('der', c in der), ('fuerte', c in fuerte)) if cond)
            celdas += f'<td class="{clases}">{texto}</td>'
        filas += f'<tr class="{"resaltada" if resaltar is not None and resaltar(fila) else ""}">{celdas}</tr>'
    html(f'<div class="marco"><table class="{clase}"><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table></div>')


def base_grafica(chart, alto=300):
    return (chart.properties(height=alto, background=SUPERFICIE)
            .configure(font='Montserrat')
            .configure_axis(labelColor=TEXTO_SEC, titleColor=TEXTO, labelFontSize=12, titleFontSize=13,
                            gridColor='#ECE8E0', domainColor='#C9C2B4', tickColor='#C9C2B4')
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor=TEXTO, titleColor=TEXTO, labelFontSize=12))


# ---------------------------------------------------------------- carga de recursos
@st.cache_resource
def obtener_modelos():
    return {nombre: joblib.load(CARPETA_MODELOS / f'{archivo}.joblib')
            for archivo, nombre in [('gradient_boosting', 'Gradient Boosting optimizado'),
                                    ('random_forest', 'Random Forest optimizado'),
                                    ('regresion_lineal', 'Regresión lineal')]}


@st.cache_data
def obtener_resultados():
    with open(CARPETA_MODELOS / 'resultados.json', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def obtener_datos():
    datos = pd.read_csv(CARPETA / 'datos' / 'datos_casas_limpio.csv', dtype={'codigo_postal': str},
                        parse_dates=['fecha_venta'])
    return datos


@st.cache_data
def obtener_predicciones():
    return pd.read_csv(CARPETA_MODELOS / 'predicciones_test.csv')


resultados = obtener_resultados()
casas = obtener_datos()
predicciones = obtener_predicciones()
VARIABLES = resultados['variables_numericas'] + ['codigo_postal']

NOMBRES = {
    'precio': 'Precio (USD)', 'superficie_m2': 'Superficie construida (m²)', 'terreno_m2': 'Terreno (m²)',
    'habitaciones': 'Habitaciones', 'banos': 'Baños', 'pisos': 'Pisos', 'frente_agua': 'Frente al agua',
    'vista': 'Vista (0 a 4)', 'condicion': 'Condición (1 a 5)', 'calidad': 'Calidad de construcción (1 a 13)',
    'antiguedad': 'Antigüedad (años)', 'renovada': 'Renovada', 'latitud': 'Latitud', 'longitud': 'Longitud',
    'superficie_vecinos_m2': 'Superficie de los vecinos (m²)', 'meses_desde_inicio': 'Meses desde mayo 2014',
    'codigo_postal': 'Código postal', 'precio_m2': 'Precio por m² (USD)', 'const': 'Intercepto',
}


# ---------------------------------------------------------------- encabezado y navegación
SECCIONES = ['Proyecto', 'Datos', 'Exploración', 'Regresión', 'Modelos', 'Predicción', 'Conclusiones']
with st.container(key='encabezado'):
    c_marca, c_nav = st.columns([1, 2.8], vertical_alignment='center')
    with c_marca:
        html('<div class="marca"><span class="logo" aria-hidden="true">$</span>Precios de vivienda</div>')
    with c_nav:
        seccion = st.radio('Secciones', SECCIONES, horizontal=True, label_visibility='collapsed', key='seccion')

cv = pd.DataFrame(resultados['cv']).query('Modelo in @MODELOS_APP').reset_index(drop=True)
test = pd.DataFrame(resultados['test']).query('Modelo in @MODELOS_APP').reset_index(drop=True)
MEJOR = resultados['mejor_modelo']
fila_mejor = test[test['Modelo'] == MEJOR].iloc[0]


# ================================================================ 1. PROYECTO
if seccion == 'Proyecto':
    html("""
    <section class="tarjeta">
      <span class="etiqueta">Proyecto Final · Estadística Aplicada a la Investigación · Caso 2</span>
      <h1>Predicción del valor de propiedades inmobiliarias</h1>
      <div class="campos">
        <div class="campo"><span>Alumno</span><span>Victor Francisco Duarte Villalobos</span></div>
        <div class="campo"><span>Matrícula</span><span>403230</span></div>
        <div class="campo"><span>Materia</span><span>Estadística Aplicada a la Investigación</span></div>
        <div class="campo"><span>Institución</span><span>Universidad Autónoma de Chihuahua</span></div>
        <div class="campo"><span>Programa</span><span>Maestría en Ingeniería en Computación</span></div>
        <div class="campo"><span>Datos</span><span>King County, Washington (2014–2015)</span></div>
      </div>
    </section>
    """)
    html("""
    <div class="dos-columnas" style="margin-top: 40px">
      <div>
        <h2 class="seccion">Problema</h2>
        <p class="texto">El precio de una vivienda depende de su tamaño y calidad, pero también de dónde está y de cuándo
        se vende. El problema es de regresión supervisada: estimar el precio de venta en dólares a partir de las
        características de la propiedad, su ubicación, su entorno y la fecha de venta.</p>
      </div>
      <div>
        <h2 class="seccion">Objetivos</h2>
        <ol class="lista">
          <li>Identificar los factores más influyentes en el precio con correlación y ANOVA.</li>
          <li>Ajustar un modelo de regresión lineal múltiple para entender las relaciones básicas.</li>
          <li>Comparar modelos de Machine Learning y seleccionar el más preciso.</li>
        </ol>
      </div>
    </div>
    """)
    html(f"""
    <h2 class="seccion" style="margin-top: 40px">Base de datos</h2>
    <p class="texto"><i>House Sales in King County, USA</i> (Kaggle, dominio público): ventas de viviendas en el condado
    de King, que incluye Seattle, entre mayo de 2014 y mayo de 2015.</p>
    <div class="cifras">
      <div class="cifra"><span class="rotulo">Ventas analizadas</span><span class="valor">{resultados['n_limpio']:,}</span>
        <span class="detalle">de {resultados['n_original']:,} registros originales</span></div>
      <div class="cifra"><span class="rotulo">Precio mediano</span><span class="valor">{casas['precio'].median() / 1000:,.0f} mil</span>
        <span class="detalle">USD</span></div>
      <div class="cifra"><span class="rotulo">Superficie mediana</span><span class="valor">{casas['superficie_m2'].median():,.0f} m²</span>
        <span class="detalle">construidos</span></div>
      <div class="cifra"><span class="rotulo">Precio mediano por m²</span><span class="valor">{casas['precio_m2'].median():,.0f}</span>
        <span class="detalle">USD por m²</span></div>
    </div>
    <div class="cifras" style="margin-top: 16px">
      <div class="cifra"><span class="rotulo">Variables predictoras</span><span class="valor">16</span>
        <span class="detalle">15 numéricas y el código postal</span></div>
      <div class="cifra"><span class="rotulo">Train / Test</span><span class="valor">{resultados['n_train']:,} / {resultados['n_test']:,}</span>
        <span class="detalle">80 % / 20 %, agrupado por vivienda</span></div>
      <div class="cifra"><span class="rotulo">Modelos comparados</span><span class="valor">{len(test)}</span>
        <span class="detalle">con validación cruzada de 5 particiones</span></div>
      <div class="cifra destacada"><span class="rotulo">R² del mejor modelo en Test</span>
        <span class="valor">{fila_mejor['R2']:.3f}</span><span class="detalle">{MEJOR}</span></div>
    </div>
    """)
    html(f"""
    <h2 class="seccion" style="margin-top: 48px">Metodología</h2>
    <div class="lista-modelos">
      <article><span class="num">01</span><div><h3>Preparación de datos</h3>
        <div class="sub">Limpieza · conversión a m² · variables de fecha y antigüedad</div>
        <p>Se eliminaron registros imposibles (0 habitaciones o 0 baños y una casa con 33 habitaciones en 150 m²) y se
        conservaron las ventas repetidas de una misma casa como precios históricos.</p></div></article>
      <article><span class="num">02</span><div><h3>Análisis estadístico</h3>
        <div class="sub">Descriptivos · Pearson y Spearman · ANOVA de un factor con Tukey</div>
        <p>Se identificaron las variables más relacionadas con el precio y se probó si el precio medio cambia entre
        niveles de calidad, vista, condición y frente al agua.</p></div></article>
      <article><span class="num">03</span><div><h3>Regresión lineal múltiple</h3>
        <div class="sub">Mínimos cuadrados (statsmodels) · ecuación normal · Gradient Descent</div>
        <p>Modelo base interpretable: coeficientes con valores-p, R², R² ajustada y MSE. Los tres métodos dan los
        mismos coeficientes.</p></div></article>
      <article><span class="num">04</span><div><h3>Machine Learning</h3>
        <div class="sub">Regresión lineal · Random Forest · Gradient Boosting</div>
        <p>La regresión lineal es la referencia; Random Forest y Gradient Boosting son modelos de árboles. Hiperparámetros optimizados con validación cruzada agrupada por vivienda (GroupKFold) y evaluación final,
        una sola vez, en el conjunto de prueba.</p></div></article>
    </div>
    """)


# ================================================================ 2. DATOS
elif seccion == 'Datos':
    html('<h1 class="titulo">Datos</h1>'
         '<p class="texto">Origen, limpieza, diccionario de variables y estadísticas descriptivas.</p>')
    eliminadas = resultados['n_original'] - resultados['n_limpio']
    html(f"""
    <div class="cifras">
      <div class="cifra"><span class="rotulo">Registros originales</span><span class="valor">{resultados['n_original']:,}</span>
        <span class="detalle">21 columnas, sin valores nulos</span></div>
      <div class="cifra"><span class="rotulo">Registros eliminados</span><span class="valor">{eliminadas}</span>
        <span class="detalle">0 habitaciones, 0 baños o 33 habitaciones</span></div>
      <div class="cifra"><span class="rotulo">Ventas repetidas</span><span class="valor">{resultados['ids_repetidos']}</span>
        <span class="detalle">misma casa vendida dos veces: se conservan</span></div>
      <div class="cifra"><span class="rotulo">Códigos postales</span><span class="valor">{casas['codigo_postal'].nunique()}</span>
        <span class="detalle">usados como ubicación del vecindario</span></div>
    </div>
    """)

    html('<h2 class="seccion" style="margin-top: 36px">Diccionario de variables</h2>'
         '<p class="texto">Cómo se cubre cada requisito del Caso 2 con las columnas del dataset.</p>')
    diccionario = pd.DataFrame([
        ('Superficie', 'superficie_m2, terreno_m2', 'Construcción y terreno, convertidos de ft² a m² (× 0.092903)'),
        ('Habitaciones', 'habitaciones, baños, pisos', 'Distribución de la vivienda'),
        ('Ubicación geográfica', 'latitud, longitud, codigo_postal', '70 códigos postales del condado'),
        ('Entorno', 'frente_agua, vista, superficie_vecinos_m2', 'Frente al lago, calidad de la vista y tamaño de las 15 casas vecinas'),
        ('Propiedad', 'calidad, condicion, antiguedad, renovada', 'Calidad de construcción (1 a 13), estado (1 a 5), años al venderse'),
        ('Precios históricos', 'fecha_venta, meses_desde_inicio', 'Mes de la venta y ventas repetidas de una misma casa'),
        ('Variable objetivo', 'precio', 'Precio de venta en dólares'),
    ], columns=['Requisito del caso', 'Columnas', 'Descripción'])
    tabla_html(diccionario, fuerte=['Requisito del caso'])
    html('<p class="nota">El dataset no incluye distancias a escuelas, hospitales o transporte ni índices económicos '
         '(PIB per cápita, criminalidad). La ubicación (latitud, longitud y código postal) se usa como sustituto del '
         'efecto del vecindario.</p>')

    html('<h2 class="seccion" style="margin-top: 36px">Medidas de tendencia central y dispersión</h2>')
    desc = pd.DataFrame(resultados['descriptivos'])
    desc['Variable'] = desc['Variable'].map(lambda v: NOMBRES.get(v, v))
    cols = ['Variable', 'Media', 'Mediana', 'Moda', 'Desviación estándar', 'Varianza', 'Mínimo', 'Máximo', 'Rango']
    fmt = {c: (lambda v: f'{v:,.2f}') for c in cols[1:]}
    fmt['Varianza'] = lambda v: f'{v:,.4g}' if abs(v) >= 1e6 else f'{v:,.2f}'
    tabla_html(desc[cols], formatos=fmt, der=cols[1:], fuerte=['Variable'])
    html('<p class="nota">El precio tiene media mayor que la mediana (sesgo a la derecha): pocas casas muy caras elevan '
         'el promedio. Por eso en la aplicación se reporta la mediana como valor típico.</p>')

    html('<h2 class="seccion" style="margin-top: 36px">Explorar los datos</h2>')
    with st.container(key='filtros'):
        f1, f2, f3 = st.columns(3)
        cps = f1.multiselect('Códigos postales (vacío = todos)', sorted(casas['codigo_postal'].unique()))
        rango_precio = f2.slider('Precio (miles de USD)', 0, 8000, (0, 8000), step=50)
        rango_sup = f3.slider('Superficie (m²)', 0, 1300, (0, 1300), step=10)
    filtro = casas[(casas['precio'].between(rango_precio[0] * 1000, rango_precio[1] * 1000)) &
                   (casas['superficie_m2'].between(*rango_sup))]
    if cps:
        filtro = filtro[filtro['codigo_postal'].isin(cps)]
    if filtro.empty:
        aviso('No hay ventas con esos filtros.', 'alerta')
    else:
        html(f"""
        <div class="cifras" style="margin-top: 16px">
          <div class="cifra"><span class="rotulo">Ventas</span><span class="valor">{len(filtro):,}</span></div>
          <div class="cifra"><span class="rotulo">Precio mediano</span><span class="valor">{filtro['precio'].median() / 1000:,.0f} mil</span></div>
          <div class="cifra"><span class="rotulo">Superficie mediana</span><span class="valor">{filtro['superficie_m2'].median():,.0f} m²</span></div>
          <div class="cifra"><span class="rotulo">USD por m² (mediana)</span><span class="valor">{filtro['precio_m2'].median():,.0f}</span></div>
        </div>
        """)
        with st.container(key='grafica'):
            html('<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
                 '<b>Ubicación de las ventas</b><span class="leyenda">Color: precio (escala logarítmica)</span></div></div>')
            muestra_mapa = filtro.sample(min(len(filtro), 6000), random_state=1)
            mapa = alt.Chart(muestra_mapa).mark_circle(size=12, opacity=0.65).encode(
                x=alt.X('longitud:Q', scale=alt.Scale(zero=False), title='Longitud'),
                y=alt.Y('latitud:Q', scale=alt.Scale(zero=False), title='Latitud'),
                color=alt.Color('precio:Q', scale=alt.Scale(type='log', scheme='viridis'), title='Precio (USD)',
                                legend=alt.Legend(format='~s')),
                tooltip=[alt.Tooltip('precio:Q', format=',.0f', title='Precio'), 'codigo_postal:N',
                         alt.Tooltip('superficie_m2:Q', format=',.0f', title='m²'), 'calidad:Q'],
            )
            st.altair_chart(base_grafica(mapa, 460), width='stretch', theme=None)
            html('<p class="nota" style="margin:0">Las casas más caras se concentran al norte y al este del lago '
                 'Washington (Bellevue, Medina, Mercer Island). Se muestran hasta 6,000 ventas.</p>')

        with st.container(key='grafica2'):
            html('<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
                 '<b>Precios históricos: mediana del precio por m² por mes</b></div></div>')
            por_mes = (filtro.assign(mes=filtro['fecha_venta'].dt.to_period('M').dt.to_timestamp())
                       .groupby('mes').agg(precio_m2=('precio_m2', 'median'), ventas=('precio', 'size')).reset_index())
            linea = alt.Chart(por_mes).mark_line(point=True, color=PRIMARIO).encode(
                x=alt.X('mes:T', title=None, axis=alt.Axis(format='%b %Y')),
                y=alt.Y('precio_m2:Q', scale=alt.Scale(zero=False), title='USD por m²'),
                tooltip=[alt.Tooltip('mes:T', format='%B %Y'), alt.Tooltip('precio_m2:Q', format=',.0f'), 'ventas:Q'])
            st.altair_chart(base_grafica(linea, 260), width='stretch', theme=None)

        html('<h2 class="seccion" style="margin-top: 36px">Muestra de registros</h2>')
        muestra = filtro.drop(columns=['precio_m2', 'id']).head(30).copy()
        muestra['fecha_venta'] = muestra['fecha_venta'].dt.strftime('%Y-%m-%d')
        muestra = muestra.rename(columns={'fecha_venta': 'Fecha', 'precio': 'Precio', 'habitaciones': 'Hab.',
                                          'banos': 'Baños', 'superficie_m2': 'm²', 'terreno_m2': 'Terreno m²',
                                          'pisos': 'Pisos', 'frente_agua': 'Agua', 'vista': 'Vista',
                                          'condicion': 'Cond.', 'calidad': 'Calidad', 'antiguedad': 'Antig.',
                                          'renovada': 'Renov.', 'latitud': 'Lat.', 'longitud': 'Long.',
                                          'codigo_postal': 'C.P.', 'superficie_vecinos_m2': 'Vecinos m²',
                                          'meses_desde_inicio': 'Mes'})
        num_cols = [c for c in muestra.columns if c not in ('Fecha', 'C.P.')]
        tabla_html(muestra, formatos={'Precio': lambda x: f'{x:,.0f}', 'm²': lambda x: f'{x:,.0f}',
                                      'Terreno m²': lambda x: f'{x:,.0f}', 'Vecinos m²': lambda x: f'{x:,.0f}',
                                      'Baños': lambda x: f'{x:g}', 'Pisos': lambda x: f'{x:g}',
                                      'Lat.': lambda x: f'{x:.4f}', 'Long.': lambda x: f'{x:.4f}'},
                   der=num_cols)
        html(f'<p class="nota">Primeras 30 de {len(filtro):,} ventas con los filtros elegidos. '
             'El botón descarga todas.</p>')
        st.download_button('Descargar datos filtrados (CSV)', filtro.to_csv(index=False).encode('utf-8'),
                           'datos_casas_filtrados.csv', 'text/csv')


# ================================================================ 3. EXPLORACIÓN
elif seccion == 'Exploración':
    html('<h1 class="titulo">Análisis exploratorio</h1>'
         '<p class="texto">Distribuciones, correlación de Pearson y ANOVA de un factor (α = 0.05).</p>')

    numericas = ['precio', 'superficie_m2', 'terreno_m2', 'habitaciones', 'banos', 'calidad', 'antiguedad',
                 'superficie_vecinos_m2', 'precio_m2']
    with st.container(key='grafica'):
        v = st.selectbox('Variable', numericas, format_func=lambda c: NOMBRES[c])
        c1, c2 = st.columns(2)
        datos_v = casas[list(dict.fromkeys([v, 'precio']))].copy()
        recorte = datos_v[v].quantile(0.995)
        hist = alt.Chart(datos_v[datos_v[v] <= recorte]).mark_bar(color=PRIMARIO, opacity=0.85).encode(
            x=alt.X(f'{v}:Q', bin=alt.Bin(maxbins=50), title=NOMBRES[v], axis=alt.Axis(format='~s', labelOverlap=True)),
            y=alt.Y('count():Q', title='Ventas'))
        c1.altair_chart(base_grafica(hist, 280), width='stretch', theme=None)
        if v != 'precio':
            disp = alt.Chart(datos_v.sample(5000, random_state=2)).mark_circle(size=10, opacity=0.35, color=PRIMARIO).encode(
                x=alt.X(f'{v}:Q', title=NOMBRES[v], scale=alt.Scale(zero=False)),
                y=alt.Y('precio:Q', title='Precio (USD)', axis=alt.Axis(format='~s')))
            c2.altair_chart(base_grafica(disp, 280), width='stretch', theme=None)
        else:
            loghist = alt.Chart(pd.DataFrame({'log10_precio': np.log10(casas['precio'])})).mark_bar(
                color=ACENTO, opacity=0.85).encode(
                x=alt.X('log10_precio:Q', bin=alt.Bin(maxbins=50), title='log10(Precio)'), y=alt.Y('count():Q', title='Ventas'))
            c2.altair_chart(base_grafica(loghist, 280), width='stretch', theme=None)
        html(f'<p class="nota" style="margin:0">Histograma hasta el percentil 99.5 · media {casas[v].mean():,.2f} · '
             f'mediana {casas[v].median():,.2f} · desviación estándar {casas[v].std():,.2f} · '
             f'asimetría {casas[v].skew():.2f}. El diagrama de dispersión usa 5,000 ventas al azar.</p>')

    html('<h2 class="seccion" style="margin-top: 36px">Correlación con el precio</h2>'
         '<p class="texto">r de Pearson mide la relación lineal; ρ de Spearman, la relación monótona usando rangos (menos '
         'sensible a las casas de lujo). La prueba es H₀: ρ = 0 con t = r√(n−2)/√(1−r²).</p>')
    corr = pd.DataFrame(resultados['correlaciones'])
    corr['Variable'] = corr['Variable'].map(lambda c: NOMBRES.get(c, c))
    tabla_html(corr[['Variable', 'r de Pearson', 'r²', 'valor-p', 'rho de Spearman']].rename(
        columns={'rho de Spearman': 'ρ de Spearman'}),
        formatos={'r de Pearson': lambda x: f'{x:.4f}', 'r²': lambda x: f'{x:.4f}', 'valor-p': p_txt,
                  'ρ de Spearman': lambda x: f'{x:.4f}'},
        der=['r de Pearson', 'r²', 'valor-p', 'ρ de Spearman'], fuerte=['Variable'])
    html('<p class="nota">Con más de 21 mil ventas casi cualquier correlación es significativa; importa más la magnitud '
         'de r. La superficie construida y la calidad de construcción son las variables con la relación más fuerte.</p>')

    with st.container(key='grafica2'):
        html('<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
             '<b>Matriz de correlación de Pearson</b></div></div>')
        matriz = pd.DataFrame(resultados['pearson_matriz'])
        orden = list(matriz.columns)
        largo = matriz.reset_index(names='fila').melt('fila', var_name='columna', value_name='r')
        calor = alt.Chart(largo).mark_rect().encode(
            x=alt.X('columna:N', sort=orden, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('fila:N', sort=orden, title=None),
            color=alt.Color('r:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1], reverse=True), title='r'),
            tooltip=['fila', 'columna', alt.Tooltip('r:Q', format='.3f')])
        texto = alt.Chart(largo).mark_text(fontSize=9).encode(
            x=alt.X('columna:N', sort=orden), y=alt.Y('fila:N', sort=orden), text=alt.Text('r:Q', format='.2f'),
            color=alt.condition('abs(datum.r) > 0.5', alt.value('white'), alt.value(TEXTO)))
        st.altair_chart(base_grafica(calor + texto, 560), width='stretch', theme=None)

    html('<h2 class="seccion" style="margin-top: 36px">ANOVA de un factor</h2>'
         '<p class="texto">H₀: el precio medio es igual en todos los grupos del factor. F = CM entre / CM dentro; '
         'η² = SC entre / SC total es la proporción de la variación del precio que explica el factor. Levene revisa la '
         'igualdad de varianzas y Kruskal-Wallis confirma el resultado sin suponer normalidad.</p>')
    anova = pd.DataFrame(resultados['anova'])
    anova['Decisión'] = np.where(anova['valor-p'] < 0.05, 'Se rechaza H₀', 'No se rechaza H₀')
    tabla_html(anova[['Factor', 'Grupos', 'gl entre', 'gl dentro', 'F', 'valor-p', 'eta²', 'Levene p',
                      'Kruskal-Wallis p', 'Decisión']].rename(columns={'eta²': 'η²'}),
               formatos={'F': lambda x: f'{x:,.2f}', 'valor-p': p_txt, 'η²': lambda x: f'{x:.4f}',
                         'Levene p': p_txt, 'Kruskal-Wallis p': p_txt},
               der=['Grupos', 'gl entre', 'gl dentro', 'F', 'valor-p', 'η²', 'Levene p', 'Kruskal-Wallis p'],
               fuerte=['Factor'])

    factores = {'Calidad de construcción': 'calidad', 'Vista': 'vista', 'Condición': 'condicion',
                'Frente al agua': 'frente_agua'}
    with st.container(key='grafica3'):
        factor = st.selectbox('Factor', list(factores))
        col = factores[factor]
        datos_f = casas[[col, 'precio']].copy()
        if col == 'calidad':
            datos_f['grupo'] = pd.cut(datos_f[col], [0, 6, 7, 8, 9, 13],
                                      labels=['Baja (≤6)', 'Media (7)', 'Media-alta (8)', 'Alta (9)', 'Lujo (≥10)'])
        elif col == 'frente_agua':
            datos_f['grupo'] = datos_f[col].map({0: 'No', 1: 'Sí'})
        else:
            datos_f['grupo'] = datos_f[col].astype(str)
        resumen = (datos_f.groupby('grupo', observed=True)['precio']
                   .agg(n='size', media='mean', mediana='median', desv='std').reset_index())
        caja = alt.Chart(datos_f).mark_boxplot(extent=1.5, outliers=False, color=PRIMARIO, size=40).encode(
            x=alt.X('grupo:N', title=factor, sort=list(resumen['grupo'].astype(str))),
            y=alt.Y('precio:Q', title='Precio (USD)', axis=alt.Axis(format='~s')))
        g1, g2 = st.columns([1.4, 1])
        g1.altair_chart(base_grafica(caja, 320), width='stretch', theme=None)
        with g2:
            resumen.columns = ['Grupo', 'n', 'Media', 'Mediana', 'Desv. est.']
            tabla_html(resumen, formatos={'n': lambda x: f'{x:,}', 'Media': lambda x: f'{x:,.0f}',
                                          'Mediana': lambda x: f'{x:,.0f}', 'Desv. est.': lambda x: f'{x:,.0f}'},
                       der=['n', 'Media', 'Mediana', 'Desv. est.'], fuerte=['Grupo'])
        html('<p class="nota" style="margin:0">Cajas sin valores atípicos; precios en USD.</p>')


# ================================================================ 4. REGRESIÓN
elif seccion == 'Regresión':
    ols = resultados['ols']
    html('<h1 class="titulo">Regresión lineal múltiple</h1>'
         '<p class="texto">Modelo estadístico base, ajustado por mínimos cuadrados con las 15 variables numéricas '
         'en el conjunto de entrenamiento (statsmodels). Se comprobó que la ecuación normal y Gradient Descent dan los '
         'mismos coeficientes.</p>')
    html('<div class="ecuacion">Precio = β₀ + β₁·Habitaciones + β₂·Baños + β₃·Superficie + … + β₁₅·Meses + ε'
         '&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;Ecuación normal: θ = (XᵀX)⁻¹Xᵀy &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;'
         'Gradient Descent: θ := θ − η∇J(θ)</div>')
    met = pd.DataFrame(ols['metricas'])
    tr, te = met.iloc[0], met.iloc[1]
    html(f"""
    <div class="cifras" style="margin-top: 20px">
      <div class="cifra"><span class="rotulo">R² (entrenamiento)</span><span class="valor">{ols['r2']:.4f}</span>
        <span class="detalle">R² ajustada {ols['r2_aj']:.4f}</span></div>
      <div class="cifra"><span class="rotulo">Prueba F global</span><span class="valor">{ols['f']:,.0f}</span>
        <span class="detalle">valor-p {p_txt(ols['f_p'])}</span></div>
      <div class="cifra"><span class="rotulo">R² en Test</span><span class="valor">{te['R²']:.4f}</span>
        <span class="detalle">casas no vistas al ajustar</span></div>
      <div class="cifra"><span class="rotulo">RMSE en Test</span><span class="valor">{te['RMSE'] / 1000:,.0f} mil</span>
        <span class="detalle">MSE {te['MSE']:.3e} USD²</span></div>
    </div>
    """)
    html('<h2 class="seccion" style="margin-top: 36px">Coeficientes</h2>'
         '<p class="texto">Cada coeficiente es el cambio en el precio (USD) por una unidad más de la variable, '
         'manteniendo constantes las demás. H₀: βⱼ = 0 (prueba t). El coeficiente estandarizado permite comparar '
         'variables con unidades distintas.</p>')
    coef = pd.DataFrame(ols['coeficientes'])
    coef['Variable'] = coef['Variable'].map(lambda c: NOMBRES.get(c, c))
    coef['Sig.'] = coef['valor-p'].map(lambda p: si_no(p < 0.05))
    tabla_html(coef[['Variable', 'Coeficiente', 'Error estándar', 't', 'valor-p', 'Coef. estandarizado', 'Sig.']],
               formatos={'Coeficiente': lambda x: f'{x:,.2f}', 'Error estándar': lambda x: f'{x:,.2f}',
                         't': lambda x: f'{x:.2f}', 'valor-p': p_txt,
                         'Coef. estandarizado': lambda x: '' if pd.isna(x) else f'{x:.4f}'},
               der=['Coeficiente', 'Error estándar', 't', 'valor-p', 'Coef. estandarizado', 'Sig.'],
               fuerte=['Variable'])

    with st.container(key='grafica'):
        html('<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
             '<b>Coeficientes estandarizados</b><span class="leyenda">Cambio en desviaciones estándar del precio por '
             'una desviación estándar de la variable</span></div></div>')
        # Altair interpreta el punto de 'Coef. estandarizado' como campo anidado; se renombra la columna
        c_std = (coef.dropna(subset=['Coef. estandarizado'])
                 .rename(columns={'Coef. estandarizado': 'coef_std'})[['Variable', 'coef_std']])
        barras = alt.Chart(c_std).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y('Variable:N', sort=alt.EncodingSortField('coef_std', op='sum', order='descending'), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X('coef_std:Q', title=None),
            color=alt.condition('datum.coef_std > 0', alt.value(PRIMARIO), alt.value(ACENTO)),
            tooltip=['Variable', alt.Tooltip('coef_std:Q', title='Coef. estandarizado', format='.4f')])
        barras = barras.properties(padding={'left': 24, 'right': 8, 'top': 5, 'bottom': 5})  # evita recortar etiquetas largas
        st.altair_chart(base_grafica(barras, 420), width='stretch', theme=None)

    html('<h2 class="seccion" style="margin-top: 36px">Métricas y supuestos</h2>')
    tabla_html(met, formatos={'R²': lambda x: f'{x:.4f}', 'MSE': lambda x: f'{x:.4e}',
                              'RMSE': lambda x: f'{x:,.2f}', 'MAE': lambda x: f'{x:,.2f}'},
               der=['R²', 'MSE', 'RMSE', 'MAE'], fuerte=['Conjunto'])
    html(f"""
    <ul class="lista" style="margin-top: 16px">
      <li>El R² de Test es parecido al de entrenamiento, así que el modelo lineal no está sobreajustado.</li>
      <li>Breusch-Pagan: valor-p {p_txt(ols['breusch_pagan_p'])}. La varianza de los residuos no es constante: el modelo
      se equivoca más en las casas caras (heterocedasticidad). Los coeficientes siguen siendo insesgados, pero los
      errores estándar deben leerse con cautela.</li>
      <li>La relación entre ubicación y precio no es lineal (latitud y longitud por sí solas no capturan los vecindarios),
      lo que motiva usar el código postal y modelos no lineales en la sección de Modelos.</li>
    </ul>
    """)


# ================================================================ 5. MODELOS
elif seccion == 'Modelos':
    html('<h1 class="titulo">Comparación de modelos</h1>'
         '<p class="texto">El modelo se eligió con validación cruzada (GroupKFold, 5 particiones, agrupando por vivienda) '
         'en el entrenamiento. El conjunto de prueba se usó una sola vez, al final.</p>')
    html(f"""
    <div class="cifras">
      <div class="cifra destacada"><span class="rotulo">Mejor modelo</span><span class="valor" style="font-size:22px">{MEJOR}</span>
        <span class="detalle">menor RMSE en validación cruzada</span></div>
      <div class="cifra"><span class="rotulo">R² en Test</span><span class="valor">{fila_mejor['R2']:.4f}</span>
        <span class="detalle">proporción de la variación explicada</span></div>
      <div class="cifra"><span class="rotulo">RMSE en Test</span><span class="valor">{fila_mejor['RMSE'] / 1000:,.1f} mil</span>
        <span class="detalle">USD</span></div>
      <div class="cifra"><span class="rotulo">MAE en Test</span><span class="valor">{fila_mejor['MAE'] / 1000:,.1f} mil</span>
        <span class="detalle">USD de error promedio</span></div>
    </div>
    """)

    def tabla_metricas(df, titulo, detalle):
        df = df.copy()
        mejores = {'MAE': df['MAE'].min(), 'RMSE': df['RMSE'].min(), 'R2': df['R2'].max()}
        filas = ''
        for _, f in df.iterrows():
            celdas = ''
            for m, fmt in [('MAE', '{:,.0f}'), ('MSE', '{:.4e}'), ('RMSE', '{:,.0f}'), ('R2', '{:.4f}')]:
                clase = 'der mejor' if m in mejores and f[m] == mejores[m] else 'der'
                celdas += f'<td class="{clase}">{fmt.format(f[m])}</td>'
            extra = f'<td class="der sec">{f["RMSE desv."]:,.0f}</td>' if 'RMSE desv.' in df.columns else ''
            filas += (f'<tr class="{"resaltada" if f["Modelo"] == MEJOR else ""}"><td class="fuerte">{f["Modelo"]}</td>'
                      f'{celdas}{extra}</tr>')
        cab_extra = '<th class="der">Desv. RMSE</th>' if 'RMSE desv.' in df.columns else ''
        html(f'<div class="fila-titulo"><h2 class="seccion">{titulo}</h2><span>{detalle}</span></div>'
             '<div class="marco"><table class="tabla compacta"><thead><tr><th>Modelo</th><th class="der">MAE (USD)</th>'
             '<th class="der">MSE (USD²)</th><th class="der">RMSE (USD)</th><th class="der">R²</th>'
             f'{cab_extra}</tr></thead><tbody>{filas}</tbody></table></div>')

    tabla_metricas(test, 'Resultados en Test', f'{resultados["n_test"]:,} ventas (20 %), ordenadas por RMSE')
    html('<p class="nota">En negritas, el mejor valor de cada métrica; sombreado, el modelo seleccionado.</p>')

    with st.container(key='grafica'):
        metrica = st.radio('Métrica', ['R2', 'RMSE', 'MAE'], horizontal=True,
                           format_func=lambda m: {'R2': 'R²', 'RMSE': 'RMSE', 'MAE': 'MAE'}[m])
        barras = alt.Chart(test).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y('Modelo:N', sort=alt.EncodingSortField(metrica, order='descending' if metrica == 'R2' else 'ascending'),
                    title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X(f'{metrica}:Q', title=f'{metrica} en Test' if metrica != 'R2' else 'R² en Test',
                    axis=alt.Axis(format='~s' if metrica != 'R2' else '.2f')),
            color=alt.condition(f'datum.Modelo == "{MEJOR}"', alt.value(PRIMARIO), alt.value('#A9C4CA')),
            tooltip=['Modelo', alt.Tooltip(f'{metrica}:Q', format=',.4f')])
        st.altair_chart(base_grafica(barras, 330), width='stretch', theme=None)

    tabla_metricas(cv, 'Validación cruzada en entrenamiento', 'Promedio de 5 particiones; criterio de selección')
    hp = resultados['hiperparametros']
    filas_hp = pd.DataFrame([{'Modelo': 'Regresión lineal', 'Hiperparámetros': 'No tiene; se ajusta por mínimos cuadrados'}]
                            + [{'Modelo': m, 'Hiperparámetros': ', '.join(f'{k.replace("model__", "")} = {v}'
                                                                           for k, v in p.items())}
                               for m, p in hp.items() if m in MODELOS_APP])
    html('<h2 class="seccion" style="margin-top: 36px">Hiperparámetros seleccionados</h2>')
    tabla_html(filas_hp, fuerte=['Modelo'])

    cvt = pd.DataFrame(resultados['cv_vs_test'])
    html(f'<h2 class="seccion" style="margin-top: 36px">{MEJOR}: validación cruzada frente a Test</h2>')
    tabla_html(cvt, formatos={'CV': lambda x: f'{x:,.4f}' if abs(x) < 10 else f'{x:,.2f}',
                              'Test': lambda x: f'{x:,.4f}' if abs(x) < 10 else f'{x:,.2f}'},
               der=['CV', 'Test'], fuerte=['Métrica'])

    with st.container(key='grafica2'):
        html('<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
             '<b>Precio real frente a precio predicho (Test)</b></div></div>')
        modelo_vis = st.selectbox('Modelo', list(COLOR_MODELO))
        datos_rp = predicciones[['real', modelo_vis]].rename(columns={modelo_vis: 'predicho'})
        datos_rp['residuo'] = datos_rp['real'] - datos_rp['predicho']
        maximo = float(max(datos_rp['real'].max(), datos_rp['predicho'].max()))
        puntos = alt.Chart(datos_rp).mark_circle(size=10, opacity=0.35, color=COLOR_MODELO[modelo_vis]).encode(
            x=alt.X('real:Q', title='Precio real (USD)', axis=alt.Axis(format='~s')),
            y=alt.Y('predicho:Q', title='Precio predicho (USD)', axis=alt.Axis(format='~s')),
            tooltip=[alt.Tooltip('real:Q', format=',.0f'), alt.Tooltip('predicho:Q', format=',.0f')])
        diagonal = alt.Chart(pd.DataFrame({'x': [0, maximo], 'y': [0, maximo]})).mark_line(
            color=ACENTO, strokeDash=[6, 4]).encode(x='x:Q', y='y:Q')
        resid = alt.Chart(datos_rp).mark_circle(size=10, opacity=0.35, color=COLOR_MODELO[modelo_vis]).encode(
            x=alt.X('predicho:Q', title='Precio predicho (USD)', axis=alt.Axis(format='~s')),
            y=alt.Y('residuo:Q', title='Residuo (USD)', axis=alt.Axis(format='~s')))
        cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=ACENTO, strokeDash=[6, 4]).encode(y='y:Q')
        r1, r2 = st.columns(2)
        r1.altair_chart(base_grafica(puntos + diagonal, 340), width='stretch', theme=None)
        r2.altair_chart(base_grafica(resid + cero, 340), width='stretch', theme=None)
        html('<p class="nota" style="margin:0">La línea punteada es la predicción perfecta. Los residuos se abren conforme '
             'aumenta el precio: el error absoluto crece en las casas caras.</p>')

    html(f'<h2 class="seccion" style="margin-top: 36px">Error por rango de precio ({MEJOR}, Test)</h2>')
    rango = pd.DataFrame(resultados['error_por_rango'])
    rango.columns = ['Rango de precio', 'Casas', 'MAE (USD)', 'MAPE (%)', 'Mediana del error (%)']
    tabla_html(rango, formatos={'Casas': lambda x: f'{x:,}', 'MAE (USD)': lambda x: f'{x:,.0f}',
                                'MAPE (%)': lambda x: f'{x:.2f}', 'Mediana del error (%)': lambda x: f'{x:.2f}'},
               der=['Casas', 'MAE (USD)', 'MAPE (%)', 'Mediana del error (%)'], fuerte=['Rango de precio'])

    with st.container(key='grafica3'):
        html(f'<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0 0 8px">'
             f'<b>Importancia de las variables por permutación ({MEJOR})</b>'
             '<span class="leyenda">Caída del R² en Test al desordenar la variable</span></div></div>')
        imp = pd.DataFrame(resultados['importancias'])
        imp['Variable'] = imp['Variable'].map(lambda c: NOMBRES.get(c, c))
        barras_imp = alt.Chart(imp).mark_bar(color=PRIMARIO, cornerRadiusEnd=3).encode(
            y=alt.Y('Variable:N', sort='-x', title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X('Caída de R²:Q', title=None),
            tooltip=['Variable', alt.Tooltip('Caída de R²:Q', format='.4f')])
        barras_imp = barras_imp.properties(padding={'left': 24, 'right': 8, 'top': 5, 'bottom': 5})
        st.altair_chart(base_grafica(barras_imp, 420), width='stretch', theme=None)


# ================================================================ 6. PREDICCIÓN
elif seccion == 'Predicción':
    modelos = obtener_modelos()
    html('<h1 class="titulo">Predicción del precio</h1>'
         '<p class="texto">Captura las características de una vivienda y presiona Estimar. Se muestran las predicciones '
         'de los tres modelos guardados.</p>')
    ubicacion = casas.groupby('codigo_postal').agg(latitud=('latitud', 'median'), longitud=('longitud', 'median'),
                                                   mediana=('precio', 'median'), ventas=('precio', 'size'),
                                                   vecinos=('superficie_vecinos_m2', 'median'))
    lista_cp = list(ubicacion.sort_values('mediana').index)

    with st.form('formulario'):
        html('<h2 class="subseccion">Ubicación y fecha</h2>')
        u1, u2 = st.columns(2)
        cp = u1.selectbox('Código postal', lista_cp, index=lista_cp.index('98103'),
                          format_func=lambda c: f'{c}  ·  mediana {ubicacion.loc[c, "mediana"] / 1000:,.0f} mil USD')
        meses = u2.slider('Mes de venta (0 = mayo 2014, 12 = mayo 2015)', 0, 12, 12)
        html('<h2 class="subseccion" style="margin-top: 16px">Tamaño y distribución</h2>')
        d1, d2, d3, d4 = st.columns(4)
        superficie = d1.number_input('Superficie construida (m²)', 30.0, 1300.0, 180.0, step=5.0)
        terreno = d2.number_input('Terreno (m²)', 50.0, 160000.0, 500.0, step=10.0)
        habitaciones = d3.number_input('Habitaciones', 1, 11, 3)
        banos = d4.number_input('Baños', 0.5, 8.0, 2.0, step=0.25)
        html('<h2 class="subseccion" style="margin-top: 16px">Calidad y entorno</h2>')
        e1, e2, e3, e4 = st.columns(4)
        calidad = e1.slider('Calidad de construcción', 3, 13, 8)
        condicion = e2.slider('Condición', 1, 5, 3)
        antiguedad = e3.number_input('Antigüedad (años)', 0, 120, 20)
        pisos = e4.selectbox('Pisos', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)
        f1, f2, f3, f4 = st.columns(4)
        vista = f1.slider('Vista (0 a 4)', 0, 4, 0)
        frente = f2.selectbox('Frente al agua', ['No', 'Sí'])
        renovada = f3.selectbox('Renovada', ['No', 'Sí'])
        vecinos = f4.number_input('Superficie de los vecinos (m²)', 40.0, 600.0,
                                  float(round(ubicacion.loc['98103', 'vecinos'])), step=5.0)
        enviado = st.form_submit_button('Estimar', type='primary', width='stretch')

    if enviado:
        entrada = pd.DataFrame([{
            'habitaciones': habitaciones, 'banos': banos, 'superficie_m2': superficie, 'terreno_m2': terreno,
            'pisos': pisos, 'frente_agua': int(frente == 'Sí'), 'vista': vista, 'condicion': condicion,
            'calidad': calidad, 'antiguedad': antiguedad, 'renovada': int(renovada == 'Sí'),
            'latitud': ubicacion.loc[cp, 'latitud'], 'longitud': ubicacion.loc[cp, 'longitud'],
            'superficie_vecinos_m2': vecinos, 'meses_desde_inicio': meses, 'codigo_postal': cp,
        }])[VARIABLES]
        avisos = []
        for col, valor in [('superficie_m2', superficie), ('terreno_m2', terreno), ('superficie_vecinos_m2', vecinos)]:
            lo, hi = casas[col].quantile(0.005), casas[col].quantile(0.995)
            if not lo <= valor <= hi:
                avisos.append(f'<b>{NOMBRES[col]}</b> = {valor:,.0f} está fuera del 99 % central de los datos '
                              f'({lo:,.0f} a {hi:,.0f}); la predicción es menos confiable (extrapolación).')
        if superficie / habitaciones < 12:
            avisos.append('La superficie por habitación es muy pequeña; revisa los datos capturados.')
        mae = test.set_index('Modelo')['MAE']
        filas = []
        for nombre, modelo in modelos.items():
            filas.append({'Modelo': nombre, 'Precio': float(modelo.predict(entrada)[0]), 'MAE': float(mae[nombre])})
        st.session_state['prediccion'] = {'filas': filas, 'avisos': avisos, 'cp': cp, 'superficie': superficie}

    if 'prediccion' in st.session_state:
        pred = st.session_state['prediccion']
        html('<h2 class="seccion" style="margin-top: 36px">Resultado</h2>')
        for a in pred['avisos']:
            aviso(a, 'alerta')
        principal = next(f for f in pred['filas'] if f['Modelo'] == MEJOR)
        cp = pred['cp']
        html(f"""
        <div class="cifras tres-c">
          <div class="cifra destacada"><span class="rotulo">Precio estimado ({NOMBRE_CORTO[MEJOR]})</span>
            <span class="valor">{principal['Precio']:,.0f} USD</span>
            <span class="detalle">error promedio en Test ± {principal['MAE']:,.0f} USD</span></div>
          <div class="cifra"><span class="rotulo">Precio por m² estimado</span>
            <span class="valor">{principal['Precio'] / pred['superficie']:,.0f}</span>
            <span class="detalle">USD por m²</span></div>
          <div class="cifra"><span class="rotulo">Mediana del código postal {cp}</span>
            <span class="valor">{ubicacion.loc[cp, 'mediana']:,.0f}</span>
            <span class="detalle">{int(ubicacion.loc[cp, 'ventas'])} ventas en los datos</span></div>
        </div>
        """)
        tope = max(f['Precio'] + f['MAE'] for f in pred['filas']) * 1.1
        barras = ''
        for f in pred['filas']:
            ancho = max(f['Precio'], 0) / tope * 100
            ini, fin = max(f['Precio'] - f['MAE'], 0) / tope * 100, (f['Precio'] + f['MAE']) / tope * 100
            barras += (f'<div class="fila"><span>{NOMBRE_CORTO[f["Modelo"]]}</span><div class="pista">'
                       f'<div class="barra" style="width:{ancho:.2f}%;background:{COLOR_MODELO[f["Modelo"]]}"></div>'
                       f'<div class="rango" style="left:{ini:.2f}%;width:{fin - ini:.2f}%"></div></div>'
                       f'<span>{f["Precio"]:,.0f}</span></div>')
        leyenda = ''.join(f'<span><i style="background:{c}"></i>{NOMBRE_CORTO[m]}</span>' for m, c in COLOR_MODELO.items())
        html(f"""
        <figure class="figura" style="margin-left:0;margin-right:0">
          <figcaption class="cabeza"><b>Precio estimado por modelo (USD)</b><span class="leyenda">{leyenda}</span></figcaption>
          <div class="barras">{barras}</div>
          <p class="nota" style="margin-top: 18px">La franja gris es ± el error absoluto medio (MAE) de cada modelo en Test.
          Es una estimación orientativa con datos de King County 2014–2015, no un avalúo profesional.</p>
        </figure>
        """)


# ================================================================ 7. CONCLUSIONES
else:
    ols = resultados['ols']
    lineal = test[test['Modelo'] == 'Regresión lineal'].iloc[0]
    anova = pd.DataFrame(resultados['anova']).set_index('Factor')
    corr = pd.DataFrame(resultados['correlaciones']).set_index('Variable')
    imp = pd.DataFrame(resultados['importancias'])
    top_imp = ', '.join(NOMBRES.get(v, v).lower() for v in imp['Variable'].head(3))
    html(f"""
    <h1 class="titulo">Conclusiones y recomendaciones</h1>
    <h2 class="seccion">Hallazgos</h2>
    <ol class="lista">
      <li><b>Factores más influyentes.</b> La superficie construida (r = {corr.loc['superficie_m2', 'r de Pearson']:.3f}) y la
      calidad de construcción (r = {corr.loc['calidad', 'r de Pearson']:.3f}) son las variables con mayor correlación con el
      precio. En el ANOVA, la calidad explica el {anova.loc['Calidad de construcción', 'eta²'] * 100:.1f} % de la variación del
      precio (η²) y la vista el {anova.loc['Vista', 'eta²'] * 100:.1f} %; todos los factores fueron significativos (p &lt; 0.05),
      también con Kruskal-Wallis.</li>
      <li><b>Ubicación.</b> La mediana del precio cambia varias veces de un código postal a otro. En el modelo seleccionado,
      las variables más importantes son {top_imp}.</li>
      <li><b>Regresión lineal.</b> Con 15 variables numéricas el modelo lineal explica R² = {ols['r2']:.3f} del precio en
      entrenamiento. Al agregar el código postal, la regresión lineal llega a R² = {lineal['R2']:.3f} en Test, pero queda
      por debajo de los modelos de árboles, que captan relaciones no lineales entre ubicación, tamaño y precio.</li>
      <li><b>Mejor modelo.</b> {MEJOR}: R² = {fila_mejor['R2']:.3f}, RMSE = {fila_mejor['RMSE']:,.0f} USD y
      MAE = {fila_mejor['MAE']:,.0f} USD en Test, el menor error de todos los modelos comparados.</li>
    </ol>
    <h2 class="seccion" style="margin-top: 36px">Limitaciones</h2>
    <ol class="lista">
      <li>Los datos son de un solo mercado (King County, Washington) y de 2014–2015; no sirven para valuar viviendas de
      Chihuahua u otro mercado sin reentrenar con datos locales.</li>
      <li>El dataset no trae acceso a servicios (escuelas, hospitales, transporte) ni índices económicos; la ubicación
      los resume solo de forma indirecta.</li>
      <li>El error crece en las casas de lujo, que son pocas en los datos.</li>
      <li>Un año de ventas no permite modelar ciclos de precios de largo plazo.</li>
    </ol>
    <h2 class="seccion" style="margin-top: 36px">Recomendaciones</h2>
    <ol class="lista">
      <li>Usar el modelo como estimación inicial de precio y no como avalúo; reportar siempre el margen de error.</li>
      <li>Agregar datos de servicios y de nivel socioeconómico por código postal (censo) para cubrir el entorno.</li>
      <li>Reentrenar periódicamente con ventas recientes y vigilar el error por rango de precio.</li>
      <li>Para casas de lujo, complementar con una valuación especializada.</li>
    </ol>
    """)
