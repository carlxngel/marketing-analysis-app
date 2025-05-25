import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Análisis y Optimización de Campañas de Marketing", layout="wide")

# Title and divider
st.title("Análisis y Optimización de Campañas de Marketing")
st.markdown("---")

# --- Sidebar Navigation ---
st.sidebar.image("logo upgrade.png", width=100)
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Seleccione una sección",
    ("Introducción", "Preprocesamiento", "Análisis Exploratorio (EDA)", "Insights y Recomendaciones")
)


# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("limpio_marketingcampaigns.csv")
        # Convertir columnas para análisis
        df['inversión_num'] = df['inversión'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        df['facturación_num'] = df['facturación'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        df['roi_num'] = df['retorno inversión'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        df['ratio_conv_num'] = df['ratio conversión'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        df['duracion_num'] = pd.to_numeric(df['duración días'], errors='coerce')
        df['beneficio_neto_num'] = df['beneficio neto'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        df['fecha inicio'] = pd.to_datetime(df['fecha inicio'], errors='coerce')
        df['mes'] = df['fecha inicio'].dt.month
        return df
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        # Crear DataFrame vacío para evitar errores
        return pd.DataFrame()

df = load_data()

# --- Introducción ---
if section == "Introducción":
    # Custom CSS for consistent styling
    st.markdown("""
    <style>
    /* General styles */
    .section-title {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin: 2em 0 1em 0;
        padding-bottom: 0.5em;
        border-bottom: 3px solid #ddd;  /* Increased border thickness */
    }
    
    .card {
        background-color: white;
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);  /* Increased shadow */
        margin: 1em 0;
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;  /* Added border */
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);  /* Increased hover shadow */
    }
    
    .metric-card {
        background-color: #f8f9fa;  /* Light background */
        padding: 1.5em;
        border-radius: 10px;
        text-align: center;
        border-left: 6px solid #1f77b4;  /* Thicker accent border */
        margin: 0.5em;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);  /* Added shadow */
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #0d6efd;  /* Brighter blue */
        margin: 0.2em 0;
    }
    
    .metric-label {
        color: #343a40;  /* Darker text */
        font-size: 1em;
        font-weight: 500;  /* Semi-bold */
    }
    
    .process-step {
        background-color: white;
        padding: 1.2em;
        border-radius: 8px;
        margin: 0.5em 0;
        display: flex;
        align-items: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);  /* Increased shadow */
        border: 1px solid #dee2e6;  /* Added border */
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    
    .animated {
        animation: fadeIn 1s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # Project Objectives Section
    st.markdown('<h1 class="section-title animated">🎯 Objetivos del Proyecto</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    objectives = [
        {"icon": "📊", "title": "Identificar Canales Óptimos", "desc": "Análisis de eficiencia por canal y tipo de campaña"},
        {"icon": "💰", "title": "Optimizar Presupuestos", "desc": "Maximización del ROI en cada campaña"},
        {"icon": "📈", "title": "Detectar Estacionalidad", "desc": "Identificación de patrones temporales"},
        {"icon": "💡", "title": "Generar Recomendaciones", "desc": "Estrategias basadas en datos"}
    ]
    
    for i, obj in enumerate(objectives):
        with col1 if i < 2 else col2:
            st.markdown(f"""
            <div class="card animated">
                <h3>{obj['icon']} {obj['title']}</h3>
                <p>{obj['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Methodology Section
    st.markdown('<h1 class="section-title animated">🛠️ Metodología</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["💻 Herramientas", "📝 Proceso", "⚡ Técnicas"])
    
    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        tools = [
            {"icon": "🐍", "name": "Python", "desc": "Análisis de datos"},
            {"icon": "📊", "name": "Pandas", "desc": "Manipulación de datos"},
            {"icon": "💻", "name": "VS Code", "desc": "Desarrollo"},
            {"icon": "📈", "name": "Power BI", "desc": "Visualización"}
        ]
        
        for col, tool in zip([col1, col2, col3, col4], tools):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2em">{tool['icon']}</div>
                    <div class="metric-value">{tool['name']}</div>
                    <div class="metric-label">{tool['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[1]:
        process_steps = [
            {"icon": "🔍", "step": "Exploración inicial", "desc": "Análisis preliminar"},
            {"icon": "🧹", "step": "Limpieza", "desc": "Preparación de datos"},
            {"icon": "📊", "step": "Análisis", "desc": "Identificación de patrones"},
            {"icon": "📈", "step": "Visualización", "desc": "Creación de dashboards"},
            {"icon": "💡", "step": "Insights", "desc": "Conclusiones"}
        ]
        
        for step in process_steps:
            st.markdown(f"""
            <div class="process-step">
                <div style="font-size: 1.5em; margin-right: 1em">{step['icon']}</div>
                <div>
                    <h4 style="margin: 0">{step['step']}</h4>
                    <p style="margin: 0; color: #666">{step['desc']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tabs[2]:
        col1, col2 = st.columns(2)
        techniques = [
            {"icon": "📊", "name": "Análisis Estadístico"},
            {"icon": "📈", "name": "Visualización Avanzada"},
            {"icon": "🔍", "name": "Detección de Patrones"},
            {"icon": "🔗", "name": "Análisis de Correlaciones"},
            {"icon": "🤖", "name": "Machine Learning Básico"}
        ]
        
        for i, tech in enumerate(techniques):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; align-items: center">
                        <span style="font-size: 1.5em; margin-right: 0.5em">{tech['icon']}</span>
                        <span>{tech['name']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- Preprocesamiento ---
elif section == "Preprocesamiento":
    # Custom CSS para mantener consistencia con la sección de introducción
    st.markdown("""
    <style>
    .data-card {
        background-color: white;
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin: 1em 0;
        border: 1px solid #e0e0e0;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.2em;
        border-radius: 8px;
        text-align: center;
        border-left: 6px solid #1f77b4;
        margin: 0.5em;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    
    .step-container {
        background-color: white;
        padding: 1em;
        border-radius: 8px;
        margin: 0.5em 0;
        border: 1px solid #dee2e6;
        transition: transform 0.3s ease;
    }
    
    .step-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
    
    .code-container {
        background-color: #f8f9fa;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align: center; color: black;">Preprocesamiento y Limpieza de Datos</h1>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Registros</h3>
            <h2>1,000</h2>
            <p>Campañas totales</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>🧹 Limpieza</h3>
            <h2>100%</h2>
            <p>Datos válidos</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>📈 Variables</h3>
            <h2>15</h2>
            <p>Campos finales</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3>⚡ Enriquecimiento</h3>
            <h2>6</h2>
            <p>Nuevas métricas</p>
        </div>
        """, unsafe_allow_html=True)

    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Inspección", "🧹 Limpieza", "📊 Normalización", "💫 Enriquecimiento", "✅ Resultado"])
    
    with tab1:
        st.markdown("""
        <div class="data-card">
            <h3>🔍 Inspección Inicial del Dataset</h3>
            <p>Proceso sistemático de revisión de la calidad y estructura de los datos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-container">
                <h4>📋 Dimensiones Detectadas</h4>
                <ul>
                    <li>1000 filas (campañas)</li>
                    <li>10 columnas (variables)</li>
                    <li>12 duplicados identificados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-container">
                <h4>⚠️ Problemas Detectados</h4>
                <ul>
                    <li>42 fechas fin faltantes</li>
                    <li>38 valores ROI ausentes</li>
                    <li>35 tasas conversión nulas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


    with tab2:
        st.markdown("""
        <div class="data-card">
            <h3>🧹 Proceso de Limpieza</h3>
            <p>Tratamiento sistemático de valores nulos, duplicados y anomalías.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-container">
                <h4>📊 Datos Numéricos</h4>
                <ul>
                    <li>Imputación por mediana</li>
                    <li>Eliminación valores negativos</li>
                    <li>Corrección formatos decimales</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-container">
                <h4>📝 Datos Categóricos</h4>
                <ul>
                    <li>Etiqueta "sin datos" para nulos</li>
                    <li>Normalización de categorías</li>
                    <li>Corrección de errores tipográficos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        

    with tab3:
        st.markdown("""
        <div class="data-card">
            <h3>📊 Normalización de Datos</h3>
            <p>Estandarización de formatos y unidades para análisis consistente.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ejemplos de normalización
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="step-container">
            <h4>🗓️ Fechas</h4>
            <code>01/01/2024</code> → <code>2024-01-01</code>
            <br>
            <code>1-ene-24</code> → <code>2024-01-01</code>
            <br>
            <code>2024.01.01</code> → <code>2024-01-01</code>
            <br>
            <code>01 enero 2024</code> → <code>2024-01-01</code>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-container">
            <h4>💶 Valores Monetarios</h4>
            <code>1000.50</code> → <code>1.000,50</code>
            <br>
            <code>1,000.50</code> → <code>1.000,50</code>
            <br>
            <code>1000,5</code> → <code>1.000,50</code>
            <br>
            <code>1.000,5€</code> → <code>1.000,50</code>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="step-container">
            <h4>📊 Porcentajes</h4>
            <code>50%</code> → <code>0.50</code>
            <br>
            <code>0,5</code> → <code>0.50</code>
            <br>
            <code>50.0%</code> → <code>0.50</code>
            <br>
            <code>50,00%</code> → <code>0.50</code>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="data-card">
            <h3>💫 Enriquecimiento de Datos</h3>
            <p>Creación de nuevas variables y métricas derivadas para análisis profundo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Variables derivadas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="step-container">
                <h4>⏱️ Temporales</h4>
                <ul>
                    <li>Duración en días</li>
                    <li>Categoría duración</li>
                    <li>Mes de inicio</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-container">
                <h4>💰 Financieras</h4>
                <ul>
                    <li>Beneficio neto</li>
                    <li>ROI ajustado</li>
                    <li>Categoría inversión</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="step-container">
                <h4>📈 Rendimiento</h4>
                <ul>
                    <li>Éxito campaña</li>
                    <li>Eficiencia relativa</li>
                    <li>Score conversión</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab5:
        st.markdown("""
        <div class="data-card">
            <h3>✅ Resultado Final</h3>
            <p>Dataset limpio y enriquecido listo para análisis exploratorio.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas finales
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-container">
                <h4>📊 Estadísticas Finales</h4>
                <ul>
                    <li>988 registros únicos</li>
                    <li>15 variables totales</li>
                    <li>0 valores nulos</li>
                    <li>100% datos válidos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="step-container">
                <h4>🎯 Mejoras Implementadas</h4>
                <ul>
                    <li>6 nuevas métricas derivadas</li>
                    <li>Formatos estandarizados</li>
                    <li>Categorización completa</li>
                    <li>Documentación detallada</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Vista previa del dataset
        st.markdown("<h4 style='text-align: center;'>Vista Previa del Dataset Final</h4>", unsafe_allow_html=True)
        st.dataframe(df.head())

# --- EDA ---
elif section == "Análisis Exploratorio (EDA)":
    st.markdown("""
    <style>
    .data-card {
        background-color: white;
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin: 1em 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Canales de Marketing", "Tipos de campaña", "Rendimiento y ROI", "Patrones Temporales"])

    with tab1:
        st.markdown("""
        <div class="data-card">
            <h3>1. Análisis por Canal de Marketing</h3>
            <p>Exploración detallada del rendimiento y distribución de canales de marketing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de campañas por canal
            channel_counts = df['canal'].value_counts()
            fig_channel_dist = px.pie(values=channel_counts.values, 
                        names=channel_counts.index, 
                        title='Distribución de Campañas por Canal', 
                        hole=0.4)
            st.plotly_chart(fig_channel_dist, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights de Distribución:</strong>
            <ul>
                <li>El canal Promotion es ligeramente más utilizado.</li>
                <li>Las empresas utilizan de forma equilibrada los diferentes canales de marketing, sin depender excesivamente de uno solo.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # ROI promedio por canal
            channel_roi = df.groupby('canal')['roi_num'].mean().reset_index()
            fig_channel_roi = px.bar(channel_roi, 
                       x='canal', 
                       y='roi_num',
                       title='ROI Promedio por Canal',
                       color='canal')
            st.plotly_chart(fig_channel_roi, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights de ROI:</strong>
            <ul>
                <li>Referral lidera en ROI (0.575)</li>
                <li>Promotion muestra ROI más bajo pese a mayor uso</li>
                <li>Las diferencias de ROI entre canales no son significativas</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
            # Análisis de Campaña
            st.markdown("""
            <div class="data-card">
                <h3>2. Análisis de Campaña</h3>
                <p>Evaluación de ingresos y duración por tipo de campaña.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Ingresos promedio por tipo de campaña
                df_filtered = df[~df['tipo'].isin(['B2B', 'sin datos'])]
                campaign_revenue = df_filtered.groupby('tipo')['facturación_num'].mean().reset_index()
                fig_campaign_rev = px.bar(campaign_revenue,
                            x='tipo',
                            y='facturación_num',
                            title='Ingresos Promedio por Tipo de Campaña',
                            color='tipo')
                fig_campaign_rev.update_layout(xaxis_title="Tipo de Campaña",
                             yaxis_title="Facturación Promedio")
                st.plotly_chart(fig_campaign_rev, use_container_width=True)

                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
                <strong>Insights de Ingresos:</strong>
                <ul>
                <li>Las campañas de social media, podcast y email generan los ingresos promedio más altos, superando los 500k.</li>
                <li>Las campañas de webinar generar ingresos ligeramente inferiores a las otras tres principales.</li>
                <li>Las campañas de eventos tienen una facturación promedio significativamente menor.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Distribución de duración por tipo de campaña
                df_filtered = df[~df['tipo'].isin(['B2B', 'sin datos'])]
                fig_duration = px.box(df_filtered,
                        x='tipo',
                        y='duracion_num',
                        color='tipo', 
                        title='Distribución de Duración por Tipo de Campaña')
                fig_duration.update_layout(xaxis_title="Tipo de Campaña",
                             yaxis_title="Duración (días)")
                st.plotly_chart(fig_duration, use_container_width=True)

                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
                <strong>Insights de Duración:</strong>
                <ul>
                <li>Todas las campañas tienen una distribución de duración similar.</li>
                <li>No se observa una diferencia clara de duración entre los tipos de campaña.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    with tab3:
        st.markdown("""
        <div class="data-card">
            <h3>3. Análisis de Rendimiento y ROI</h3>
            <p>Evaluación de la relación entre inversión, ROI y rendimiento general.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot de Inversión vs ROI
            fig_inv_roi = px.scatter(df, 
                       x='inversión_num',
                       y='roi_num',
                       color='canal',
                       title='Relación entre Inversión y ROI')
            st.plotly_chart(fig_inv_roi, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights Inversión vs ROI:</strong>
            <ul>
            <li>No existe una correlación fuerte entre el nivel de inversión y el ROI obtenido</li>
            <li>Las campañas de menor inversión muestran mayor variabilidad en el ROI</li>
            <li>Se identifica un punto óptimo de inversión entre 0.3-0.5M</li>
            <li>Los diferentes canales muestran patrones similares de ROI</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Histograma de ROI
            fig_roi_hist = px.histogram(df, 
                          x='roi_num',
                          title='Distribución del ROI',
                          nbins=30)
            st.plotly_chart(fig_roi_hist, use_container_width=True)

            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights Distribución ROI:</strong>
            <ul>
            <li>La distribución del ROI muestra una forma aproximadamente normal</li>
            <li>La mayoría de campañas tienen un ROI entre 0.4 y 0.7</li>
            <li>Hay pocas campañas con ROI extremadamente alto (>0.8) o bajo (<0.2)</li>
            <li>El ROI promedio se sitúa alrededor de 0.54</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="data-card">
            <h3>4. Análisis de Patrones Temporales</h3>
            <p>Identificación de estacionalidad y tendencias temporales.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI promedio por mes
            monthly_roi = df.groupby('mes')['roi_num'].mean().reset_index()
            fig_monthly_roi = px.line(monthly_roi,
                                    x='mes',
                                    y='roi_num',
                                    title='ROI Promedio por Mes',
                                    markers=True)
            st.plotly_chart(fig_monthly_roi, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights de Estacionalidad:</strong>
            <ul>
            <li>Se identifican 4 picos claros de ROI en los meses 1, 3, 9 y 12</li>
            <li>El mes de julio (7) muestra el ROI más bajo del año</li>
            <li>Existe un patrón estacional trimestral consistente</li>
            <li>Los picos coinciden con cierres de trimestre fiscal</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Duración vs Facturación
            fig_dur_fact = px.scatter(df,
                                    x='duracion_num',
                                    y='facturación_num',
                                    color='canal',
                                    title='Duración vs Facturación')
            st.plotly_chart(fig_dur_fact, use_container_width=True)
            
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4;">
            <strong>Insights de Duración vs Facturación:</strong>
            <ul>
            <li>Campañas más largas no necesariamente generan mayor facturación</li>
            <li>La duración óptima se encuentra entre 300-500 días</li>
            <li>No hay diferencias significativas entre canales</li>
            <li>Las campañas cortas muestran mayor variabilidad en facturación</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


# --- Insights y Recomendaciones ---
elif section == "Insights y Recomendaciones":
    # Custom CSS para mantener consistencia con introducción
    st.markdown("""
    <style>
    .card {
        background-color: white;
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin: 1em 0;
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5em;
        border-radius: 10px;
        text-align: center;
        border-left: 6px solid #1f77b4;
        margin: 0.5em;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    
    .section-title {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin: 2em 0 1em 0;
        padding-bottom: 0.5em;
        border-bottom: 3px solid #ddd;
    }
    
    .process-step {
        background-color: white;
        padding: 1.2em;
        border-radius: 8px;
        margin: 0.5em 0;
        display: flex;
        align-items: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    .animated {
        animation: fadeIn 1s ease;
    }
    
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    st.markdown('<h1 class="section-title animated">📊 Insights y Recomendaciones</h1>', unsafe_allow_html=True)

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        {"icon": "📈", "value": "0.54", "label": "ROI Promedio", "delta": "+15%"},
        {"icon": "🎯", "value": "Referral", "label": "Mejor Canal", "delta": "0.575 ROI"},
        {"icon": "⏱️", "value": "400 días", "label": "Duración Óptima", "delta": "+25%"},
        {"icon": "💡", "value": "20%", "label": "Potencial Mejora", "delta": "proyectado"}
    ]

    for col, metric in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card animated">
                <div style="font-size: 2em">{metric['icon']}</div>
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['label']}</div>
                <div style="color: #28a745; font-size: 0.9em">▲ {metric['delta']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Principales hallazgos
    st.markdown('<h2 class="section-title">🔍 Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    findings = [
        {
            "icon": "📊", 
            "title": "Canales", 
            "desc": "Referral lidera ROI con 57.5%, superando por 25% el promedio. "
            "Las campañas de referidos muestran mayor retención y valor del cliente a largo plazo."
        },
        {
            "icon": "💰", 
            "title": "Inversión", 
            "desc": "Punto óptimo de inversión identificado entre 0.3-0.5M con ROI promedio de 0.54. "
            "Inversiones mayores muestran rendimientos decrecientes."
        },
        {
            "icon": "📈", 
            "title": "Conversión", 
            "desc": "Email destaca con tasa de conversión 35% superior al promedio. "
            "Especialmente efectivo en retención de clientes y reactivación."
        },
        {
            "icon": "🕒", 
            "title": "Temporalidad", 
            "desc": "4 picos estacionales identificados en Q1,Q2,Q3,Q4 con máximos en marzo y septiembre. "
            "Las campañas alineadas muestran 40% mejor rendimiento."
        }
    ]
    
    for i, finding in enumerate(findings):
        with col1 if i < 2 else col2:
            st.markdown(f"""
            <div class="card animated" style="height: 180px; overflow: hidden;">
                <h3>{finding['icon']} {finding['title']}</h3>
                <p>{finding['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Plan de acción basado en insights
    st.markdown('<h2 class="section-title">🎯 Recomendaciones</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="process-step">
            <div style="font-size: 1.5em; margin-right: 1em">⚡</div>
            <div>
                <h4>Potenciar Canal Referral</h4>
                <p>Especialmente con campañas de bajo coste y alta segmentación como Email Marketing.</p>
                
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="process-step">
            <div style="font-size: 1.5em; margin-right: 1em">🎯</div>
            <div>
                <h4>Ajuste del ROI objetivo</h4>
                <p>Establecer un ROI objetivo mínimo de 0.6. Ajustar campañas que estén por debajo.</p>
        
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="process-step">
            <div style="font-size: 1.5em; margin-right: 1em">⏱️</div>
            <div>
                <h4>Ajuste Temporal</h4>
                <p>Priorizar meses pico (marzo y septiembre). Optimizar duración a 400 días.</p>
        
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="process-step">
            <div style="font-size: 1.5em; margin-right: 1em">💰</div>
            <div>
                <h4>Inversión Óptima</h4>
                <p>Mantener rango 0.3-0.5M. Evitar inversiones excesivas.</p>
        
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("**Proyecto desarrollado para Upgrade Hub por Carla Molina - 2025**")