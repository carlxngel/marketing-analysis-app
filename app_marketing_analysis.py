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

# --- Sidebar Navigation ---
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Ir a la sección:",
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
    st.title("Análisis y Optimización de Campañas de Marketing")
    st.markdown("""
    ## Proyecto de Análisis Avanzado de Marketing Digital
    
    Este proyecto aborda el análisis exhaustivo de campañas de marketing digital, aplicando técnicas de ciencia de datos para extraer insights accionables y recomendaciones estratégicas. Desde la limpieza y preprocesamiento hasta la visualización avanzada y la interpretación de resultados, este análisis ofrece una visión completa del rendimiento de las campañas.
    
    ### Objetivos del Proyecto:
    
    1. **Identificar los canales y tipos de campañas más eficientes** en términos de ROI y conversión.
    2. **Analizar la relación entre inversión y beneficio** para optimizar la asignación de presupuesto.
    3. **Detectar patrones estacionales** que influyan en el rendimiento de las campañas.
    4. **Proporcionar recomendaciones accionables** basadas en datos para mejorar estrategias futuras.
    
    ### Estructura del Proyecto:
    
    - **Preprocesamiento de datos**: Limpieza, transformación y enriquecimiento del dataset.
    - **Análisis Exploratorio (EDA)**: Visualizaciones y análisis estadísticos para extraer patrones y relaciones.
    - **Insights y Recomendaciones**: Conclusiones clave y propuestas estratégicas.
    
    ### Metodología:
    
    Utilizamos Python como lenguaje principal, con bibliotecas como pandas para manipulación de datos, 
    matplotlib y seaborn para visualización, y técnicas estadísticas para el análisis de rendimiento 
    y correlaciones entre variables.
    """)

    # Mostrar algunos KPIs generales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Campañas", f"{len(df)}")
    with col2:
        st.metric("ROI Promedio", f"{df['roi_num'].mean():.2f}")
    with col3:
        st.metric("Tasa Conversión Media", f"{df['ratio_conv_num'].mean():.2f}")
    with col4:
        st.metric("Canales Utilizados", f"{df['canal'].nunique()}")

# --- Preprocesamiento ---
elif section == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")
    
    st.markdown("""
    ## Proceso de Preprocesamiento y Limpieza de Datos
    
    El preprocesamiento de datos es fundamental para garantizar la calidad y fiabilidad del análisis posterior. 
    En este proyecto, aplicamos múltiples técnicas para transformar los datos brutos en un conjunto limpio y enriquecido.
    """)

    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inspección Inicial", "Limpieza de Datos", "Normalización", "Enriquecimiento", "Resultado"])
    
    with tab1:
        st.subheader("1. Inspección Inicial y Detección de Problemas")
        st.markdown("""
        ### 1.1. Carga e Inspección del Dataset Original
        
        - **Dimensiones**: Verificación del número de filas y columnas para entender el volumen de datos.
        - **Tipos de datos**: Identificación de tipos incorrectos que pudieran afectar al análisis.
        - **Valores nulos**: Detección de datos faltantes que requieren tratamiento.
        - **Duplicados**: Identificación de filas duplicadas para su eliminación.
        
        **Decisiones técnicas**:
        - Se utilizó `df.info()` para examinar los tipos de datos y detectar columnas problemáticas.
        - Se empleó `df.isnull().sum()` para cuantificar valores nulos por columna.
        - Se implementó un mapa de calor con `sns.heatmap(df.isnull(), cbar=False)` para visualizar la distribución de nulos.
        - Se verificaron duplicados con `df.duplicated().sum()` y se eliminaron con `df.drop_duplicates()`.
        """)
        
        # Mostrar estadísticas del preprocesamiento
        st.code("""
        # Dimensiones del dataframe
        df.shape  # (1000, 10)
        
        # Verificar duplicados
        df.duplicated().sum()  # 12 duplicados
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        
        # Verificar valores nulos
        df.isnull().sum()
        # campaign_name      0
        # start_date         0
        # end_date          42
        # budget            15
        # roi               38
        # type              28
        # target_audience   31
        # channel           22
        # conversion_rate   35
        # revenue           19
        """, language="python")

    with tab2:
        st.subheader("2. Limpieza y Tratamiento de Datos")
        st.markdown("""
        ### 2.1. Tratamiento de Valores Nulos
        
        **Datos numéricos**:
        - En columnas como `budget`, `roi`, `conversion_rate` y `revenue`, se convirtieron primero a formato numérico con `pd.to_numeric()`.
        - Se imputaron los valores nulos con la mediana de cada columna, ya que es más robusta a outliers que la media.
        
        **Datos categóricos**:
        - En columnas como `type`, `target_audience`, `channel` y `end_date`, se reemplazaron los nulos con la etiqueta `"sin datos"`.
        - Esta decisión permite mantener todas las filas para el análisis y distinguir claramente los datos faltantes.
        
        ### 2.2. Eliminación de Valores Anómalos
        
        - Se eliminaron filas con valores negativos en la columna `inversión`, ya que no tienen sentido en el contexto de campañas de marketing.
        - Se optó por conservar otros outliers para su análisis en la fase de EDA, donde se evaluaría su impacto.
        """)
        
        st.code("""
        # Ejemplo de tratamiento de valores nulos numéricos
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
        mediana_budget = df['budget'].median()
        df['budget'] = df['budget'].fillna(mediana_budget)
        
        # Ejemplo de tratamiento de valores nulos categóricos
        df['type'] = df['type'].fillna('sin datos')
        
        # Eliminación de valores negativos en inversión
        df = df[df['inversión'] >= 0].copy()
        """, language="python")

    with tab3:
        st.subheader("3. Normalización y Conversión de Formatos")
        st.markdown("""
        ### 3.1. Estandarización de Fechas
        
        - Las columnas `start_date` y `end_date` se convirtieron a formato `datetime` para facilitar cálculos temporales.
        - Se utilizó el parámetro `errors='coerce'` para convertir valores no parseables a `NaT` (Not a Time).
        - Se detectaron y reportaron fechas no válidas para su posterior revisión.
        
        ### 3.2. Normalización de Formatos Numéricos
        
        - Se crearon funciones personalizadas para convertir valores numéricos al formato español (coma decimal, punto para miles).
        - Este paso fue crucial para mantener la consistencia en la presentación de datos y evitar errores en cálculos posteriores.
        
        ### 3.3. Renombrado de Columnas
        
        - Se renombraron las columnas al español para mantener consistencia en el idioma de todo el dataset.
        - Ejemplos: `campaign_name` → `nombre campaña`, `budget` → `inversión`, etc.
        """)
        
        st.code("""
        # Conversión a datetime
        df['fecha inicio'] = pd.to_datetime(df['fecha inicio'], errors='coerce')
        df['fecha fin'] = pd.to_datetime(df['fecha fin'], errors='coerce')
        
        # Función para formato español
        def formato_espanol(valor):
            try:
                valor_float = float(valor)
                return f"{valor_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except:
                return valor
                
        # Aplicación a columnas numéricas
        columnas_numericas = ['inversión', 'retorno inversión', 'ratio conversión', 'facturación']
        for col in columnas_numericas:
            df[col] = df[col].apply(formato_espanol)
        """, language="python")

    with tab4:
        st.subheader("4. Enriquecimiento del Dataset")
        st.markdown("""
        ### 4.1. Creación de Nuevas Variables Derivadas
        
        **Duración de campañas**:
        - Se calculó la `duración días` como la diferencia entre `fecha fin` y `fecha inicio`.
        - Las duraciones no computables (por fechas incorrectas) se rellenaron con `"sin datos"`.
        
        **Categorización de duración**:
        - Se clasificaron las campañas en `corta`, `media` y `larga` basándose en la mediana de duración.
        - Criterio: < 75% mediana = corta, 75-125% mediana = media, > 125% mediana = larga.
        
        **Beneficio neto**:
        - Se calculó como la diferencia entre `facturación` e `inversión`.
        - Se aplicó formato español para mantener consistencia.
        
        **Éxito de campaña**:
        - Se creó una variable booleana `campaña exitosa` basada en si el beneficio neto es positivo.
        - Permite segmentar rápidamente campañas rentables vs. no rentables.
        
        **Categorización de inversión y beneficio**:
        - Se clasificaron en `bajo`, `medio` y `alto` utilizando la mediana como referencia.
        - Estas categorías facilitan análisis comparativos y segmentación.
        """)
        
        st.code("""
        # Cálculo de duración
        df['duración días'] = (df['fecha fin'] - df['fecha inicio']).dt.days
        df['duración días'] = df['duración días'].fillna('sin datos')
        
        # Categorización de duración
        def categorizar_duracion(dias, mediana):
            try:
                dias = float(dias)
                if np.isnan(dias):
                    return 'sin datos'
                if dias < mediana * 0.75:
                    return 'corta'
                elif dias <= mediana * 1.25:
                    return 'media'
                else:
                    return 'larga'
            except:
                return 'sin datos'
                
        # Beneficio neto
        def calcular_beneficio_neto(row):
            try:
                inversion = float(str(row['inversión']).replace('.', '').replace(',', '.'))
                facturacion = float(str(row['facturación']).replace('.', '').replace(',', '.'))
                beneficio = facturacion - inversion
                return f"{beneficio:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except:
                return 'sin datos'
                
        df['beneficio neto'] = df.apply(calcular_beneficio_neto, axis=1)
        """, language="python")

    with tab5:
        st.subheader("5. Dataset Final Preprocesado")
        st.markdown("""
        ### 5.1. Resultado del Preprocesamiento
        
        El dataset resultante está limpio, enriquecido y listo para el análisis exploratorio, con:
        
        - **0 valores nulos** en todas las columnas.
        - **0 duplicados** en el conjunto de datos.
        - **Formatos consistentes** en fechas y valores numéricos.
        - **Variables derivadas** que amplían las posibilidades de análisis.
        - **Categorías estandarizadas** para facilitar la segmentación.
        
        ### 5.2. Decisiones técnicas clave
        
        - Usar la **mediana para imputación** de valores numéricos por su robustez ante outliers.
        - Mantener la **etiqueta "sin datos"** en vez de eliminar filas para conservar el máximo de información.
        - Crear **categorizaciones relativas a la mediana** para proporcionar contexto a los valores.
        - Calcular el **beneficio neto y éxito** como métricas fundamentales para evaluar campañas.
        """)
        
        # Mostrar muestra del dataset limpio
        st.subheader("Vista previa del dataset procesado")
        st.dataframe(df.head())
        
        # Mostrar estadísticas del dataset limpio
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total filas después de limpieza", len(df))
            st.metric("Valores nulos remanentes", df.isnull().sum().sum())
        with col2:
            st.metric("Nuevas columnas derivadas", 6)
            st.metric("Campañas exitosas (%)", f"{(df['campaña exitosa'] == 'Sí').mean()*100:.1f}%")

# --- EDA ---
elif section == "Análisis Exploratorio (EDA)":
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    st.markdown("""
    ## Análisis Exploratorio de Datos
    
    El EDA nos permite descubrir patrones, relaciones y anomalías en los datos a través de visualizaciones y análisis estadísticos. 
    En este proyecto, realizamos múltiples análisis para comprender el rendimiento de las campañas de marketing desde diferentes ángulos.
    """)
    
    # Tabs para organizar los diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["Análisis por Canal", "Rendimiento y ROI", "Patrones Temporales", "Relaciones entre Variables"])
    
    with tab1:
        st.subheader("1. Análisis por Canal de Marketing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico 1: Número de campañas por canal
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            canal_counts = df['canal'].value_counts()
            bars = sns.barplot(
                x=canal_counts.index, 
                y=canal_counts.values,
                palette="viridis"
            )
            for i, v in enumerate(canal_counts.values):
                ax1.text(i, v, str(v), ha='center', va='bottom')
            ax1.set_title('Número de campañas por canal de marketing')
            ax1.set_xlabel('Canal')
            ax1.set_ylabel('Número de campañas')
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
            st.markdown("""
            **Análisis de distribución de canales:**
            
            - El gráfico muestra una distribución relativamente equilibrada entre los principales canales: Promotion, Referral, Organic y Paid.
            - Promotion es el canal más utilizado, seguido de cerca por Referral.
            - Esta distribución equilibrada sugiere una estrategia multicanal, donde no se depende excesivamente de un solo canal.
            """)
        
        with col2:
            # Gráfico 2: ROI promedio por canal
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            roi_by_channel = df.groupby('canal')['roi_num'].mean().sort_values(ascending=True)
            bars = plt.barh(
                roi_by_channel.index,
                roi_by_channel.values,
                alpha=0.8,
                color=plt.cm.viridis(np.linspace(0, 1, len(roi_by_channel)))
            )
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width, 
                        bar.get_y() + bar.get_height()/2,
                        f'{roi_by_channel.values[i]:.3f}',
                        ha='left',
                        va='center',
                        fontweight='bold')
            ax2.set_title('ROI promedio por canal de marketing')
            ax2.set_xlabel('ROI')
            ax2.set_ylabel('Canal')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            st.markdown("""
            **Análisis de eficiencia por canal:**
            
            - Referral es el canal con mayor ROI promedio (0.575), lo que lo convierte en el más eficiente.
            - Paid y Promotion muestran un ROI similar (aproximadamente 0.54).
            - Organic tiene el ROI más bajo (0.51), aunque la diferencia con los demás canales no es muy significativa.
            - Este análisis sugiere que las campañas de referral ofrecen el mejor retorno sobre la inversión.
            """)
        
        # Gráfico 3: Relación entre Canal, Inversión y Facturación
        st.subheader("Relación entre Canal, Inversión y Facturación")
        fig3 = go.Figure()
        
        for canal in df['canal'].unique():
            df_canal = df[df['canal'] == canal]
            fig3.add_trace(go.Scatter(
                x=df_canal['inversión_num'],
                y=df_canal['facturación_num'],
                mode='markers',
                marker=dict(
                    size=df_canal['facturación_num']/10000,
                    sizemode='area',
                    sizeref=2.*max(df['facturación_num'])/(40.**2),
                    sizemin=4
                ),
                name=canal
            ))
            
        fig3.update_layout(
            title='Relación entre Canal, Inversión y Facturación',
            xaxis_title='Inversión (€)',
            yaxis_title='Facturación (€)',
            height=600
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        **Análisis de la relación inversión-facturación por canal:**
        
        - La visualización muestra que no hay un canal que destaque claramente por tener campañas con facturación significativamente mayor.
        - Todos los canales presentan una distribución similar de inversión y facturación.
        - Se observa una concentración de campañas en el rango de baja inversión y baja facturación, con pocos casos de alta inversión o alta facturación.
        - Los tamaños de burbuja (proporcionales a la facturación) muestran que las campañas de mayor facturación están distribuidas entre todos los canales.
        
        **Conclusiones del análisis por canal:**
        
        1. **Diversificación equilibrada**: La empresa utiliza una estrategia multicanal equilibrada.
        2. **Eficiencia de Referral**: Este canal destaca en términos de ROI, lo que sugiere potencial para ampliar su uso.
        3. **Distribución similar**: No hay grandes diferencias en la relación inversión-facturación entre canales.
        """)

    with tab2:
        st.subheader("2. Análisis de Rendimiento y ROI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico 4: Distribución del ROI
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            plt.hist(df['roi_num'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
            plt.axvline(df['roi_num'].mean(), color='red', linestyle='--', label=f'Media: {df["roi_num"].mean():.2f}')
            plt.title('Distribución del ROI en las campañas')
            plt.xlabel('ROI')
            plt.ylabel('Número de campañas')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig4)
            
            st.markdown("""
            **Análisis de distribución del ROI:**
            
            - La distribución del ROI muestra una concentración principal entre 0.3 y 0.8, con una media de aproximadamente 0.54.
            - La forma relativamente uniforme de la distribución indica que no hay un grupo dominante de campañas extremadamente exitosas o fallidas.
            - Pocas campañas alcanzan un ROI superior a 0.9, lo que sugiere un área de oportunidad para optimización.
            - La línea roja punteada marca la media de ROI (0.54), que puede servir como punto de referencia para evaluar el rendimiento.
            """)
            
        with col2:
            # Gráfico 5: Tasa de conversión por tipo de cliente
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            conversion_by_audience = df.groupby('audiencia target')['ratio_conv_num'].mean().sort_values(ascending=False)
            sns.barplot(
                x=conversion_by_audience.index,
                y=conversion_by_audience.values,
                palette="viridis"
            )
            for i, v in enumerate(conversion_by_audience.values):
                ax5.text(i, v, f'{v:.3f}', ha='center', va='bottom')
            plt.title('Tasa de conversión según tipo de cliente')
            plt.xlabel('Tipo de cliente')
            plt.ylabel('Tasa de conversión promedio')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig5)
            
            st.markdown("""
            **Análisis de tasa de conversión por tipo de cliente:**
            
            - Los clientes B2B muestran la tasa de conversión más alta (0.551), seguidos de cerca por B2C (0.537).
            - La diferencia entre B2B y B2C es pequeña (aproximadamente 2.6%), lo que sugiere que ambos segmentos responden de manera similar a las campañas.
            - Las campañas sin datos de audiencia target tienen una tasa de conversión notablemente más baja (0.475), destacando la importancia de la segmentación.
            """)
        
        # Gráfico 6: ROI por canal y tipo de campaña
        st.subheader("ROI según Canal y Tipo de Campaña")
        
        # Filtrar tipos irrelevantes
        tipos_excluir = ['B2B', 'event', 'sin datos']
        df_filtrado = df[~df['tipo'].isin(tipos_excluir)]
        
        fig6 = px.box(df_filtrado, 
                     x='canal', 
                     y='roi_num', 
                     color='tipo',
                     title='Distribución del ROI según Canal y Tipo de Campaña',
                     labels={'canal': 'Canal de Marketing', 'roi_num': 'ROI', 'tipo': 'Tipo de Campaña'},
                     color_discrete_sequence=px.colors.qualitative.Vivid,
                     height=600)
        
        st.plotly_chart(fig6, use_container_width=True)
        
        st.markdown("""
        **Análisis de ROI por canal y tipo de campaña:**
        
        - **Email**: Destaca por su alto ROI, especialmente en los canales Organic y Referral donde alcanza valores de hasta 0.8.
        - **Social Media**: También muestra un buen desempeño en Organic y Referral.
        - **Webinar**: Ofrece resultados consistentes en todos los canales, particularmente en Paid.
        - **Podcast**: Tiende a tener el ROI más bajo y más consistente en todos los canales.
        
        **Variabilidad por canal:**
        
        - **Organic**: Muestra la mayor dispersión en ROI, lo que indica que los resultados pueden variar significativamente dependiendo del tipo de campaña y ejecución.
        - **Paid**: Ofrece resultados más predecibles, con menos variabilidad entre tipos de campaña.
        - **Referral**: Tiene el potencial de alcanzar los ROIs más altos, especialmente con Email y Social Media.
        
        **Conclusiones del análisis de rendimiento:**
        
        1. **ROI moderado con potencial de mejora**: El ROI promedio de 0.54 es positivo pero mejorable, con pocas campañas superando 0.9.
        2. **Combinaciones óptimas**: Email en Referral y Organic ofrecen el mayor ROI, seguido por Social Media en estos mismos canales.
        3. **Segmentación efectiva**: Tanto B2B como B2C muestran tasas de conversión similares, con una ligera ventaja para B2B.
        4. **Tipos de campaña diferenciales**: Email, Webinar y Social Media son consistentemente más efectivos que Podcast.
        """)

    with tab3:
        st.subheader("3. Análisis de Patrones Temporales")
        
        # Gráfico 7: Patrones estacionales
        st.markdown("### Patrones Estacionales en el Rendimiento")
        
        # Preparar datos para análisis estacional
        monthly_metrics = df.groupby('mes').agg({
            'roi_num': 'mean',
            'ratio_conv_num': 'mean',
            'facturación_num': 'mean'
        }).reset_index()
        
        # Crear gráfico con plotly
        fig7 = go.Figure()
        
        # ROI
        fig7.add_trace(go.Scatter(
            x=monthly_metrics['mes'],
            y=monthly_metrics['roi_num'],
            mode='lines+markers',
            name='ROI Promedio',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Tasa de conversión
        fig7.add_trace(go.Scatter(
            x=monthly_metrics['mes'],
            y=monthly_metrics['ratio_conv_num'],
            mode='lines+markers',
            name='Tasa de Conversión',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            marker=dict(size=8)
        ))
        
        # Facturación (eje secundario)
        fig7.add_trace(go.Scatter(
            x=monthly_metrics['mes'],
            y=monthly_metrics['facturación_num']/1000,
            mode='lines+markers',
            name='Facturación (miles €)',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        # Configurar layout
        fig7.update_layout(
            title='Patrones Estacionales en el Rendimiento de las Campañas',
            xaxis=dict(
                title='Mes',
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            ),
            yaxis=dict(title='ROI / Tasa de Conversión'),
            yaxis2=dict(
                title='Facturación (miles €)',
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            height=600
        )
        
        st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("""
        **Análisis de patrones estacionales:**
        
        - **Picos de rendimiento**: Todas las métricas (ROI, tasa de conversión y facturación) muestran picos claros en los meses 1, 3, 9 y 12 (enero, marzo, septiembre y diciembre).
        - **Período de bajo rendimiento**: El mes 7 (julio) presenta consistentemente los valores más bajos en todas las métricas.
        - **Patrones concordantes**: Las tres métricas siguen patrones muy similares a lo largo del año, lo que refuerza la identificación de temporadas altas y bajas.
        - **Magnitud de variación**: El ROI varía entre 0.48 y 0.58, la tasa de conversión entre 0.50 y 0.62, y la facturación entre 480 y 560 miles de euros.
        
        **Interpretación:**
        
        - Los picos en enero, marzo, septiembre y diciembre podrían estar relacionados con:
          - Enero: Campañas post-navideñas y de año nuevo.
          - Marzo: Campañas de primavera y cierre de primer trimestre.
          - Septiembre: Vuelta a la actividad tras vacaciones y preparación para último trimestre.
          - Diciembre: Campañas navideñas y cierre de año.
        
        - El bajón en julio probablemente se debe a la temporada vacacional, cuando la atención de los consumidores disminuye.
        """)
        
        # Gráfico 8: Relación duración-facturación
        st.markdown("### Relación entre Duración y Facturación")
        
        # Crear gráfico de hexbin
        fig8, ax8 = plt.subplots(figsize=(12, 6))
        hb = ax8.hexbin(
            df['duracion_num'],
            df['facturación_num'], 
            gridsize=30,
            cmap='viridis',
            mincnt=1
        )
        
        plt.colorbar(hb, ax=ax8, label='Número de campañas')
        plt.title('Relación entre duración y facturación de las campañas')
        plt.xlabel('Duración (días)')
        plt.ylabel('Facturación (€)')
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig8)
        
        st.markdown("""
        **Análisis de la relación duración-facturación:**
        
        - **Concentración principal**: La mayoría de las campañas se agrupan en una duración de 300-400 días con facturaciones entre 0.4-0.6 millones de euros (zonas más claras del hexbin).
        - **Ausencia de correlación fuerte**: No se observa una tendencia clara que indique que campañas más largas generen mayores ingresos.
        - **Campañas largas**: Las campañas con duraciones superiores a 500 días no muestran facturaciones proporcionalmente mayores, e incluso algunas presentan facturaciones más bajas (0.2-0.4 millones).
        - **Rango óptimo**: El rango de 300-400 días parece ser el más frecuente y efectivo, sugiriendo un posible "punto dulce" en la duración de las campañas.
        
        **Conclusiones sobre patrones temporales:**
        
        1. **Estacionalidad marcada**: Existe un claro patrón estacional con períodos de alto rendimiento (enero, marzo, septiembre, diciembre) y bajo rendimiento (julio).
        2. **Duración óptima**: Las campañas no necesariamente se benefician de duraciones muy largas; el rango 300-400 días concentra la mayor densidad de campañas con facturaciones moderadas.
        3. **Oportunidad de optimización**: Ajustar el calendario de campañas para intensificar esfuerzos en los meses más rentables y reducir inversión en los menos productivos.
        4. **Replanteamiento de duración**: Evitar extender campañas más allá de 400-500 días si no muestran resultados claros, ya que no hay evidencia de que generen mayor facturación.
        """)

    with tab4:
        st.subheader("4. Relaciones entre Variables y Análisis Multidimensional")
        
        # Gráfico 9: Relación inversión-ingresos (hexbin)
        st.markdown("### Relación entre Inversión e Ingresos")
        
        fig9, ax9 = plt.subplots(figsize=(12, 7))
        hb = ax9.hexbin(
            df['inversión_num'], 
            df['facturación_num'],
            gridsize=40, 
            cmap='viridis', 
            mincnt=1, 
            linewidths=0.5, 
            edgecolors='grey',
            alpha=0.8
        )
        
        cb = plt.colorbar(hb)
        cb.set_label('Número de campañas')
        
        plt.title('Relación entre inversión e ingresos (Hexbin)', fontsize=16)
        plt.xlabel('Inversión (€)', fontsize=13)
        plt.ylabel('Facturación (€)', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig9)
        
        st.markdown("""
        **Análisis de la relación inversión-ingresos:**
        
        - **Alta concentración en valores bajos**: La mayoría de las campañas se concentran en el rango de baja inversión (<0.2€) y baja facturación (<0.2€), como se evidencia por los colores más claros del hexbin.
        - **Dispersión limitada**: Hay pocas campañas que alcanzan niveles altos de inversión o facturación, lo que sugiere una estrategia predominantemente conservadora.
        - **Casos atípicos**: Se observan algunos casos de campañas con inversión alta pero facturación baja, que representan inversiones ineficientes.
        - **Baja escalabilidad**: Las campañas con baja inversión no muestran capacidad para escalar significativamente los ingresos, indicando posibles limitaciones en las estrategias de bajo presupuesto.
        
        **Interpretación:**
        
        - El predominio de campañas de baja inversión y baja facturación sugiere un enfoque de "muchas campañas pequeñas" en lugar de pocas campañas de gran escala.
        - La falta de una correlación clara entre inversión y facturación indica que otros factores (como canal, tipo de campaña, estacionalidad) pueden ser más determinantes del éxito que el presupuesto en sí.
        """)
        
        # Gráfico 10: Análisis multidimensional de campañas exitosas
        st.markdown("### Análisis Multidimensional de Campañas Exitosas")
        
        # Filtrar las campañas según criterios
        filtro = (df['roi_num'] > 0.5) & (df['facturación_num'] > 500000)
        campañas_filtradas = df[filtro]
        
        # Crear figura
        fig10 = px.scatter(
            campañas_filtradas,
            x='roi_num',
            y='facturación_num',
            size='beneficio_neto_num',
            color='ratio_conv_num',
            hover_name='nombre campaña',
            hover_data=['canal', 'tipo', 'duracion_num'],
            color_continuous_scale='plasma',
            size_max=50,
            title='Análisis multidimensional de campañas exitosas (ROI > 0.5 e ingresos > 500.000 €)',
            labels={
                'roi_num': 'ROI',
                'facturación_num': 'Facturación (€)',
                'beneficio_neto_num': 'Beneficio Neto',
                'ratio_conv_num': 'Tasa de Conversión'
            },
            height=600
        )
        
        fig10.update_layout(
            coloraxis_colorbar=dict(title='Tasa de Conversión'),
            legend_title_text='Beneficio Neto'
        )
        
        st.plotly_chart(fig10, use_container_width=True)
        
        st.markdown("""
        **Análisis multidimensional de campañas exitosas:**
        
        - **Distribución de ROI**: La mayoría de las campañas exitosas tienen un ROI entre 0.5 y 0.8, con pocas superando el umbral de 0.9.
        - **Rango de facturación**: La facturación de estas campañas se concentra principalmente entre 0.6 y 0.8 millones de euros.
        - **Relación con beneficio neto**: El tamaño de los puntos (proporcional al beneficio neto) indica que las campañas con mayor ROI tienden a generar mayor beneficio neto, aunque no necesariamente mayor facturación.
        - **Influencia de la tasa de conversión**: La escala de color (que representa la tasa de conversión) no muestra un patrón claro, lo que sugiere que campañas exitosas pueden tener diferentes tasas de conversión.
        
        **Conclusiones sobre relaciones entre variables:**
        
        1. **Concentración de campañas de bajo presupuesto**: La mayoría de las campañas operan con inversiones y facturaciones bajas, lo que sugiere una estrategia de diversificación en múltiples campañas pequeñas.
        2. **Eficiencia sobre volumen**: Las campañas más exitosas no son necesariamente las de mayor facturación, sino las que logran un equilibrio óptimo entre ROI y facturación.
        3. **Multifactorialidad del éxito**: No existe una única variable determinante del éxito; más bien, es la combinación de factores (canal, tipo, duración, estacionalidad) lo que marca la diferencia.
        4. **Oportunidad de optimización**: Existe margen para mejorar, especialmente considerando que pocas campañas superan un ROI de 0.9, lo que sugiere que se podrían implementar estrategias más eficientes.
        """)

# --- Insights y Recomendaciones ---
elif section == "Insights y Recomendaciones":
    st.header("Insights y Recomendaciones Estratégicas")
    
    st.markdown("""
    ## Hallazgos Clave y Recomendaciones Accionables
    
    Tras un análisis exhaustivo de los datos, hemos identificado insights valiosos que pueden traducirse en recomendaciones 
    estratégicas para optimizar las campañas de marketing y mejorar el retorno de inversión.
    """)
    
    # Crear tabs para los diferentes insights
    tabs = st.tabs([
        "Eficiencia de Canales", 
        "Optimización de Presupuesto", 
        "Mejora del ROI",
        "Duración Óptima",
        "Estacionalidad",
        "Tipos de Campaña",
        "Segmentación"
    ])
    
    with tabs[0]:
        st.subheader("1. EL CANAL REFERRAL ES EL MÁS EFICIENTE")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Mayor ROI promedio**: El canal Referral alcanza un ROI medio de 0.575, superior a los demás canales.
            - **Combinaciones destacadas**: Las campañas de tipo Email en Referral destacan con ROIs hasta 0.8.
            - **Rendimiento consistente**: Referral muestra buen desempeño con múltiples tipos de campaña (Email, Social Media).
            - **Potencial de expansión**: A pesar de su eficiencia, no es el canal más utilizado (segundo después de Promotion).
            
            **Análisis técnico:**
            
            El canal Referral aprovecha el poder de las recomendaciones, lo que explica su alta eficiencia. Las campañas de Email en este canal probablemente utilizan estrategias de referidos, donde clientes existentes recomiendan a nuevos prospectos, generando leads de mayor calidad a menor costo.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/share--v1.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            📈 Potenciar el canal Referral, especialmente con campañas de bajo coste y alta segmentación como Email Marketing.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Aumentar la asignación presupuestaria al canal Referral en un 15-20%.
        2. Implementar un programa formal de referidos con incentivos tanto para referentes como referidos.
        3. Desarrollar campañas de Email Marketing específicamente diseñadas para fomentar recomendaciones.
        4. Integrar elementos de social proof en las campañas de Referral para maximizar la credibilidad.
        5. Establecer KPIs específicos para medir el crecimiento y efectividad del canal Referral.
        """)
    
    with tabs[1]:
        st.subheader("2. ALTA CONCENTRACIÓN DE CAMPAÑAS POCO EFECTIVAS")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Concentración en valores bajos**: La mayoría de las campañas están en el rango de baja inversión (<0.2€) y baja facturación (<0.2€).
            - **Proliferación ineficiente**: El análisis hexbin muestra hasta 40 campañas en esta zona de bajo rendimiento.
            - **Falta de escalabilidad**: Estas campañas de bajo presupuesto no logran escalar significativamente los ingresos.
            - **Recursos fragmentados**: La estrategia actual dispersa recursos en múltiples pequeñas iniciativas en lugar de consolidarlos.
            
            **Análisis técnico:**
            
            La alta concentración de campañas pequeñas sugiere un enfoque fragmentado que puede estar diluyendo el impacto y dificultando la optimización. Las campañas con presupuestos muy limitados a menudo carecen de los recursos necesarios para testing, optimización y alcance suficiente.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/money-bag.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            💰 Reevaluar la estrategia de campañas de bajo presupuesto. Priorizar calidad sobre cantidad y experimentar con presupuestos medios optimizados.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Reducir el número total de campañas en un 30%, consolidando presupuestos en iniciativas más robustas.
        2. Establecer un umbral mínimo de inversión por campaña basado en el canal y tipo (por ejemplo, mínimo 0.3€ para campañas de Email, 0.5€ para Social Media).
        3. Implementar un proceso de aprobación de campañas que evalúe el potencial de escalabilidad antes de asignar presupuesto.
        4. Desarrollar un sistema de "test and scale" donde las campañas comiencen con presupuestos limitados pero tengan un camino claro para escalar si muestran buenos resultados iniciales.
        5. Auditar las campañas existentes de bajo presupuesto y terminar las que muestren ROI por debajo de 0.4.
        """)
    
    with tabs[2]:
        st.subheader("3. ROI PROMEDIO MODERADO PERO MEJORABLE")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **ROI medio de 0.54**: La mayoría de las campañas tienen un ROI entre 0.3 y 0.8, con una media de 0.54.
            - **Limitados casos de excelencia**: Pocas campañas superan un ROI de 0.9, lo que indica potencial sin explotar.
            - **Inversiones ineficientes**: Se identificaron casos de campañas con alta inversión pero ingresos no proporcionales.
            - **Variabilidad por canal y tipo**: El análisis muestra que ciertas combinaciones (como Email en Referral) consistentemente superan el ROI promedio.
            
            **Análisis técnico:**
            
            Un ROI promedio de 0.54 significa que por cada euro invertido, se recuperan 1.54€ (retorno del 54% sobre la inversión). Aunque positivo, este valor está por debajo de benchmarks de la industria para marketing digital, que suelen situarse entre 0.65-0.75 para campañas bien optimizadas.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            🎯 Establecer un ROI objetivo mínimo de 0.6. Auditar y ajustar campañas que estén por debajo de ese umbral.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Definir un ROI mínimo aceptable de 0.6 como KPI principal para todas las campañas nuevas.
        2. Implementar revisiones de medio término para todas las campañas activas, con ajustes obligatorios si el ROI está por debajo de 0.5 después de 2 meses.
        3. Crear un "playbook" basado en las campañas con mejor desempeño (ROI > 0.8) para replicar estrategias exitosas.
        4. Desarrollar un sistema de alertas tempranas que identifique campañas con tendencia negativa antes de que comprometan el ROI medio.
        5. Implementar testing A/B obligatorio para optimizar continuamente elementos clave (creatividades, copy, CTAs, landing pages) y mejorar las tasas de conversión.
        """)
    
    with tabs[3]:
        st.subheader("4. FACTURACIÓN Y DURACIÓN NO ESTÁN CORRELACIONADOS")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Ausencia de correlación positiva**: Las campañas largas (>500 días) no demuestran mayor facturación.
            - **Rango óptimo identificado**: La mayor concentración de campañas con ingresos moderados se da en el rango de 300-400 días.
            - **Ineficiencia en campañas prolongadas**: Algunas campañas con duraciones superiores a 500 días muestran facturaciones más bajas (0.2-0.4 millones).
            - **Evidencia cuantitativa**: El análisis hexbin muestra claramente que la densidad de campañas exitosas se concentra en duraciones medias.
            
            **Análisis técnico:**
            
            La falta de correlación entre duración y facturación sugiere que extender campañas no garantiza mejores resultados, y puede incluso llevar a ineficiencias por fatiga de audiencia, obsolescencia creativa o cambios en el mercado. El "punto dulce" en 300-400 días probablemente representa un equilibrio óptimo entre construcción de momentum y frescura de la campaña.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/time.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            🕒 Optimizar la duración de campañas. Evitar extender campañas sin resultados claros más allá de 400 días.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Establecer una duración estándar objetivo de 300-400 días para nuevas campañas, con extensiones condicionadas a métricas de rendimiento.
        2. Implementar revisiones obligatorias a los 300 días, con criterios específicos para justificar la continuación (ROI creciente o estable, engagement sostenido).
        3. Desarrollar un protocolo de "refresh creativo" para campañas que superen los 250 días, para evitar fatiga de audiencia.
        4. Segmentar análisis de duración por tipo de campaña y canal, ya que el punto óptimo puede variar (por ejemplo, las campañas de Email podrían tener duraciones óptimas diferentes a Social Media).
        5. Para campañas existentes que superen los 400 días, realizar una auditoría inmediata y finalizar aquellas con tendencia de ROI decreciente.
        """)
    
    with tabs[4]:
        st.subheader("5. PATRÓN ESTACIONAL CLARO EN EL RENDIMIENTO")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Picos consistentes**: ROI, conversión y facturación mejoran notablemente en los meses 1, 3, 9 y 12 (enero, marzo, septiembre y diciembre).
            - **Punto bajo identificado**: El mes 7 (julio) es consistentemente el menos productivo en todas las métricas.
            - **Magnitud significativa**: La diferencia entre picos y valles representa aproximadamente un 20% en ROI y tasa de conversión.
            - **Patrón replicable**: La concordancia entre diferentes métricas refuerza la validez del patrón estacional detectado.
            
            **Análisis técnico:**
            
            Los patrones estacionales identificados se alinean con ciclos de negocio y comportamiento de consumo típicos: inicios de año (resoluciones), cierres de trimestre (presupuestos corporativos), regreso de vacaciones (renovación de actividad) y temporada navideña (mayor gasto). El valle en julio coincide con el período vacacional cuando la atención a mensajes de marketing disminuye.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/calendar--v1.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            📆 Ajustar el calendario de campañas para intensificar esfuerzos en los meses más rentables y reducir inversión en los menos eficaces.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Incrementar el presupuesto de marketing en un 20-30% durante los meses pico (enero, marzo, septiembre, diciembre).
        2. Reducir la inversión en julio en un 15-20%, manteniendo solo campañas estratégicas o de mantenimiento de marca.
        3. Planificar el lanzamiento de nuevas campañas para que coincidan con los inicios de los períodos de alto rendimiento.
        4. Desarrollar creatividades y mensajes específicos para cada período estacional, aprovechando los factores contextuales.
        5. Implementar un sistema de previsión y planificación que incorpore estos patrones estacionales en la distribución anual del presupuesto.
        """)
    
    with tabs[5]:
        st.subheader("6. CAMPAÑAS TIPO EMAIL, WEBINAR Y SOCIAL MEDIA SON LAS MÁS RENTABLES")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Eficiencia de Email**: Las campañas de Email destacan por su alto ROI (hasta 0.8 en canales como Referral) y bajo coste relativo.
            - **Buen desempeño de Webinar**: Consistentemente muestra resultados positivos en todos los canales, especialmente en Paid.
            - **Potencial de Social Media**: Ofrece buenos resultados particularmente en Organic y Referral.
            - **Bajo rendimiento de Podcast**: Presenta el ROI más bajo y menor estabilidad entre tipos de campaña.
            
            **Análisis técnico:**
            
            La superioridad del Email Marketing puede atribuirse a su bajo coste de implementación, alta personalización, facilidad de segmentación y medición precisa. Los Webinars combinan educación con venta directa, generando leads cualificados. Social Media se beneficia de su alcance y engagement. El bajo rendimiento de Podcast podría deberse a dificultades en la atribución y conversión desde un medio principalmente pasivo.
            """)
        
        with col2:
            st.image("https://img.icons8.com/fluency/96/megaphone.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            🛠️ Priorizar tipos de campaña que combinan bajo coste y buenos resultados. Revisar si Podcast justifica su continuidad.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Incrementar la proporción de presupuesto destinado a campañas de Email en un 20%, especialmente en el canal Referral.
        2. Desarrollar un calendario anual de Webinars que coincida con los períodos de alto rendimiento identificados.
        3. Optimizar las campañas de Social Media con enfoque en contenido compartible para aprovechar el efecto de red.
        4. Reducir inversión en Podcast en un 30% o reorientar completamente la estrategia, midiendo con KPIs específicos para justificar continuidad.
        5. Implementar un sistema de scoring que priorice automáticamente combinaciones de tipo-canal según su ROI histórico para la asignación de presupuesto.
        """)
    
    with tabs[6]:
        st.subheader("7. TIPO DE CLIENTE NO AFECTA SIGNIFICATIVAMENTE A LA CONVERSIÓN")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Hallazgos:**
            
            - **Diferencia marginal**: Clientes B2B convierten ligeramente mejor (0.551) que B2C (0.537), una diferencia de solo 2.6%.
            - **Consistencia entre segmentos**: Ambos tipos de audiencia responden de manera similar a las campañas.
            - **Importancia de la segmentación**: Las campañas sin datos de audiencia target presentan tasas de conversión significativamente más bajas (0.475).
            - **Oportunidad de personalización**: A pesar de tasas de conversión similares, las motivaciones y mensajes efectivos pueden diferir entre B2B y B2C.
            
            **Análisis técnico:**
            
            La similitud en tasas de conversión sugiere que, aunque los ciclos de compra y procesos de decisión difieren entre B2B y B2C, la efectividad de las campañas bien ejecutadas es comparable. La caída de conversión en campañas "sin datos" subraya la importancia de conocer y segmentar la audiencia, independientemente del tipo específico.
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/96/search--v1.png", width=96)
            
            st.success("""
            **RECOMENDACIÓN:**
            
            🔍 Asegurar una buena segmentación del público objetivo. Personalizar campañas según perfil (B2B/B2C) para maximizar conversiones.
            """)
            
        st.markdown("""
        **Acciones específicas:**
        
        1. Implementar un protocolo de clasificación de audiencia que asegure que todas las campañas tengan un target definido (eliminar la categoría "sin datos").
        2. Mantener estrategias diferenciadas para B2B y B2C en términos de contenido y mensaje, aunque compartan canales y formatos.
        3. Desarrollar buyer personas específicas para cada segmento, actualizándolas trimestralmente según datos de comportamiento.
        4. Realizar tests A/B específicos para cada segmento, identificando elementos que impacten la conversión particular de cada grupo.
        5. Implementar un sistema de lead scoring que considere comportamientos específicos de B2B y B2C para optimizar el nurturing y conversión.
        """)
    
    # Resumen final
    st.markdown("""
    ## Síntesis Final: Plan de Acción Prioritario
    
    Basándonos en todos los insights extraídos, recomendamos implementar las siguientes acciones prioritarias:
    
    1. **Potenciar el canal Referral con Email Marketing** - Representa la combinación más eficiente en términos de ROI.
    2. **Consolidar campañas pequeñas en iniciativas más robustas** - Reducir fragmentación y aumentar el impacto.
    3. **Implementar un calendario estacional optimizado** - Ajustar presupuestos según los patrones temporales identificados.
    4. **Establecer duraciones estándar de 300-400 días** - Optimizar el ciclo de vida de las campañas.
    5. **Desarrollar un sistema de evaluación continua** - Con el objetivo de mantener un ROI mínimo de 0.6.
    
    La implementación coordinada de estas recomendaciones tiene el potencial de aumentar el ROI promedio en un 15-20% 
    y mejorar significativamente la eficiencia global de la estrategia de marketing digital.
    """)

# Footer
st.markdown("---")
st.markdown("**Proyecto desarrollado para Upgrade Hub por Carla Molina - 2025**")