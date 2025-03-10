import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import folium
from lifelines import KaplanMeierFitter
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString, MultiPoint, Point
import streamlit.components.v1 as components

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8", sep=",")
    return df

def calcular_porcentagem_sobrevida(df, campo_drs, n_drs, incluir_in_situ=True):
    """
    Calcula a porcentagem de sobrevida para um dado DRS (identificado pela coluna nDRS)
    usando o Kaplan-Meier. Considera como óbito os casos em que ULTINFO é 3 ou 4.
    """
    df_drs = df[df[campo_drs] == n_drs]
    n = df_drs.shape[0]
    
    # Calcula as proporções de estadiamento, se as categorias existirem
    try:
        if incluir_in_situ:
            in_situ = df_drs.ECGRUP_CAT.value_counts(normalize=True).sort_index()['In Situ']
        else:
            in_situ = None
    except KeyError:
        in_situ = None
    try:
        ini = df_drs.ECGRUP_CAT.value_counts(normalize=True).sort_index()['Inicial']
    except KeyError:
        ini = None
    try:
        avanc = df_drs.ECGRUP_CAT.value_counts(normalize=True).sort_index()['Avançado']
    except KeyError:
        avanc = None

    # Define o evento: ULTINFO em [3, 4] indica óbito
    E = df_drs["ULTINFO"].isin([3, 4])
    T = df_drs.meses_diag

    kmf = KaplanMeierFitter()
    surv_curve = kmf.fit(T, E).survival_function_
    try:
        sobrevida_36_meses = surv_curve.loc[36]['KM_estimate']
    except KeyError:
        sobrevida_36_meses = 0
    try:
        sobrevida_60_meses = surv_curve.loc[60]['KM_estimate']
    except KeyError:
        sobrevida_60_meses = 0

    return {
        'nDRS': n_drs,
        'Total': n,
        'In Situ': in_situ,
        'Inicial': ini,
        'Avançado': avanc,
        'sobrevida_36_meses': sobrevida_36_meses,
        'sobrevida_60_meses': sobrevida_60_meses,
    }

def geometrycollection_to_multipolygon(geometry_collection):
    if geometry_collection.geom_type == "GeometryCollection":
        polygons = []
        for geom in geometry_collection.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygons.extend(geom.geoms)
            elif isinstance(geom, (LineString, MultiLineString, Point, MultiPoint)):
                polygons.append(geom.buffer(0.01))
        geometry_collection = MultiPolygon(polygons)
    return geometry_collection

# =============================================================================
# TÍTULO E CARREGAMENTO DOS DADOS
# =============================================================================

st.title("Câncer de Colo do Útero")

csv_path = r"D:\FOSP\colo_utero1.csv"
df_original = load_data(csv_path)
df_original["Ano"] = df_original["ANODIAG"]

# Conversões e mapeamentos
df_original["ULTINFO"] = pd.to_numeric(df_original["ULTINFO"], errors="coerce")
staging_map = {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "4": 4}
df_original["ECGRUP"] = df_original["ECGRUP"].astype(str)
df_original["ECGRUP_NUM"] = df_original["ECGRUP"].map(staging_map)

# =============================================================================
# BARRA LATERAL – FILTROS
# =============================================================================

st.sidebar.header("Filtros")
df = df_original.copy()  # Trabalha com uma cópia para aplicar os filtros

# ---------- Filtro por Faixa Etária ----------
min_idade = int(df["IDADE"].min())
max_idade = int(df["IDADE"].max())
idade_range = st.sidebar.slider(
    "Selecione a faixa etária",
    min_idade,
    max_idade,
    (min_idade, max_idade)
)
df = df[df["IDADE"].between(idade_range[0], idade_range[1])]

# ---------- Filtro de Estadiamento (ECGRUP_NUM) ----------
valores_ecgrup = [0, 1, 2, 3, 4]
ecgrup_labels = [str(x) for x in valores_ecgrup]
opcoes_ecgrup = ["Todas"] + ecgrup_labels

selected_ecgrup_raw = st.sidebar.segmented_control(
    "Grupo de Estadiamento (multi)",
    opcoes_ecgrup,
    selection_mode="multi",
    default=["Todas"]
)

if "Todas" in selected_ecgrup_raw:
    selected_ecgrup = valores_ecgrup
else:
    selected_ecgrup = [int(x) for x in selected_ecgrup_raw]

df = df[df["ECGRUP_NUM"].isin(selected_ecgrup)]

# ---------- Filtro ULTINFO ----------
ultinfo_map = {
    1: "VIVO, COM CÂNCER",
    2: "VIVO",
    3: "ÓBITO POR CÂNCER",
    4: "ÓBITO"
}
valores_ultinfo = list(ultinfo_map.keys())
labels_ultinfo = [ultinfo_map[k] for k in valores_ultinfo]
opcoes_ultinfo = ["Todas"] + labels_ultinfo

selected_ultinfo_raw = st.sidebar.segmented_control(
    "Resultado do Paciente (ULTINFO)",
    opcoes_ultinfo,
    selection_mode="multi",
    default=["Todas"]
)

if "Todas" in selected_ultinfo_raw:
    selected_ultinfo = valores_ultinfo
else:
    selected_ultinfo = []
    for label in selected_ultinfo_raw:
        for k, v in ultinfo_map.items():
            if v == label:
                selected_ultinfo.append(k)

df = df[df["ULTINFO"].isin(selected_ultinfo)]

# ---------- Filtro utilizando a coluna nDRS (número da DRS) ----------
df["nDRS"] = pd.to_numeric(df["nDRS"], errors="coerce")
df = df.dropna(subset=["nDRS"])
df["nDRS"] = df["nDRS"].astype(int)

# =============================================================================
# MÉTRICA E GRÁFICOS (USANDO O DATAFRAME FILTRADO)
# =============================================================================

# Exibir o contador de pacientes filtrados (usa o df já filtrado)
#total_pacientes = len(df)
#st.metric(label="Total de Pacientes Filtrados", value=total_pacientes)

# Gráfico 1: Tendência Geral de Diagnósticos por Ano
st.subheader("Tendência Geral de Diagnósticos por Ano") 
trend_df = df.groupby("Ano").size().reset_index(name="Número de Diagnósticos").sort_values("Ano")
fig_trend = px.line(trend_df, x="Ano", y="Número de Diagnósticos", markers=True)
st.plotly_chart(fig_trend)

# Gráfico 2: Casos por Ano – Resultado do Paciente (ULTINFO)
st.subheader("Casos por Ano")
ultinfo_trend = df.groupby(["Ano", "ULTINFO"]).size().reset_index(name="Contagem")
ultinfo_trend = ultinfo_trend.sort_values("Ano")
ultinfo_trend["Resultado"] = ultinfo_trend["ULTINFO"].map(ultinfo_map)
fig_ultinfo = px.bar(
    ultinfo_trend,
    x="Ano",
    y="Contagem",
    color="Resultado",
    barmode="group"
)
st.plotly_chart(fig_ultinfo)

# =============================================================================
# MAPA INTERATIVO COM FOLIUM
# =============================================================================

st.header("Sobrevida por DRS")

# Se a coluna 'meses_diag' não existir, cria-a (exemplo: utilizando ULTIDIAG)
if "meses_diag" not in df.columns and "ULTIDIAG" in df.columns:
    df["meses_diag"] = (df.ULTIDIAG / 30).round()
    df.loc[df["meses_diag"] > 60, "meses_diag"] = 61

# Carrega o shapefile/KML das DRS – use raw string para evitar problemas com as barras invertidas
kml_path = r"D:\FOSP\sp_drs_group_interactive.kml"
df_drs = gpd.read_file(kml_path)
drs_shapefile = df_drs.copy()
if 'Description' in drs_shapefile.columns:
    drs_shapefile.drop(columns=['Description'], inplace=True)
    
# Adiciona os nomes e a numeração das DRS
NOME_DRS = ["Grande São Paulo", "Araçatuba", "Araraquara", "Baixada Santista", "Barretos", 
            "Bauru", "Campinas", "Franca", "Marília", "Piracicaba", "Presidente Prudente", 
            "Registro", "Ribeirão Preto", "São João da Boa Vista", "São José do Rio Preto", 
            "Sorocaba", "Taubaté"]
drs_shapefile['Nome'] = NOME_DRS
if "Name" in drs_shapefile.columns:
    drs_shapefile.drop(columns=['Name'], inplace=True)
# Atribui o número de cada DRS na coluna nDRS
NO_DRS = list(range(1, len(NOME_DRS) + 1))
drs_shapefile["nDRS"] = NO_DRS

# Converte as geometrias, se necessário
drs_shapefile['geometry'] = drs_shapefile['geometry'].apply(geometrycollection_to_multipolygon)
drs_shapefile = gpd.GeoDataFrame(drs_shapefile, geometry='geometry')

# Cálculo das métricas de sobrevida para cada DRS (utilizando a coluna nDRS)
list_ndrs = np.sort(df.nDRS.unique())
results = []
for n_drs in list_ndrs:
    new_data = calcular_porcentagem_sobrevida(df.copy(), 'nDRS', n_drs)
    results.append(new_data)
results_df = pd.DataFrame(results)

# Merge dos resultados com o shapefile, utilizando a coluna nDRS
drs_shapefile = drs_shapefile.merge(results_df, on="nDRS", how="left")
drs_shapefile['sobrevida_36_meses_escrita'] = drs_shapefile['sobrevida_36_meses'].apply(lambda x: f"sobrevida: {round(x, 2):.2f}")
drs_shapefile['nDRS_escrita'] = drs_shapefile['nDRS'].apply(lambda x: f"DRS{x}")

# Converte o GeoDataFrame para GeoJSON
geojson = drs_shapefile.to_json()

# Cria o mapa interativo com Folium
current_tile = 'cartodbpositron'
m = folium.Map(location=[-23.5505, -46.6333], zoom_start=6, tiles=current_tile)
choropleth = folium.Choropleth(
    geo_data=geojson,
    name='choropleth',
    data=drs_shapefile,
    columns=['nDRS', 'sobrevida_36_meses'],
    key_on='feature.properties.nDRS',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name='Sobrevida em 36 meses',
    highlight=True,
    line_color='black'
).add_to(m)

# Adiciona tooltips com o nome da DRS e a sobrevida formatada
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['Nome', 'sobrevida_36_meses_escrita'], labels=False)
)

# Adiciona marcadores com o número da DRS nos centróides das regiões
for index, row in drs_shapefile.iterrows():
    centroid = row['geometry'].centroid
    marker = folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(
            html=f"""<div style="font-weight: bold; font-size: 12px; color: black;">{row['nDRS_escrita']}</div>"""
        )
    )
    marker.add_to(m)

# Renderiza o mapa no dashboard
components.html(m._repr_html_(), height=600)