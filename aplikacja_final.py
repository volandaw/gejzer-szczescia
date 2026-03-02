import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.regression import load_model, predict_model
import os
import pathlib
folder_aplikacji = pathlib.Path(__file__).parent.resolve()

# 1. Konfiguracja i Ładowanie zasobów
st.set_page_config(page_title="Gejzer Szczęścia by VolandJ", layout="wide")
model = load_model(os.path.join(folder_aplikacji, 'moj_finalny_model_szczescia2'))
df_hist = pd.read_csv('world-happiness-report1.csv')

# 2. Opis aplikacji na panelu bocznym
st.sidebar.title("🚀 O aplikacji")
st.sidebar.info(
    """
    **Gejzer Szczęścia** to interaktywne narzędzie analityczne 
    łączące historyczne dane World Happiness Report (2006-2023) 
    z zaawansowaną prognozą AI. 
    
    Aplikacja pozwala na symulowanie różnych scenariuszy 
    ekonomiczno-społecznych dla wybranych krajów i obserwowanie 
    ich wpływu na przewidywany poziom szczęścia do roku 2028.
    """
)

# 3. Wybór krajów
st.title("📈 Panel Symulacji Szczęścia Narodu")
wszystkie_kraje = sorted(df_hist['Country name'].unique())
wybrane_kraje = st.multiselect(
    "Wybierz kraje do analizy porównawczej:", 
    options=wszystkie_kraje, 
    default=['Poland', 'Switzerland']
)

# 4. Dynamiczne suwaki i zbieranie parametrów
parametry_krajow = {}
if wybrane_kraje:
    st.sidebar.header("⚙️ Parametry wzrostu dla państw")
    for kraj in wybrane_kraje:
        with st.sidebar.expander(f"📍 Symulacja: {kraj}", expanded=False):
            gdp_inc = st.slider(f"Roczna zmiana PKB", -0.5, 1.0, 0.2, key=f"g_{kraj}")
            soc_inc = st.slider(f"Zmiana wsparcia społ.", -0.05, 0.05, 0.01, key=f"s_{kraj}")
            parametry_krajow[kraj] = {'gdp': gdp_inc, 'soc': soc_inc}

    # 5. Silnik obliczeniowy i generowanie wykresu
    plot_data_list = []
    for kraj in wybrane_kraje:
        # Historia
        k_hist = df_hist[df_hist['Country name'] == kraj].copy()
        if 'Happiness Score' not in k_hist.columns and 'Life Ladder' in k_hist.columns:
            k_hist = k_hist.rename(columns={'Life Ladder': 'Happiness Score'})
        k_hist['Typ'] = 'Historia'
        plot_data_list.append(k_hist[['Country name', 'Year', 'Happiness Score', 'Typ']])

        # Prognoza
        ostatni = k_hist.iloc[-1:].copy()
        p_gdp = parametry_krajow[kraj]['gdp']
        p_soc = parametry_krajow[kraj]['soc']

        for rok in [2026, 2027, 2028]:
            dane_sim = ostatni.copy()
            dane_sim['Year'] = rok
            roznica_lat = rok - ostatni['Year'].iloc[0]
            dane_sim['GDP'] += p_gdp * roznica_lat
            dane_sim['Social support'] += p_soc * roznica_lat
            
            pred = predict_model(model, data=dane_sim)
            
            plot_data_list.append(pd.DataFrame({
                'Country name': [kraj], 'Year': [rok],
                'Happiness Score': [pred['prediction_label'].iloc[0]],
                'Typ': 'Prognoza ML'
            }))

    if plot_data_list:
        df_final = pd.concat(plot_data_list)
        fig = px.line(df_final, x='Year', y='Happiness Score', color='Country name',
                      line_dash='Typ', markers=True, height=600,
                      template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# 6. Stopka autorska
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888;'>
        <p style='font-size: 1.2em;'>Autor opracowania: <b>VolandJ</b></p>
        <p>Silnik AI: Extra Trees Regressor | Technologia: Python, PyCaret, Streamlit</p>
        <p>© 2026 Projekty AI - World Happiness Analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)