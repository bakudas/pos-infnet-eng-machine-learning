import streamlit as st
import requests


def call_inference(data):
    """
    Faz uma chamada para o endpoint de inferência do modelo
    """
    rows = [
        list(data.values())
    ]

    resp = requests.post(
        'http://localhost:5001/invocations',
        json={
            'inputs': rows,
        }
    )

    inference = resp.json()
    return inference['predictions'][0]


st.markdown("""
# Análise de Arremessos do Kobe Bryant

Esta página permite fazer inferências sobre a probabilidade de sucesso de um arremesso baseado em suas características.
""")

col1, col2 = st.columns(2)

# Características do arremesso
with col1:
    st.subheader("Localização do Arremesso")
    lat = st.slider("Latitude", -90.0, 90.0, 0.0, 0.1)
    lon = st.slider("Longitude", -180.0, 180.0, 0.0, 0.1)
    shot_distance = st.slider("Distância do Arremesso (pés)", 0, 40, 15)

with col2:
    st.subheader("Contexto do Jogo")
    period = st.selectbox("Período", [1, 2, 3, 4, 5, 6, 7])
    minutes_remaining = st.slider("Minutos Restantes", 0, 12, 6)
    playoffs = st.checkbox("É um jogo de playoffs?")

# Preparando os dados para inferência
input_data = {
    'lat': lat,
    'lon': lon,
    'minutes_remaining': minutes_remaining,
    'period': period,
    'playoffs': int(playoffs),
    'shot_distance': shot_distance
}

# Mostrando os dados inseridos
st.subheader("Dados do Arremesso")
st.json(input_data)

# Fazendo a inferência
if st.button("Calcular Probabilidade de Sucesso"):
    probability = call_inference(input_data)

    # Criando uma barra de progresso para visualizar a probabilidade
    st.progress(probability)

    # Mostrando o resultado
    st.markdown(f"""
    ### Resultado da Análise

    **Probabilidade de Sucesso: {probability:.1%}**

    {'🎯 Alta probabilidade de sucesso!' if probability > 0.6 else '⚠️ Baixa probabilidade de sucesso.'}
    """)