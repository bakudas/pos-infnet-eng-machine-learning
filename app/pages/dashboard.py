"""Dashboard Streamlit para monitoramento do modelo."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient

def show_all_runs():
    """Mostra todas as runs disponíveis e suas tags."""
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])

    st.subheader("Runs Disponíveis")

    if not runs:
        st.warning("Nenhuma run encontrada no experimento.")
        return

    # Cria um DataFrame com informações das runs
    runs_data = []
    for run in runs:
        run_info = {
            'Run ID': run.info.run_id,
            'Nome': run.data.tags.get('mlflow.runName', 'Sem nome'),
            'Status': run.info.status,
            'Data de Início': run.info.start_time,
            'Tags': str(run.data.tags)
        }
        runs_data.append(run_info)

    runs_df = pd.DataFrame(runs_data)
    st.dataframe(runs_df)

def load_metrics():
    """Carrega métricas do MLflow."""
    client = MlflowClient()

    # Busca runs do PipelineAplicacao
    runs_pipeline = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.mlflow.runName = 'PipelineAplicacao'"
    )

    # Busca runs do MonitoramentoModelo
    runs_monitor = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.mlflow.runName = 'MonitoramentoModelo'"
    )

    # Combina os resultados
    all_runs = list(runs_pipeline) + list(runs_monitor)
    return all_runs

def main():
    """Função principal do dashboard."""
    st.title("Monitoramento do Modelo de Arremessos do Kobe")

    # Mostra todas as runs disponíveis
    show_all_runs()

    # Carrega métricas
    runs = load_metrics()

    # Verifica se há runs disponíveis
    if not runs:
        st.warning("""
        ⚠️ Nenhum dado de monitoramento disponível no momento.

        Isso pode acontecer por alguns motivos:
        1. O pipeline de aplicação ainda não foi executado
        2. O monitoramento ainda não foi executado
        3. Os runs não estão com as tags corretas

        Por favor, execute o pipeline de aplicação e o monitoramento para ver os dados aqui.
        """)
        return

    # Sidebar
    st.sidebar.title("Configurações")
    selected_metric = st.sidebar.selectbox(
        "Selecione a métrica",
        ["log_loss", "f1_score"]
    )

    # Métricas principais
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Log Loss",
            f"{runs[0].data.metrics['log_loss']:.4f}"
        )

    with col2:
        st.metric(
            "F1 Score",
            f"{runs[0].data.metrics['f1_score']:.4f}"
        )

    # Carrega dados de produção
    try:
        predictions = pd.read_parquet("data/03_primary/production_predictions.parquet")
    except FileNotFoundError:
        st.warning("""
        ⚠️ Arquivo de predições não encontrado.

        O arquivo `data/03_primary/production_predictions.parquet` não existe.
        Execute o pipeline de aplicação para gerar as predições.
        """)
        return

    # Gráficos de distribuição das features
    st.subheader("Análise das Features")

    # Mapa de calor das probabilidades por localização
    fig = px.scatter(
        predictions,
        x="lon",
        y="lat",
        color="Score",
        title="Mapa de Calor das Probabilidades por Localização",
        labels={"lon": "Longitude", "lat": "Latitude", "Score": "Probabilidade de Acerto"}
    )
    st.plotly_chart(fig)

    # Distribuição das probabilidades por distância
    fig = px.box(
        predictions,
        x="shot_distance",
        y="Score",
        title="Distribuição das Probabilidades por Distância do Arremesso"
    )
    st.plotly_chart(fig)

    # Distribuição das probabilidades por período
    fig = px.box(
        predictions,
        x="period",
        y="Score",
        title="Distribuição das Probabilidades por Período"
    )
    st.plotly_chart(fig)

    # Comparação entre playoffs e temporada regular
    fig = px.box(
        predictions,
        x="playoffs",
        y="Score",
        title="Distribuição das Probabilidades: Playoffs vs Temporada Regular",
        labels={"playoffs": "Playoffs", "Score": "Probabilidade de Acerto"}
    )
    st.plotly_chart(fig)

    # Distribuição das probabilidades por minutos restantes
    fig = px.scatter(
        predictions,
        x="minutes_remaining",
        y="Score",
        title="Probabilidades por Minutos Restantes",
        labels={"minutes_remaining": "Minutos Restantes", "Score": "Probabilidade de Acerto"}
    )
    st.plotly_chart(fig)

    # Alertas
    st.subheader("Alertas")

    # Verifica drift nas predições
    mean_confidence = predictions['Score'].mean()
    std_confidence = predictions['Score'].std()

    if mean_confidence < 0.4 or mean_confidence > 0.6:
        st.warning("⚠️ Drift detectado na distribuição das predições")

    if std_confidence > 0.3:
        st.warning("⚠️ Alta variabilidade nas predições")

    # Verifica distribuição das features
    if predictions['shot_distance'].mean() > 25:
        st.warning("⚠️ Alta média de distância dos arremessos")

    if predictions['minutes_remaining'].mean() < 3:
        st.warning("⚠️ Baixa média de minutos restantes nos arremessos")

    # Informações adicionais
    st.subheader("Informações do Modelo")

    # Carrega informações do modelo
    client = MlflowClient()
    model_versions = client.search_model_versions("name='kobe_shot_predictor'")

    st.write(f"Versão atual do modelo: {model_versions[0].version}")
    st.write(f"Última atualização: {model_versions[0].current_stage}")

if __name__ == "__main__":
    main()