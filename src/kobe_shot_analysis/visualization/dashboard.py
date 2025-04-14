"""Dashboard Streamlit para monitoramento do modelo."""

import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient

def load_metrics():
    """Carrega métricas do MLflow."""
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.mlflow.runName IN ('PipelineAplicacao', 'MonitoramentoModelo')"
    )
    return runs

def main():
    """Função principal do dashboard."""
    st.title("Monitoramento do Modelo de Arremessos do Kobe")

    # Carrega métricas
    runs = load_metrics()

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

    # Gráficos
    st.subheader("Distribuição das Predições")

    # Carrega dados de produção
    predictions = pd.read_parquet("data/03_primary/production_predictions.parquet")

    # Gráfico de distribuição das predições
    fig = px.histogram(
        predictions,
        x="Score",
        title="Distribuição das Probabilidades de Acerto"
    )
    st.plotly_chart(fig)

    # Gráfico de distribuição por período
    fig = px.box(
        predictions,
        x="period",
        y="Score",
        title="Distribuição das Probabilidades por Período"
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

    # Informações adicionais
    st.subheader("Informações do Modelo")

    # Carrega informações do modelo
    client = MlflowClient()
    model_versions = client.search_model_versions("name='kobe_shot_predictor'")

    st.write(f"Versão atual do modelo: {model_versions[0].version}")
    st.write(f"Última atualização: {model_versions[0].current_stage}")

if __name__ == "__main__":
    main()