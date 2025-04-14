"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.12
"""

import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
import streamlit as st
from pycaret.classification import predict_model, setup

def evaluate_production_data(production_data: pd.DataFrame, model: tuple, params: dict = None) -> tuple[pd.DataFrame, dict]:
    """Avalia o modelo nos dados de produção.

    Args:
        production_data (pd.DataFrame): Dados de produção.
        model (tuple): Modelo treinado e métricas.
        params (dict, optional): Parâmetros de avaliação.

    Returns:
        tuple: DataFrame com predições e dicionário com métricas
    """
    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Extrai o modelo
        trained_model, _ = model

        # Configura ambiente PyCaret
        setup(
            data=production_data,
            target='shot_made_flag',
            silent=True
        )

        # Faz predições
        predictions = predict_model(trained_model, data=production_data)

        # Calcula métricas
        metrics = {
            "log_loss": log_loss(
                production_data['shot_made_flag'],
                predictions['Score']
            ),
            "f1_score": f1_score(
                production_data['shot_made_flag'],
                predictions['Label']
            )
        }

        # Registra métricas no MLflow
        mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)

        # Salva as predições
        predictions.to_parquet("data/03_primary/production_predictions.parquet")

        return predictions, metrics

def deploy_model(model: tuple) -> tuple:
    """Deploy do modelo usando MLflow.

    Args:
        model (tuple): Modelo treinado e métricas.

    Returns:
        tuple: Modelo deployado e URI.
    """
    with mlflow.start_run(run_name="DeploymentModelo"):
        # Extrai o modelo
        trained_model, metrics = model

        # Registra o modelo no MLflow
        mlflow.sklearn.log_model(
            trained_model,
            "model",
            registered_model_name="kobe_shot_predictor"
        )

        # Obtém a URI do modelo
        model_uri = mlflow.get_artifact_uri("model")

        return trained_model, model_uri

def monitor_model(production_predictions: pd.DataFrame, params: dict = None) -> dict:
    """Monitora a saúde do modelo em produção.

    Args:
        production_predictions (pd.DataFrame): Predições do modelo em produção.
        params (dict, optional): Parâmetros de monitoramento.

    Returns:
        dict: Métricas de monitoramento.
    """
    with mlflow.start_run(run_name="MonitoramentoModelo"):
        # Calcula métricas de monitoramento
        monitoring_metrics = {
            "prediction_distribution": production_predictions['Label'].value_counts().to_dict(),
            "confidence_distribution": {
                "mean": production_predictions['Score'].mean(),
                "std": production_predictions['Score'].std()
            }
        }

        # Registra métricas no MLflow
        mlflow.log_metrics(monitoring_metrics)
        if params:
            mlflow.log_params(params)

        return monitoring_metrics
