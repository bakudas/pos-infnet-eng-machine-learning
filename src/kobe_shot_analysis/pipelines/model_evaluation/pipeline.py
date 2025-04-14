"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node
from .nodes import evaluate_production_data, deploy_model, monitor_model

def create_pipeline(**kwargs) -> Pipeline:
    """Cria o pipeline de avaliação e monitoramento do modelo."""
    return Pipeline(
        [
            node(
                func=evaluate_production_data,
                inputs=["production_data", "best_model", "params:model_evaluation"],
                outputs=["production_predictions", "production_metrics"],
                name="evaluate_production_node",
                tags=["model_evaluation"]
            ),
            node(
                func=deploy_model,
                inputs="best_model",
                outputs=None,
                name="deploy_model_node",
                tags=["model_evaluation"]
            ),
            node(
                func=monitor_model,
                inputs=["production_predictions", "params:model_evaluation"],
                outputs="monitoring_metrics",
                name="monitor_model_node",
                tags=["model_evaluation"]
            ),
        ],
        tags=["model_evaluation"]
    )
