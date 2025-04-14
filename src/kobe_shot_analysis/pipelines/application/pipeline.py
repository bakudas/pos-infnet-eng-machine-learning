"""
This is a boilerplate pipeline 'application'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.get_predictions_production,
            inputs=["treinamento_logistical_regression", "raw_train_prod"],
            outputs="predictions",
            tags=["predictions"]
        ),
        node(
            nodes.get_metrics,
            inputs=["treinamento_logistical_regression", "raw_train_prod"],
            outputs="predictions_metrics",
            tags=["metrics"]
        ),
        node(
            nodes.generate_roc_auc_plot,
            inputs=["treinamento_logistical_regression", "raw_train_prod"],
            outputs="roc_auc_plot",
            tags=["metrics"]
        ),
    ])