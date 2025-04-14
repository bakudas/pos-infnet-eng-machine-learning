
"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
            nodes.treinamento,
            inputs=["params:model_lr", "base_train", "params:session_id"],
            outputs="treinamento_logistical_regression",
            tags=["treinamento"]
        ),
        node(
            nodes.treinamento,
            inputs=["params:model_dt", "base_train", "params:session_id"],
            outputs="treinamento_decision_tree",
            tags=["treinamento"]
        ),
        node(
            nodes.get_metrics,
            inputs=["treinamento_logistical_regression", "base_test"],
            outputs="base_test_lr_metrics",
            tags=["metrics"]
        ),
        node(
            nodes.get_metrics,
            inputs=["treinamento_decision_tree", "base_test"],
            outputs="base_test_dt_metrics",
            tags=["metrics"]
        )
    ])
