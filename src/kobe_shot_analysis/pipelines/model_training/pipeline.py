
"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node
from .nodes import train_models, evaluate_models

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_models,
            inputs="base_train",
            outputs=["trained_lr_model", "trained_dt_model"],
            name="train_models_node",
        ),
        node(
            func=evaluate_models,
            inputs=["trained_lr_model", "trained_dt_model", "base_test"],
            outputs="best_model",
            name="evaluate_models_node"
        ),
    ])