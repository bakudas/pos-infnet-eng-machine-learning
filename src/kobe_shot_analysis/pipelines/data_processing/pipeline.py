"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node
from .nodes import process_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=process_data,
            inputs="dataset_kobe_dev",
            outputs=["base_train", "base_test"],
            name="process_data_node"
        )
    ])