"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.prepare_dataset,
            inputs=["raw_train_dev"],
            outputs="dataset_filtered",
            tags=["prepare"]
        ),
        node(
            nodes.split_dataset,
            inputs=["dataset_filtered", "params:session_id", "params:test_size"],
            outputs=["base_train", "base_test", "split_train_size", "split_test_size"],
            tags=["prepare"]
        )
    ])