"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node
from .nodes import apply_model, plot_inference_reports

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=apply_model,
                inputs="dataset_kobe_prod",
                outputs="predictions",
                name="apply_model_node",
            ),
            node(
            func=plot_inference_reports,
            inputs="predictions",
            outputs="report_status",
            name="plot_inference_reports_node",
        ),
        ]
    )
