"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
import kobe_shot_analysis.pipelines.data_processing as data_processing
from kobe_shot_analysis.pipelines.model_training.pipeline import create_pipeline as create_model_training_pipeline
import kobe_shot_analysis.pipelines.application as application


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    #pipelines = find_pipelines()
    #pipelines["__default__"] = sum(pipelines.values())
    #return pipelines

    data_processing_pipeline = data_processing.create_pipeline()
    model_training_pipeline = create_model_training_pipeline()
    application_pipeline = application.create_pipeline()


    return {
        "__default__": data_processing_pipeline + model_training_pipeline + application_pipeline,
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
        "application": application_pipeline,
    }