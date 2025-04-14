"""
This is a boilerplate pipeline 'application'
generated using Kedro 0.19.12
"""

import pandas as pd
from ..utils.utils import get_model_metrics, plot_auc_roc

def get_predictions_production(model, data):
    """
    This function is used to get predictions from the model.
    :param model: The model to be used for prediction.
    :param data: The data to be used for prediction.
    :return: The predictions made by the model.
    """
    columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]
    data = pd.DataFrame(data, columns=columns)
    predictions = model.predict(data)
    data["predictions"] = predictions
    return data

def get_metrics(model, data):
    columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    data = pd.DataFrame(data, columns=columns)
    data.dropna(inplace=True)
    return get_model_metrics(model, data)

def generate_roc_auc_plot(model, data):
    columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    data = pd.DataFrame(data, columns=columns)
    data.dropna(inplace=True)
    y_true = data["shot_made_flag"]
    data = data.drop(columns=["shot_made_flag"])
    y_proba = model.predict_proba(data)[:, 1]
    return plot_auc_roc(y_true, y_proba)