"""
This module contains utility functions for the Kobe pipeline.
"""

from sklearn.metrics import log_loss, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def get_model_metrics(model, dataset):
    """

    """
    X_test = dataset.drop(columns=['shot_made_flag'])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    log_loss_value = log_loss(dataset['shot_made_flag'].values, y_proba)
    f1_value = f1_score(dataset['shot_made_flag'].values, y_pred)

    metrics = {
        "log_loss": log_loss_value,
        "f1_score": f1_value
    }

    return {
        key: {'value': val, 'step': 1}
        for key, val in metrics.items()
    }

def plot_auc_roc(y_true, y_proba, title="AUC-ROC Curve"):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    ax.grid(True)
    return fig