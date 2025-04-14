"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.12
"""

import os
import mlflow
from pycaret.classification import setup, create_model, finalize_model, save_model, predict_model
from sklearn.metrics import log_loss, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def train_models(train_df):
    """
    1) Faz setup do PyCaret.
    2) Treina Logistic Regression e Decision Tree.
    3) Finaliza cada modelo.
    4) (Opcional) Salva manualmente com PyCaret e mlflow.log_artifact(),
       mas também retornamos os objetos Python para o Kedro salvá-los via PickleDataSet.
    """
    setup(
        data=train_df,
        target="shot_made_flag",
        session_id=42,
        verbose=False
    )

    # Logistic Regression
    lr = create_model("lr")
    lr_final = finalize_model(lr)

    # Podemos salvar via PyCaret + MLflow se quiser:
    save_model(lr_final, "lr_model")
    mlflow.log_artifact("lr_model.pkl")

    # Decision Tree
    dt = create_model("dt")
    dt_final = finalize_model(dt)

    save_model(dt_final, "dt_model")
    mlflow.log_artifact("dt_model.pkl")

    # Retornamos os objetos em Python, que o Kedro depois vai salvar em disco
    # nos outputs: ["trained_lr_model", "trained_dt_model"]
    return lr_final, dt_final


def evaluate_models(lr_model, dt_model, test_df, raw_score=True):
    """
    Recebe 2 modelos e a base de teste.
    Calcula métricas e salva best_model como pkl.
    """
    preds_lr = predict_model(lr_model, data=test_df)
    preds_dt = predict_model(dt_model, data=test_df)

    # A coluna de score é 'Score' quando raw_score=False e 'Score_1' quando raw_score=True
    score_col = "Score" if not raw_score else "Score_1"

    ll_lr = log_loss(preds_lr["shot_made_flag"], preds_lr["prediction_score"])
    f1_lr = f1_score(preds_lr["shot_made_flag"], preds_lr["prediction_label"])
    mlflow.log_metric("lr_log_loss", ll_lr)
    mlflow.log_metric("lr_f1", f1_lr)

    ll_dt = log_loss(preds_dt["shot_made_flag"], preds_dt["prediction_score"])
    f1_dt = f1_score(preds_dt["shot_made_flag"], preds_dt["prediction_label"])
    mlflow.log_metric("dt_log_loss", ll_dt)
    mlflow.log_metric("dt_f1", f1_dt)

    if ll_lr < ll_dt:
        best = lr_model
        chosen = "LogisticRegression"
    else:
        best = dt_model
        chosen = "DecisionTree"

    mlflow.log_param("chosen_model", chosen)

    # Salvar best_model via PyCaret + MLflow
    save_model(best, "best_model")
    mlflow.log_artifact("best_model.pkl")

    # Retornar o objeto Python do melhor modelo, que Kedro salvará num .pkl
    return best

