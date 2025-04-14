"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.12
"""

import os
import mlflow
from pycaret.classification import load_model, predict_model, setup, create_model, finalize_model, save_model, predict_model
import pandas as pd
from sklearn.metrics import log_loss, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def apply_model(prod_df: pd.DataFrame, dropna: bool = True):
    """
    Carrega o modelo (do disco ou MLflow), aplica ao dataset de produção
    e registra métricas se tiver 'shot_made_flag'.

    Não chamamos start_run() aqui, pois kedro-mlflow já mantém a run ativa.
    """
    # Remover nulos
    if dropna:
        prod_df = prod_df.dropna(
            subset=["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
        )

    # Carregar modelo final salvo por train_models
    model = load_model("data/06_models/best_model_kedro")


    # Fazer predições
    scored_df = predict_model(model, data=prod_df)

    # Se tiver a variável de resposta, registrar métricas
    if "shot_made_flag" in scored_df.columns:
        y_true = scored_df["shot_made_flag"]
        y_pred = scored_df["prediction_label"]
        y_proba = scored_df["prediction_score"]

        ll = log_loss(y_true, y_proba)
        f1 = f1_score(y_true, y_pred)

        mlflow.log_metric("log_loss_prod", ll)
        mlflow.log_metric("f1_prod", f1)

    return scored_df

def plot_inference_reports(scored_df, reports_dir: str = "data/08_reporting"):
    """
    Gera plots de relatório (matriz de confusão, ROC) a partir do DataFrame scored_df,
    que deve conter:
      - 'shot_made_flag' (rótulo verdadeiro) [opcional, só gera gráficos se existir]
      - 'prediction_label' (classe predita, 0/1)
      - 'prediction_score' (probabilidade da classe positiva).

    Salva os gráficos em 'reports_dir'. Retorna uma string indicando status.
    """
    # Garantir que a pasta exista
    os.makedirs(reports_dir, exist_ok=True)

    # Se não houver rótulo ('shot_made_flag'), não podemos plotar Confusion Matrix / ROC
    if "shot_made_flag" not in scored_df.columns:
        print("Não há rótulo 'shot_made_flag' na base de produção. Não é possível gerar gráficos supervisionados.")
        return "No label available"

    y_true = scored_df["shot_made_flag"]

    # Verificar se temos 'Label' (classe predita)
    if "prediction_label" not in scored_df.columns:
        print("Coluna 'Label' não encontrada em scored_df. Não é possível gerar Confusion Matrix.")
        return "No predictions"

    y_pred = scored_df["prediction_label"]

    y_proba = scored_df["prediction_score"]  # se usar raw_score=True e classe 1

    # 1) Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion Matrix (Inference)")
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
    plt.close()

    # 2) Curva ROC, se tivermos probabilidades
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_value = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
        plt.plot([0,1], [0,1], '--')
        plt.title("ROC Curve (Inference)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(reports_dir, "roc_curve.png"))
        plt.close()
    else:
        print("Sem probabilidade para gerar ROC.")

    print(f"Plots salvos em {reports_dir}")
    return "Plots generated"