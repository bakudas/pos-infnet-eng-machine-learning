# src/kobe_shots/pipelines/data_processing/nodes.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

def process_data(df: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    # 1) Remover nulos e selecionar colunas
    df = df.dropna(subset=["lat", "lon", "minutes_remaining",
                           "period", "playoffs", "shot_distance", "shot_made_flag"])

    cols = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df = df[cols]
    mlflow.log_param("columns_used", cols)
    mlflow.log_metric("n_rows_filtered", len(df))

    # 2) Split estratificado
    X = df.drop("shot_made_flag", axis=1)
    y = df["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size_rows", len(X_test))

    train_df = X_train.copy()
    train_df["shot_made_flag"] = y_train
    test_df = X_test.copy()
    test_df["shot_made_flag"] = y_test

    return train_df, test_df