"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset(raw_train_dev) -> pd.DataFrame:
    raw_train_dev.dropna(inplace=True)
    raw_train_dev = raw_train_dev[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']]
    return raw_train_dev

def split_dataset(dataset_filtered, session_id, test_size) -> pd.DataFrame:
    train, test = train_test_split(dataset_filtered, test_size=test_size, random_state=session_id, stratify=dataset_filtered['shot_made_flag'])

    return train, test, len(train), len(test)