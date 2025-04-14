"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.12
"""
from pycaret.classification import ClassificationExperiment
from ..utils.utils import get_model_metrics

def treinamento_best_model(base_train, session_id) -> ClassificationExperiment:
    exp = ClassificationExperiment()
    exp.setup(data=base_train, target='shot_made_flag', session_id=session_id, use_gpu=True)
    best_model = exp.compare_models(sort='f1')
    return best_model

def treinamento(model_name, base_train, session_id) -> ClassificationExperiment:
    """
    Train model and register in MLFlow
    Args:
        model_name (str): Name of the model to train ('dt', 'lr').
        base_train (pd.DataFrame): Training dataset.
        session_id (int): Session ID for PyCaret.
    """
    exp = ClassificationExperiment()
    exp.setup(data=base_train, target='shot_made_flag', session_id=session_id, use_gpu=True)

    model = exp.create_model(model_name)

    exp.tune_model(model, n_iter=10, optimize='f1')
    return model

def get_metrics(model, dataset):
    return get_model_metrics(model, dataset)

