import pathlib
import pickle
import sys

import pandas as pd
import yaml
import logging


class Model:
    THIS_DIR = pathlib.Path(__file__).parent
    MODEL_FILEPATH = THIS_DIR / 'model.pkl'
    MODEL_YAML = THIS_DIR / 'model.yaml'

    def __init__(self):
        self.model = None
        with self.MODEL_YAML.open('r') as yaml_file:
            model_info = yaml.load(yaml_file, Loader=yaml.FullLoader)['model']
        self.task = model_info['model-task']
        self.pred_column_names = [
            output['column-name'] for output in model_info['outputs']
        ]
        with self.MODEL_FILEPATH.open('rb') as serialized_model:
            self.model = pickle.load(serialized_model)

    def predict(self, input_df):
        if self.task == 'multiclass_classification':
            predict_fn = self.model.predict_proba
        elif self.task == 'binary_classification':
            def predict_fn(x):
                return self.model.predict_proba(x)[:, 1]
        else:
            predict_fn = self.model.predict
        return pd.DataFrame(predict_fn(input_df), columns=self.pred_column_names)

def get_model():
    return Model()
