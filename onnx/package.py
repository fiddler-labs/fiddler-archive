import pathlib
import pickle
import sys

import pandas as pd
import yaml
import logging

import onnxruntime as rt
import numpy


class Model:
    THIS_DIR = pathlib.Path(__file__).parent
    MODEL_FILEPATH = THIS_DIR / 'rf_iris.onnx'
    MODEL_YAML = THIS_DIR / 'model.yaml'

    def __init__(self):
        self.sess = rt.InferenceSession(self.MODEL_FILEPATH.as_posix())
        with self.MODEL_YAML.open('r') as yaml_file:
            model_info = yaml.load(yaml_file, Loader=yaml.FullLoader)['model']
        self.task = model_info['model-task']
        self.pred_column_names = ['predicted_value']

    def predict(self, input_df):
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        np_array = input_df.to_numpy()
        np_array = np_array.astype(numpy.float32)
        pred_onx = self.sess.run([label_name], {input_name: np_array})[0]
        return pd.DataFrame(pred_onx, columns=self.pred_column_names)

def get_model():
    return Model()
