import pandas as pd
import numpy as np
import pickle
import yaml

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import fiddler as fdl


def train():
  df = pd.read_csv('dataset.csv')
  df = df.drop('row_id', axis=1)
  
  target = 'species'
  
  y = df[target] 
  X = df.drop(target, axis=1)

  input_features = list(X.columns.values)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  clr = RandomForestClassifier()
  clr.fit(X_train, y_train)

  # Convert into ONNX format
  initial_type = [('float_input', FloatTensorType([None, 4]))]
  onx = convert_sklearn(clr, initial_types=initial_type)
  with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

  write_schema(df, target, input_features)

def write_schema(df, target, input_features):
  # write dataset schema
  dataset_info = fdl.DatasetInfo.from_dataframe(df)
  dataset_info.display_name = 'onnx_ds'
  dataset_info.dataset_id = 'onnx_ds'
  dataset_info.files = ['dataset.csv']
  with open('dataset.yaml', 'w') as output:
     output.write(yaml.dump({'dataset': dataset_info.to_dict()}))

  print(dataset_info)

  # write model schema
  outputs = ['setosa', 'versicolor', 'virginica']

  model_info = fdl.ModelInfo.from_dataset_info(
      dataset_info=dataset_info,
      features = input_features,
      outputs = outputs,
      target=target, 
      decision_cols=[],
      input_type=fdl.ModelInputType.TABULAR,
      model_task=fdl.ModelTask.MULTICLASS_CLASSIFICATION,
      categorical_target_class_details=[0,1,2],
      display_name='Model for predicting iris species',
      description='This is a classification model using FAR',
  )
  with open('model.yaml', 'w') as output:
     output.write(yaml.dump({'model': model_info.to_dict()}))

if __name__ == "__main__":
    train()
