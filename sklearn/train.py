import pandas as pd
import numpy as np
import pickle
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

import fiddler as fdl


def train():
  df = pd.read_csv('dataset.csv')
  print(df)

  target = 'quality'
  df[target] = df[target].astype(float)

  y = df[target] 
  X = df.drop(target, axis=1)
  print(X)

  input_features = list(X.columns.values)

  # There is only one dataset for this example. So we are splitting it here
  # User can also upload multiple files/sources to dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  regressor=LinearRegression()
  regressor.fit(X_train,y_train)

  print("model score: %.3f" % regressor.score(X_test, y_test))

  pickle.dump(regressor, open( "model.pkl", "wb" ) )
  write_schema(df, target, input_features)

def write_schema(df, target, input_features):
  # write dataset schema
  dataset_info = fdl.DatasetInfo.from_dataframe(df)
  dataset_info.display_name = 'sklearn_wine_ds'
  dataset_info.dataset_id = 'sklearn_wind_ds'
  dataset_info.files = ['dataset.csv']
  with open('dataset.yaml', 'w') as output:
     output.write(yaml.dump({'dataset': dataset_info.to_dict()}))

  print(dataset_info)

  # write model schema
  outputs = ['predicted_quality']

  model_info = fdl.ModelInfo.from_dataset_info(
      dataset_info=dataset_info,
      features = input_features,
      target=target, 
      decision_cols=[],
      input_type=fdl.ModelInputType.TABULAR,
      model_task=fdl.ModelTask.REGRESSION,
      display_name='Model for predicting wine quality',
      description='This is a Regression model using FAR',
  )
  with open('model.yaml', 'w') as output:
     output.write(yaml.dump({'model': model_info.to_dict()}))

if __name__ == "__main__":
    train()
