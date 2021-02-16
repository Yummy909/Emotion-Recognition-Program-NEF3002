import sys
import pathlib

path = pathlib.Path().absolute()

training_file = str(path) + '/features/'
save_file = str(path) + '/joblib_features/'
model_file = str(path) + '/model/'
example_file = str(path) + '/examples/'