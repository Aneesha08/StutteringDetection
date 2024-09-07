import glob
import os
import numpy as np
from tensorflow import keras


class Files:

    def __init__(self, root_folder: str = ""):
        self.root_folder = root_folder


    def predict_with_model(self, input_vectors):
        model = keras.models.load_model(self.root_folder)
        prediction = model.predict(input_vectors)
        binary_predictions = np.round(prediction).flatten()
        return binary_predictions