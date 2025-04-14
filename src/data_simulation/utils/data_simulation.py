import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataSimulation():
    def __init__(self):
        self.num_samples = 0
        self.num_features = 0
        self.noise_level = 0.0
        self.data = None

    def set_parameters(self, num_samples, num_features, noise_level):
        # insert specification of hyperparameters here that are relevant for the data simulation
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise_level = noise_level

    def generate_data(self):
        # insert code for generating data here
        data = None

        self.data = data
        return data

    def generate_figures(self, filename: str):
        # insert code for relevant plots here
        return

    def save_data(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)