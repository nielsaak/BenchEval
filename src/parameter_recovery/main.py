import os
import pandas as pd
from utils.parameter_recovery import ParameterRecovery

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/stan_fits")
    os.makedirs(output_path_figures, exist_ok=True)
    # os.makedirs(output_path_data, exist_ok=True)

    # Load data
    data_path = os.path.join(current_dir, "../data_simulation/output/results/data.pkl")

    # Model file path
    model_file = os.path.join(current_dir, "../models/recovery/hierarchical_centered_recovery.stan")

    # Create an instance of ParameterRecovery
    parameter_recovery = ParameterRecovery()

    # Load the data
    data = ParameterRecovery.load_data(data_path = data_path)

    # Run parameter recovery
    parameter_recovery.recover_parameters(data = data,
                                          stan_file = model_file,
                                          output_path_data=output_path_data,
                                          output_path_figures=output_path_figures,
                                          )
    

    print("Parameter recovery completed.")