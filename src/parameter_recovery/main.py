import os
import pandas as pd
from utils.parameter_recovery import ParameterRecovery

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")

    # Load data
    data_path = os.path.join(current_dir, "../data_simulation/output/synthetic_data.csv")
    data = pd.read_csv(data_path)

    # Model file path
    model_file = os.path.join(current_dir, "../models/stan_model.stan")

    # Parameter names to recover
    params = ["param1", "param2"]

    # Create an instance of ParameterRecovery
    parameter_recovery = ParameterRecovery(data=data,
                                           model_file=model_file,
                                          params=params)

    # Run parameter recovery
    parameter_recovery.recover_parameters(model_fit_params={"chains": 4,
                                                            "parallel_chains": 4,
                                                            "iter_sampling": 1000,
                                                            "iter_warmup": 500,
                                                            "seed": 42},
                                          output_path_data=output_path_data,
                                          output_path_figures=output_path_figures)
    

    print("Parameter recovery completed.")