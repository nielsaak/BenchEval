import os
from utils.parameter_estimation import ParameterEstimation

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")

    # Load data
    data_path = os.path.join(current_dir, "../../data/results.jsonl")

    # Model file path
    model_file = os.path.join(current_dir, "../models/stan_model.stan")

    # Create an instance of ParameterEstimation
    parameter_estimation = ParameterEstimation(model_file=model_file,
                                               output_path_figures=output_path_figures,
                                               output_path_data=output_path_data)

    # Load data
    parameter_estimation.load_data(data_path=data_path)

    # Preprocess data
    parameter_estimation.preprocess_data()

    # Run parameter estimation
    parameter_estimation.estimate_parameters(output_path_data=output_path_data,
                                             output_path_figures=output_path_figures)

    print("Parameter estimation completed.")