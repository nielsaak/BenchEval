import os
from utils.parameter_estimation import ParameterEstimation
import argparse

# python src/parameter_estimation/main.py --output_path_data output/results/model_centered --model_file src/models/final/simple_model_hierarchical_21052025_centered.stan

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    # output_path_data = os.path.join(current_dir, "output/results/model_50")

    parser = argparse.ArgumentParser(description="Parameter Estimation Paths")
    parser.add_argument("--output_path_data", type=str, required=True,
                        help="Output path for data results")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the Stan model file")
    args = parser.parse_args()

    output_path_data = os.path.join(current_dir, args.output_path_data)

    # Load data
    data_path = "data/results.processed (1).jsonl"
    rank_path = "data/european_all_simplified.csv"

    # Model file path
    model_file = args.model_file

    # Create an instance of ParameterEstimation
    parameter_estimation = ParameterEstimation(model_file=model_file,
                                               output_path_figures=output_path_figures,
                                               output_path_data=output_path_data)

    # Load data
    parameter_estimation.load_data(data_path=data_path)

    # Preprocess data
    parameter_estimation.preprocess_data(top_n=50,
                                         top_n_path=rank_path,
                                         test_index_n=5)

    # Save data histograms
    parameter_estimation.data_description_plots()

    # Prepare data for Stan
    parameter_estimation.prepare_data_for_stan()

    # Run parameter estimation
    parameter_estimation.estimate_parameters(stan_file=model_file,
                                             output_path_data=output_path_data,
                                             output_path_figures=output_path_figures,
                                             model_fit_params={"chains": 4,
                                                     "iter_sampling": 2000,
                                                     "iter_warmup": 1000,
                                                     "adapt_delta": 0.95,})
    
    # Produce summary plots
    parameter_estimation.summary_plots()

    print("Parameter estimation completed.")