import os
from utils.model_development import ModelDevelopment

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output_posterior/figures")
    output_path_data = os.path.join(current_dir, "output_posterior/results")

    # Load data
    data_path = os.path.join(current_dir, "../data_simulation/output/results/data.pkl")

    # Model file path
    model_file = "src/models/model_development/hierarchical_centered_model_development.stan"

    # Create an instance of ParameterEstimation
    parameter_estimation = ModelDevelopment(model_file=model_file,
                                               output_path_figures=output_path_figures,
                                               output_path_data=output_path_data)

    # Load data
    print("data_path:", data_path) 
    data = parameter_estimation.load_data(data_path=data_path)

    # # Run parameter estimation
    parameter_estimation.estimate_parameters(data=data,
                                             stan_file=model_file,
                                             output_path_data=output_path_data,
                                             output_path_figures=output_path_figures,
                                             model_fit_params={"chains": 4,
                                                     "iter_sampling": 2000,
                                                     "iter_warmup": 1000,
                                                     "seed": 123,
                                                     "adapt_delta": 0.95,
                                                     })
    
    
    # Produce summary plots
    parameter_estimation.summary_plots(data, type = "posterior",)

    print("Model development completed.")