from utils.prediction import Prediction
import os

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")

    # Load data
    data_path = "data/results.processed (1).jsonl"
    rank_path = "data/european_all_simplified.csv"

    # Model file path
    model_file = os.path.join(current_dir, "../models/prediction/hierarchical_centered_predictions.stan")

    # Create an instance of Prediction
    prediction = Prediction(model_file=model_file,
                            output_path_figures=output_path_figures,
                            output_path_data=output_path_data)

    # Load data for prediction
    prediction.load_data(data_path=data_path)

    # Preprocess data
    prediction.preprocess_data(top_n=100,
                                top_n_path=rank_path,
                                test_index_n=10)
    
    # Generate data to loop through
    prediction.prepare_data_for_stan()

    # Run prediction
    # prediction.make_predictions(stan_file=model_file,
    #                                          output_path_data=os.path.join(output_path_data, "thesis"),
    #                                          output_path_figures=os.path.join(output_path_figures, "thesis"),
    #                                          model_fit_params={"chains": 1,
    #                                                  "iter_sampling": 1000,
    #                                                  "iter_warmup": 500,
    #                                                 #  "adapt_delta": 0.95,
    #                                                  })

    prediction.baseline_predictions(output_path_data=os.path.join(output_path_data, "baseline"),
                                     output_path_figures=os.path.join(output_path_figures, "baseline"),)

    prediction.comparison(output_path_data=output_path_data)

    print("Prediction completed.")