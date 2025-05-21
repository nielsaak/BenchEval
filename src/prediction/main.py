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
    model_file = os.path.join(current_dir, "src/models/final/stan_model.stan")

    # Create an instance of Prediction
    prediction = Prediction(model_file=model_file,
                            output_path_figures=output_path_figures,
                            output_path_data=output_path_data)

    # Load data for prediction
    prediction.load_data(data_path=data_path)

    # Preprocess data for prediction
    prediction.preprocess_data()

    # Run prediction
    prediction.make_predictions(output_path_data=output_path_data,
                                 output_path_figures=output_path_figures)

    print("Prediction completed.")