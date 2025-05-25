import os
from utils.data_description import DataDescription

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figures = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")
    os.makedirs(output_path_figures, exist_ok=True)
    os.makedirs(output_path_data, exist_ok=True)

    # Load data
    data_path = "data/results.processed (1).jsonl"
    rank_path = "data/european_all_simplified.csv"

    # Create an instance of DataDescription
    data_description = DataDescription(output_path_figures=output_path_figures,
                                        output_path_data=output_path_data)

    # Load data
    data_description.load_data(data_path=data_path)

    # Preprocess data
    data_description.preprocess_data(top_n=100,
                                     top_n_path=rank_path)

    # Save language histograms
    data_description.plot_histograms(group_by="language")

    # Save task histograms
    data_description.plot_histograms(group_by="task")

    # Save model histograms
    data_description.plot_histograms_model_counts()

    print("Data description completed.")