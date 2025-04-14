import os
from utils.data_simulation import DataSimulation

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figure = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")

    # Create an instance of DataSimulation
    data_simulation = DataSimulation()

    # Set parameters for data simulation
    data_simulation.set_parameters(num_samples=1000, num_features=10, noise_level=0.1)

    # Generate synthetic data
    data = data_simulation.generate_data()

    # Generate figures
    data_simulation.generate_figures(output_path_figure)

    # Save the generated data to a CSV file
    data_simulation.save_data(data, os.path.join(output_path_data, "synthetic_data.csv"))

    print("Synthetic data and plots generated and saved")