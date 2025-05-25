import os
from utils.data_simulation import DataSimulation

if __name__ == "__main__":
    # Create path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path_figure = os.path.join(current_dir, "output/figures")
    output_path_data = os.path.join(current_dir, "output/results")
    os.makedirs(output_path_data, exist_ok=True)
    os.makedirs(output_path_figure, exist_ok=True)

    # Create an instance of DataSimulation
    data_simulation = DataSimulation()

    # Set parameters for data simulation
    data_simulation.set_parameters(L=10, K_1=12, K_2=7, n_repetitions=10, n_samples=200)

    # Plot the parameters
    data_simulation.plot_parameters(output_path_figure)

    # # Generate synthetic data
    data_simulation.simulate_from_params(output_path_figure=output_path_figure)

    # # Save the generated data to a CSV file
    data_simulation.save_data(os.path.join(output_path_data))

    print("Synthetic data generated and saved")