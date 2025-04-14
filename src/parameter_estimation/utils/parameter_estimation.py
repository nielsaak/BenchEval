import json
import pandas as pd

class ParameterEstimation():
    """
    A class for parameter estimation using Stan.
    """

    def __init__(self, model_file, output_path_figures, output_path_data):
        """
        Initialize the ParameterEstimation class.

        :param model_file: Path to the Stan model file.
        :param data_path: Path to the data file.
        :param output_path_figures: Path to save output figures.
        :param output_path_data: Path to save output data.
        """
        self.model_file = model_file
        self.output_path_figures = output_path_figures
        self.output_path_data = output_path_data

    def load_data(self, data_path):
        """
        Load data from the specified path.
        """

        # Open the file and read JSON lines
        with open(data_path, 'r') as file:
            records = [json.loads(line) for line in file if line.strip()]

        rows = []
        for entry in records:
            try:
                # Get the list of test results
                raw_tests = entry.get('results', {}).get('raw', {}).get('test', [])
                # Capture the base information for the record excluding 'results'
                base_info = {k: v for k, v in entry.items() if k != 'results'}
                # For each test result and each metric in that result, create a row.
                for idx, test_result in enumerate(raw_tests):
                    for metric_name, metric_value in test_result.items():
                        row = base_info.copy()
                        row["test_index"] = idx + 1 # For showing what tries go together
                        row["metric"] = metric_name
                        row["value"] = metric_value
                        rows.append(row)
            except:
                try:
                    # Get the list of test results
                    raw_tests = entry.get('results', {}).get('raw', {})
                    # Capture the base information for the record excluding 'results'
                    base_info = {k: v for k, v in entry.items() if k != 'results'}
                    # For each test result and each metric in that result, create a row.
                    for idx, test_result in enumerate(raw_tests):
                        for metric_name, metric_value in test_result.items():
                            row = base_info.copy()
                            row["test_index"] = idx + 1
                            row["metric"] = metric_name
                            row["value"] = metric_value
                            rows.append(row)
                except Exception as e:
                    print(f"Error processing entry: {entry}")
                    print(e)

        return pd.DataFrame(rows)

    def preprocess_data(self):
        """
        Preprocess the loaded data.
        """
        # Implement data preprocessing logic here
        pass
    
    def estimate_parameters(self, output_path_data, output_path_figures):
        """
        Estimate parameters using the Stan model.

        :param output_path_data: Path to save output data.
        :param output_path_figures: Path to save output figures.
        """
        # Implement parameter estimation logic here
        pass
        # Example of using PyStan or CmdStanPy for parameter estimation
        # import pystan
        # import cmdstanpy
        # model = pystan.StanModel(file=self.model_file)
        # data = self.load_data()
        # fit = model.sampling(data=data, iter=1000, chains=4)
        # fit_summary = fit.summary()
        # fit_summary.to_csv(os.path.join(output_path_data, "parameter_estimates.csv"))
        # fit.plot()
        # plt.savefig(os.path.join(output_path_figures, "parameter_estimates.png"))
        # plt.close()
        # Save the results
        # fit.save(os.path.join(output_path_data, "stan_fit.pkl"))
        # Save figures
        # plt.savefig(os.path.join(output_path_figures, "traceplot.png"))
        # plt.close()

    