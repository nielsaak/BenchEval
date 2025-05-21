import pandas as pd
from cmdstanpy import CmdStanModel
import os
import arviz as az
import matplotlib.pyplot as plt

class ParameterRecovery():
    DEFAULT_PARAMS = [
        "mu_beta_language",
        "sigma_beta_language",
        "beta_language",
        "mu_beta_task",
        "sigma_beta_task",
        "beta_task",
        "beta_metric",
        "mu_alpha",
        "sigma_alpha",
        "alpha",
        "beta_metric_phi",
        "beta_task_phi"
    ]

    def __init__(self, 
                 data_path: str, 
                 model_file: str, 
                 params: list = None,
                 group_cols: list = None):
        self.data_path = data_path
        self.data_path_no_ext = os.path.splitext(data_path)[0]
        self.model_file = model_file
        self.params = params if params else self.DEFAULT_PARAMS
        self.group_cols = group_cols if group_cols else self.DEFAULT_PARAMS
        self.data = pd.read_csv(data_path)

    def _prepare_stan_data(self, df: pd.DataFrame) -> dict:
        """
        Convert a DataFrame (one group with a unique combination of parameters)
        into the corresponding Stan data dictionary.
        Modify this method based on your model's expected data input.
        """
        # Prepare data for Stan.
        stan_data = {
            'N': df.shape[0],
            'M': df['group'].max(),
            'n_language': int(df['language'].max()),
            'n_metric': int(df['metric'].max()),
            'n_task': int(df['task'].max()),
            'group': df['group'].tolist(),
            'language': df['language'].tolist(),
            'metric': df['metric'].tolist(),
            'task': df['task'].tolist(),
            'y': df['y'].tolist()
        }
        return stan_data

    def _arviz_to_cmdstanpy(self, fit, output_path: str, **kwargs):
        cmdstanpy_data = az.from_cmdstanpy(
            posterior = fit,
            **kwargs)
        
        # az.to_json(cmdstanpy_data, output_path)
        
        return cmdstanpy_data

    def _generate_trace_plots(self, cmdstanpy_data, output_path: str):
        """
        Generate and save trace plots for the fit object.
        This is a placeholder function; implement your own plotting logic.
        """
        
        os.makedirs(output_path, exist_ok=True)

        fig = az.plot_trace(cmdstanpy_data, figsize=(20, 50))
        # Save the current figure to the specified path.
        fig.ravel()[0].figure.savefig(output_path, dpi=300, bbox_inches='tight')

        pass

    def _generate_rank_plots(self, cmdstanpy_data, output_path: str):
        """
        Generate and save rank plots for the fit object.
        This is a placeholder function; implement your own plotting logic.
        """
        os.makedirs(output_path, exist_ok=True)

        fig = az.plot_rank(cmdstanpy_data, figsize=(30, 80))
        # Save the current figure to the specified path.
        fig.ravel()[0].figure.savefig(output_path, dpi=300, bbox_inches='tight')

        pass

    def _generate_prior_predictive_checks(self, cmdstanpy_data, output_path):
        """
        Generate and save prior predictive checks for the fit object.
        This is a placeholder function; implement your own plotting logic.
        """

        pass

    def _generate_posterior_predictive_checks(self, cmdstanpy_data, output_path: str):
        """
        Generate and save posterior predictive checks for the fit object.

        Parameters:
        - cmdstanpy_data: An ArviZ InferenceData object containing posterior predictive samples.
          It should include a 'posterior_predictive' group with a key 'y_hat' for predicted values.
        - output_path: A string specifying the file path where the generated plot will be saved.

        Output:
        - Saves a posterior predictive check plot to the specified output path.
        - Raises a KeyError if 'y_hat' is missing in the posterior_predictive group.
        """
        try:
            if "y_hat" not in cmdstanpy_data.posterior_predictive:
                raise KeyError("The key 'y_hat' is missing in cmdstanpy_data.posterior_predictive.")
            fig = az.plot_ppc(cmdstanpy_data, data_pairs={"y": "y_hat"})

            # Ensure the directory exists before saving the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.figure.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"An error occurred during posterior predictive checks: {e}")
            pass

    def _generate_posterior_plots(self, cmdstanpy_data, output_path: str):
        """
        Generate and save posterior plots for the fit object.
        This is a placeholder function; implement your own plotting logic.
        """
        
        try:
            fig = az.plot_forest(
                cmdstanpy_data,
                var_names=self.params,
                figsize=(10, 30),
                colors="C1",
            )

            fig.ravel()[0].figure.savefig(output_path, dpi=300, bbox_inches='tight')

            pass
        except Exception as e:
            print(f"An error occurred during _generate_posterior_plots, probably due to conflicting var_names: {e}")
            pass

    def _generate_prior_posterior_update_plots(self, cmdstanpy_data, output_path):
        """
        Generate and save prior-posterior update plots for the fit object.
        This is a placeholder function; implement your own plotting logic.
        """
        # Example: Save prior-posterior update plots using matplotlib or seaborn
        pass

    def _generate_figures_model(self, cmdstanpy_data, output_path):
        """
        Generate and save figures based on the fit object.
        This is a placeholder function; implement your own plotting logic.
        """
        # Example: Save trace plots, posterior distributions, etc.
        # Use libraries like matplotlib or seaborn for visualization

        self._generate_trace_plots(cmdstanpy_data, output_path)
        self._generate_rank_plots(cmdstanpy_data, output_path)
        # self._generate_prior_predictive_checks(cmdstanpy_data, output_path)
        self._generate_posterior_predictive_checks(cmdstanpy_data, output_path)
        self._generate_posterior_plots(cmdstanpy_data, output_path)
        # self._generate_prior_posterior_update_plots(cmdstanpy_data, output_path)
        pass

    def _generate_figures_recovery(self, recover_object_list: list, output_path: str):
        """
        Generate and save figures for parameter recovery.
        This is a placeholder function; implement your own plotting logic.
        """

        parameter_dict = {f"est_{param}": [] for param in self.params}
        parameter_dict.update({f"true_{param}": [] for param in self.params})

        for val in recover_object_list:
            for param in self.params:
                # Calculate the estimated mean for each parameter
                est_val = val["cmdstanpy_data"].posterior[param].values.flatten().mean()
                # Get the true value from true_values; assumes true_values is a list of dicts with matching keys.
                true_val = val["true_values"][param]
                parameter_dict[f"est_{param}"].append(est_val)
                parameter_dict[f"true_{param}"].append(true_val)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(parameter_dict)
        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(output_path, "parameter_recovery.csv"), index=False)

        # Example: Use seaborn or matplotlib to create scatter plots
        # for each parameter
        for param in self.params:
            plt.figure(figsize=(8, 6))
            plt.scatter(df[f"true_{param}"], df[f"est_{param}"], alpha=0.5)
            plt.plot([df[f"true_{param}"].min(), df[f"true_{param}"].max()],
                     [df[f"true_{param}"].min(), df[f"true_{param}"].max()], 'r--')
            plt.xlabel(f'True {param}')
            plt.ylabel(f'Estimated {param}')
            plt.title(f'Parameter Recovery for {param}')
            plt.savefig(os.path.join(output_path, f'parameter_recovery_{param}.png'))
            plt.close()

        # do interaction scatter plots estimated vs. true parameters but faceted for another true parameter
        pass

    def recover_parameters(self,
                           output_path_data: str,
                           output_path_figures: str,
                           model_fit_params: dict = {"chains": 4,
                                                     "parallel_chains": 4,
                                                     "iter_sampling": 1000,
                                                     "iter_warmup": 500,
                                                     "seed": 42}):
        """
        Fit the Stan model for each combination of parameters present in the data.
        Assumes the data has columns indicating parameter combinations (e.g., 'param1', 'param2').
        Returns a dictionary where keys are tuples of parameter values and values are the fit objects.
        """
        
        # Group data by parameter combination columns; adjust column names as needed.
        grouping_columns = self.group_cols  # e.g., ['param1', 'param2']
        groups = self.data.groupby(grouping_columns)
        
        output = []
        
        # Loop over each combination
        for group_keys, group_data in groups:
            output_ = {"data": None,
                  "true_values": None,
                  "model_fit": None, 
                  "diagnostics": None,
                  "cmdstanpy_data": None}
            output_["data"] = group_data
            output_["true_values"] = group_keys

            stan_data = self._prepare_stan_data(group_data)
            
            # Compile the Stan model (or cache it outside the loop if same for all groups)
            model = CmdStanModel(stan_file=self.model_file)
            
            # Fit the model
            fit = model.sample(data=stan_data, **model_fit_params)
            
            output_["model_fit"] = fit
            print(f"Fitted parameters for group {group_keys}")

            # Save the fits to a file
            output_file = os.path.join(output_path_data, self.data_path.split("/")[-1])
            fit.save_csvfiles(dir=output_file)

            # Save the diagnostic output (str)
            output_["diagnostics"] = fit.diagnose()

            # Convert to ArviZ InferenceData
            output_["cmdstanpy_data"] = self._arviz_to_cmdstanpy(fit, output_file,
                                                      observed_data={"y": stan_data["y"]},
                                                      posterior_predictive="y_hat",)

            # Generate figures
            self._generate_figures_model(output_["cmdstanpy_data"],
                                         output_path = os.path.join(output_path_figures, self.data_path_no_ext.split("/")[-1], ".png"))

            output.append(output_)

        self._generate_figures_recovery(recover_object_list = output, output_path = output_path_figures)

        pass

    def validate_parameters(self):
        # Implement the parameter validation logic here
        return