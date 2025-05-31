import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from cmdstanpy import CmdStanModel
import cmdstanpy
import numpy as np
import arviz as az
import pickle

class ModelDevelopment():
    """
    A class for parameter estimation using Stan.
    """

    DEFAULT_PARAMETERS = [
        "mu_alpha",
        "sigma_alpha",
        "alpha_std",
        "mu_beta_language",
        "mu_beta_task",
        "sigma_beta_language",
        "sigma_beta_task",
        "beta_language_std",
        "beta_task_std",
        "phi_alpha",
        "beta_task_phi"
        ]

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
        self.data = None
        self.stan_data = None
        self.model_mapping = None
        self.task_mapping = None
        self.language_mapping = None
        self.fit = None

    def load_data(self, data_path: str):

        with open(data_path, "rb") as file:
            data = pickle.load(file)

        return data[100]
    
    def estimate_parameters(self,
                            data,
                            stan_file: str,
                            output_path_data, 
                            output_path_figures,
                            # parameter_names: list = DEFAULT_PARAMETERS,
                            model_fit_params: dict = {"chains": 1,
                                                     "iter_sampling": 2000,
                                                     "iter_warmup": 1000,
                                                     "seed": 42,
                                                    #  "output_dir": "output/stan_fits",
                                                     "adapt_delta": 0.95,}):
        """
        Estimate parameters using the Stan model.

        :param output_path_data: Path to save output data.
        :param output_path_figures: Path to save output figures.
        """

        os.makedirs(self.output_path_figures, exist_ok=True)
        os.makedirs(os.path.join(output_path_data, "overview"), exist_ok=True)

        # Compile the Stan model using cmdstanpy.
        if os.path.exists(os.path.join(output_path_data, "stan_fit")):
                print(f"Model already fitted.")

                # read from csv
                self.fit = cmdstanpy.from_csv(path=os.path.join(output_path_data, "stan_fit"), method = "sample")
        else:
            model = CmdStanModel(stan_file=stan_file)

            self.fit = model.sample(
                    data=data['data_dict'],
                    **model_fit_params,
                )
            
            print("Fitting completed.")
            print("Saving results...")

            os.makedirs(os.path.join(output_path_data, "stan_fit"), exist_ok=True)
            
            self.fit.save_csvfiles(os.path.join(output_path_data, "stan_fit"))

            print("Results saved.")
            print("Generating summary...")

            fit_summary = self.fit.summary()

            fit_summary.to_csv(os.path.join(output_path_data, "overview/summary.csv"), index=True)

            print("Summary saved.")
            print("Generating diagnostics...")

            fit_diagnostics = self.fit.diagnose()

            with open(os.path.join(output_path_data, "overview/diagnostics.txt"), 'w') as f:
                f.write(fit_diagnostics)

            print("Diagnostics saved.")

        pass

    def summary_plots(self, data, type: str):
        """
        Generate summary plots.
        """
        print("Generating summary plots...")

        if type == "prior":
            plt.hist(self.fit.stan_variable("y_pred").reshape(-1))
            plt.xlabel('y')
            plt.title('Prior Predictive Distribution', fontsize=18)
            plt.savefig(os.path.join(self.output_path_figures, "prior_predictive_distribution.png"))
            plt.close()

        elif type == "posterior":
            coords = {"obs_id": np.arange(data['data_dict']['N']),}
            dims = {"y": ["obs_id"], "y_pred": ["obs_id"]}

            cmdstanpy_data = az.from_cmdstanpy(
                posterior=self.fit,
                observed_data={'y': data['data_dict']['y']},
                coords=coords,
                dims=dims,
                posterior_predictive="y_pred",
                )

            # Create a directory for the summary plots if it doesn't exist
            os.makedirs(self.output_path_figures, exist_ok=True)

            if not os.path.exists(os.path.join(self.output_path_figures, "posterior_ppc.png")):

                plt.figure(figsize=(15, 10))
                az.plot_ppc(cmdstanpy_data, 
                    data_pairs={"y":"y_pred"})
                plt.xlabel('y')
                plt.title('Posterior Predictive Check', fontsize=18)
                plt.savefig(os.path.join(self.output_path_figures, "posterior_ppc.png"))
                plt.close()
            
            # read the summary from the fit
            fit_summary = self.fit.summary()

            # plot histogram of R_hat values
            plt.figure(figsize=(10, 6))
            plt.hist(fit_summary['R_hat'], bins=30, color='blue', alpha=0.7)
            plt.xlabel(r'$\hat{R}$', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Histogram of R-hat Values', fontsize=18)
            plt.savefig(os.path.join(self.output_path_figures, "r_hat_histogram.png"))
            plt.close()

        pass