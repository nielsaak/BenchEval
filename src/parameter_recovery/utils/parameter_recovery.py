import pandas as pd
from cmdstanpy import CmdStanModel
import os
import arviz as az
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cmdstanpy

class ParameterRecovery():
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

    def __init__(self):
        return

    def load_data(data_path: str):

        with open(data_path, "rb") as file:
            data = pickle.load(file)

        return data
    
    def _generate_figures_recovery(self, all_posteriors, true_params, parameter_name, output_path):

        if type(true_params[0]) == float or type(true_params[0]) == int:
            K = all_posteriors[0].shape[1]  # Number of parameters
            n_runs = len(all_posteriors)  # Number of runs
            hdi_prob = 0.95  # HDI probability
            
            # Prepare plot
            fig, axes = plt.subplots(1, K, figsize=(4*K, 4), sharey=True)

            markers = ['o', 's', '^', 'D']
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            if K == 1:
                axes = [axes]
            
            for k in range(K):
                ax = axes[k]
                for run_idx, (true, samples) in enumerate(zip(true_params, all_posteriors)):
                    # Extract samples for this category
                    param_samples = samples[:, k]
                    # Compute mean and HDI
                    mean_est = np.mean(param_samples)
                    # hdi_interval = az.hdi(param_samples, hdi_prob=hdi_prob)

                    if K == 1:
                        true = [true]
                    
                    # Plot mean point with horizontal jitter at true value
                    # ax.errorbar(
                    #     true[k] + (run_idx - (n_runs-1)/2) * 0.005,  # small offset
                    #     mean_est,
                    #     yerr=[[mean_est - hdi_interval[0]], [hdi_interval[1] - mean_est]],
                    #     # fmt=markers[run_idx],
                    #     label=f'Run {run_idx+1}' if k == 0 else None,
                    #     capsize=4,
                    #     markersize=6,
                    #     # color=colors[run_idx]
                    # )
                    ax.plot(
                        true[k],
                        mean_est,
                        marker='o',
                        linestyle='none',
                        label=f'Run {run_idx+1}' if k == 0 else None,
                        markersize=6,
                        # color=colors[run_idx]
                    )
                # Identity line
                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{parameter_name} {k+1}')
                ax.set_xlabel('True Value')
                if k == 0:
                    ax.set_ylabel('Posterior Mean ± HDI')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                if k == 0:
                    ax.legend(loc='upper left', fontsize='small')

            plt.suptitle(f'Parameter Recovery: Mean ± {int(hdi_prob*100)}% HDI')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the plot to file
            output_file = os.path.join(output_path, f"{parameter_name}_recovery.png")
            plt.savefig(output_file)
            plt.close()
        elif len(true_params[0].shape) == 1:
            K = all_posteriors[0].shape[1]  # Number of parameters
            n_runs = len(all_posteriors)  # Number of runs
            hdi_prob = 0.95  # HDI probability
            
            # Prepare plot
            fig, axes = plt.subplots(1, K, figsize=(4*K, 4), sharey=True)

            markers = ['o', 's', '^', 'D']
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            if K == 1:
                axes = [axes]
            
            for k in range(K):
                ax = axes[k]
                for run_idx, (true, samples) in enumerate(zip(true_params, all_posteriors)):
                    # Extract samples for this category
                    param_samples = samples[:, k]
                    # Compute mean and HDI
                    mean_est = np.mean(param_samples)
                    # hdi_interval = az.hdi(param_samples, hdi_prob=hdi_prob)

                    if K == 1:
                        true = [true]
                    
                    # Plot mean point with horizontal jitter at true value
                    # ax.errorbar(
                    #     true[k] + (run_idx - (n_runs-1)/2) * 0.005,  # small offset
                    #     mean_est,
                    #     yerr=[[mean_est - hdi_interval[0]], [hdi_interval[1] - mean_est]],
                    #     # fmt=markers[run_idx],
                    #     label=f'Run {run_idx+1}' if k == 0 else None,
                    #     capsize=4,
                    #     markersize=6,
                    #     # color=colors[run_idx]
                    # )
                    ax.plot(
                        true[k],
                        mean_est,
                        marker='o',
                        linestyle='none',
                        label=f'Run {run_idx+1}' if k == 0 else None,
                        markersize=6,
                        # color=colors[run_idx]
                    )
                # Identity line
                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{parameter_name} {k+1}')
                ax.set_xlabel('True Value')
                if k == 0:
                    ax.set_ylabel('Posterior Mean ± HDI')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                if k == 0:
                    ax.legend(loc='upper left', fontsize='small')

            plt.suptitle(f'Parameter Recovery: Mean ± {int(hdi_prob*100)}% HDI')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the plot to file
            output_file = os.path.join(output_path, f"{parameter_name}_recovery.png")
            plt.savefig(output_file)
            plt.close()
        else:
            hdi_prob = 0.95
            n_participants = true_params[0].shape[0]
            var_dim = true_params[0].shape[1]
            K = 1  # Assuming one parameter per column
            n_runs = len(all_posteriors)

            # Check that columns match expectation
            expected_cols = var_dim * n_participants
            assert all_posteriors[0].shape[1] == expected_cols, f"Expected {expected_cols} columns but got {all_posteriors[0].shape[1]}"

            markers = ['o', 's', '^', 'D', 'v', '*']
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
            axes = axes.flatten()

            for lang_idx in range(var_dim):
                ax = axes[lang_idx]

                for part_idx in range(n_participants):
                    col_idx = lang_idx * n_participants + part_idx  # index in the 60 columns

                    for run_idx, samples_run in enumerate(all_posteriors):
                        param_samples = samples_run[:, col_idx]
                        mean_est = np.mean(param_samples)
                        hdi_interval = az.hdi(param_samples, hdi_prob=hdi_prob)

                        # Horizontal jitter for visualization
                        true_val = true_params[run_idx][part_idx][lang_idx]
                        # x_jittered = true_val + (run_idx - (n_runs - 1) / 2) * 0.01
                        # ax.errorbar(
                        #     x_jittered,
                        #     mean_est,
                        #     yerr=[[mean_est - hdi_interval[0]], [hdi_interval[1] - mean_est]],
                        #     # fmt=markers[part_idx % len(markers)],
                        #     label=f'P{part_idx+1}, Run {run_idx+1}' if run_idx == 0 else None,
                        #     capsize=4,
                        #     markersize=6,
                        #     # color=colors[part_idx % len(colors)]
                        # )
                        ax.plot(
                        true_val,
                        mean_est,
                        marker='o',
                        linestyle='none',
                        label=f'P{part_idx+1}, Run {run_idx+1}' if run_idx == 0 else None,
                        markersize=6,
                        # color=colors[run_idx]
                    )

                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{parameter_name} Recovery - Language {lang_idx+1}')
                ax.set_xlabel('True Value')
                ax.set_ylabel('Posterior Mean ± HDI')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                # ax.legend(loc='upper left', fontsize='small')

            fig.suptitle(f'Parameter Recovery for {parameter_name}: Posterior Mean ± {int(hdi_prob*100)}% HDI', fontsize=16)
            fig.supxlabel('True Value', fontsize=14)
            fig.supylabel('Estimated Value', fontsize=14)

            # Create a single legend outside the plots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles[:n_participants], labels[:n_participants], loc='lower center', ncol=6, fontsize='medium')


            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for suptitle and legend
            output_file = os.path.join(output_path, f"{parameter_name}_recovery.png")
            plt.savefig(output_file)
            plt.close()

    def recover_parameters(self,
                           data,
                           stan_file: str,
                           output_path_data: str,
                           output_path_figures: str,
                           parameter_names: list = DEFAULT_PARAMETERS,
                           model_fit_params: dict = {"chains": 1,
                                                     "iter_sampling": 1000,
                                                     "iter_warmup": 500,
                                                     "seed": 42,
                                                    #  "output_dir": "output/stan_fits",
                                                     "adapt_delta": 0.95,}):
        
        for i in range(len(data)):
        # for i in range(10):
            # if fit_i exists, skip
            if os.path.exists(os.path.join(output_path_data, f"fit_{i}")):
                print(f"Skipping dataset {i}, already fitted.")
                continue

            data_ = data[i]

            # Compile the Stan model using cmdstanpy.
            model = CmdStanModel(stan_file=stan_file)

            try:
                # Fit the model using MCMC sampling.
                fit = model.sample(
                    data=data_["data_dict"],
                    **model_fit_params,
                )

                fit.save_csvfiles(os.path.join(output_path_data, f"fit_{i}"))
            except Exception as e:
                print(f"Error fitting model for dataset {i}: {e}")
                continue
        
        print(f"Model fit completed for {len(data)} datasets.")
        
        # Load the fitted models and extract the posterior samples.
        
        for i in parameter_names:
            if os.path.exists(os.path.join(output_path_figures, f"{i}_recovery.png")):
                print(f"Skipping parameter {i}, already recovered.")
                continue

            all_posteriors = []
            true_params = []
            for j in range(len(data)):
            # for j in range(10):
                try:
                    fit = cmdstanpy.from_csv(os.path.join(output_path_data, f"fit_{j}"))
                    posterior_samples = fit.draws_pd(i)
                    all_posteriors.append(posterior_samples.values)
                    true_params.append(data[j]['param_combo'][i])
                except Exception as e:
                    print(f"Error loading posterior samples for dataset {j}: {e}")
                    continue

            self._generate_figures_recovery(all_posteriors = all_posteriors, 
                                                true_params = true_params, 
                                                parameter_name = i, 
                                                output_path = output_path_figures)

        pass