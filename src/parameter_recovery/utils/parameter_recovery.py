import pandas as pd
from cmdstanpy import CmdStanModel
import os
import arviz as az
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cmdstanpy
import json
import seaborn as sns
import cmdstanpy
sns.set_theme(style="whitegrid")

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
    
    def _generate_figures_recovery(self, all_posteriors, true_params, parameter_name, output_path, point_estimate):

        title_mappings = {
            "mu_alpha": r"$\mu_{\alpha}$",
            "sigma_alpha": r"$\sigma_{\alpha}$",
            "alpha_std": r'Model $\alpha$',
            "mu_beta_language": r'Predictor 1: $\mu_{\beta}$',
            "mu_beta_task": r'Predictor 2: $\mu_{\beta}$',
            "sigma_beta_language": r'Predictor 1: $\sigma_{\beta}$',
            "sigma_beta_task": r'Predictor 2: $\sigma_{\beta}$',
            "beta_language_std": r'Model Predictor 1 $\beta$',
            "beta_task_std": r'Model Predictor 2 $\beta$',
            "phi_alpha": r'$\phi_{\alpha}$',
            "beta_task_phi": r'$\phi_{\beta}$'
        }

        # print(f"Generating figures for parameter: {true_params}")
        if type(true_params[0]) == float or type(true_params[0]) == int:
            K = all_posteriors[0].shape[1]  # Number of parameters
            n_runs = len(all_posteriors)  # Number of runs
            hdi_prob = 0.95  # HDI probability
            
            # Prepare plot
            fig, axes = plt.subplots(1, K, figsize=(8*K, 8), sharey=True)

            if K == 1:
                axes = [axes]
            
            for k in range(K):
                ax = axes[k]
                for run_idx, (true, samples) in enumerate(zip(true_params, all_posteriors)):
                    # Extract samples for this category
                    param_samples = samples[:, k]
                    # Compute mean and HDI
                    if point_estimate == "mean":
                        mean_est = np.mean(param_samples)
                    elif point_estimate == "median":
                        mean_est = np.median(param_samples)
                    else:
                        raise ValueError(f"Unknown point estimate: {point_estimate}")
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
                        # label=f'Run {run_idx+1}' if k == 0 else None,
                        markersize=6,
                        color='tab:blue'
                    )
                # Identity line
                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{title_mappings.get(parameter_name)}')
                ax.set_xlabel('True Value')
                # if k == 0:
                #     ax.set_ylabel('Posterior Mean ± HDI')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                # if k == 0:
                #     ax.legend(loc='upper left', fontsize='small')

            plt.suptitle(f'Parameter Recovery: Mean')
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
            fig, axes = plt.subplots(1, K, figsize=(8*K, 8), sharey=True)

            if K == 1:
                axes = [axes]
            
            for k in range(K):
                ax = axes[k]
                for run_idx, (true, samples) in enumerate(zip(true_params, all_posteriors)):
                    # Extract samples for this category
                    param_samples = samples[:, k]
                    # Compute mean and HDI
                    if point_estimate == "mean":
                        mean_est = np.mean(param_samples)
                    elif point_estimate == "median":
                        mean_est = np.median(param_samples)
                    else:
                        raise ValueError(f"Unknown point estimate: {point_estimate}")
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
                        # label=f'Run {run_idx+1}' if k == 0 else None,
                        markersize=6,
                        color='tab:blue'
                    )
                # Identity line
                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{title_mappings.get(parameter_name)} {k+1}')
                ax.set_xlabel('True Value')
                # if k == 0:
                #     ax.set_ylabel('Posterior Mean ± HDI')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                # if k == 0:
                #     ax.legend(loc='upper left', fontsize='small')

            plt.suptitle(f'Parameter Recovery: Mean')
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

            fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
            axes = axes.flatten()

            for lang_idx in range(var_dim):
                ax = axes[lang_idx]

                for part_idx in range(n_participants):
                    col_idx = lang_idx * n_participants + part_idx  # index in the 60 columns

                    for run_idx, samples_run in enumerate(all_posteriors):
                        param_samples = samples_run[:, col_idx]
                        if point_estimate == "mean":
                            mean_est = np.mean(param_samples)
                        elif point_estimate == "median":
                            mean_est = np.median(param_samples)
                        else:
                            raise ValueError(f"Unknown point estimate: {point_estimate}")
                        # hdi_interval = az.hdi(param_samples, hdi_prob=hdi_prob)

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
                        # label=f'P{part_idx+1}, Run {run_idx+1}' if run_idx == 0 else None,
                        markersize=6,
                        color='tab:blue'
                    )

                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_title(f'{title_mappings.get(parameter_name)} {lang_idx+1}')
                ax.set_xlabel('True Value')
                ax.set_ylabel('Posterior Mean')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                # ax.legend(loc='upper left', fontsize='small')

            fig.suptitle(f'Parameter Recovery for {title_mappings.get(parameter_name)}: Posterior Mean', fontsize=16)
            fig.supxlabel('True Value', fontsize=14)
            fig.supylabel('Estimated Value', fontsize=14)

            # Create a single legend outside the plots
            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles[:n_participants], labels[:n_participants], loc='lower center', ncol=6, fontsize='medium')


            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for suptitle and legend
            output_file = os.path.join(output_path, f"{parameter_name}_recovery.png")
            plt.savefig(output_file)
            plt.close()

    def _generate_figures_recovery_simple(self, all_posteriors, true_params, parameter_name, output_path, point_estimate):
        title_mappings = {
            "mu_alpha": r"Population Level: $\mu_{\alpha}$",
            "sigma_alpha": r"Population Level: $\sigma_{\alpha}$",
            "alpha_std": r'Model Specific: $\alpha$',
            "mu_beta_language": r'Population Level: $\mu_{\beta 1}$',
            "mu_beta_task": r'Population Level: $\mu_{\beta 2}$',
            "sigma_beta_language": r'Population Level: $\sigma_{\beta}$',
            "sigma_beta_task": r'Population Level: $\sigma_{\beta}$',
            "beta_language_std": r'Model Specific: $\beta 1$',
            "beta_task_std": r'Model Specific: $\beta 2$',
            "phi_alpha": r'Population Level: $\phi_{\alpha}$',
            "beta_task_phi": r'Population Level: $\phi_{\beta}$'
        }

        fig, ax = plt.subplots(figsize=(8, 6))

        # Case 1: Each true parameter is a single number (or a 1-element vector)
        if isinstance(true_params[0], (float, int)):
            # Expect each posterior sample to be of shape (n_samples, K),
            # where K could be >1 but we plot all points on a single axis.
            K = all_posteriors[0].shape[1]
            n_runs = len(all_posteriors)
            for run_idx, (true_val, samples) in enumerate(zip(true_params, all_posteriors)):
                for k in range(K):
                    # For scalar parameters K will be 1; for vector parameters this loops over each element.
                    if point_estimate == "mean":
                        est = np.mean(samples[:, k])
                    elif point_estimate == "median":
                        est = np.median(samples[:, k])
                    else:
                        raise ValueError(f"Unknown point estimate: {point_estimate}")
                    # If parameter is scalar, true_val may be a number; if vector, expect a list-like
                    t_val = true_val if K == 1 else true_val[k]
                    sns.scatterplot(
                        x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                    # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

        # Case 2: Each true parameter is a 1D array (vector)
        elif len(true_params[0].shape) == 1:
            K = all_posteriors[0].shape[1]
            n_runs = len(all_posteriors)
            for run_idx, (true_arr, samples) in enumerate(zip(true_params, all_posteriors)):
                for k in range(K):
                    if point_estimate == "mean":
                        est = np.mean(samples[:, k])
                    elif point_estimate == "median":
                        est = np.median(samples[:, k])
                    else:
                        raise ValueError(f"Unknown point estimate: {point_estimate}")
                    t_val = true_arr[k]
                    sns.scatterplot(
                        x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                    # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

        # Case 3: Each true parameter is a 2D array (e.g. for matrices)
        else:
            n_participants, var_dim = true_params[0].shape
            n_runs = len(all_posteriors)
            # Expect the posterior samples to have columns = n_participants * var_dim
            expected_cols = var_dim * n_participants
            if all_posteriors[0].shape[1] != expected_cols:
                raise ValueError(f"Expected {expected_cols} columns but got {all_posteriors[0].shape[1]}")
            for run_idx, (true_mat, samples) in enumerate(zip(true_params, all_posteriors)):
                for lang_idx in range(var_dim):
                    for part_idx in range(n_participants):
                        col_idx = lang_idx * n_participants + part_idx
                        if point_estimate == "mean":
                            est = np.mean(samples[:, col_idx])
                        elif point_estimate == "median":
                            est = np.median(samples[:, col_idx])
                        else:
                            raise ValueError(f"Unknown point estimate: {point_estimate}")
                        t_val = true_mat[part_idx, lang_idx]
                        sns.scatterplot(
                            x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                        # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

        # Identity line and labels
        ax.set_title(f'Parameter Recovery: {title_mappings.get(parameter_name)}', fontsize=20)
        ax.set_xlabel('True Value', fontsize=18)
        ax.set_ylabel('Posterior Mean', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        # if parameter_name contains sigma use 0 to 3, else use -3 to 3
        if "sigma" in parameter_name:
            ax.plot([0, 4], [0, 4], '--', color='gray')
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)
        elif "phi_alpha" in parameter_name:
            ax.plot([0, 4], [0, 4], '--', color='gray')
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)
        else:
            ax.plot([-3, 3], [-3, 3], '--', color='gray')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        # plt.suptitle(f'Parameter Recovery: {point_estimate.capitalize()}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_file = os.path.join(output_path, f"{parameter_name}_recovery.png")
        plt.savefig(output_file)
        plt.close()

    def _generate_figures_recovery_kde(self, all_posteriors, true_params, parameter_name, output_path, point_estimate):
            title_mappings = {
                "mu_alpha": r"Population Level: $\mu_{\alpha}$",
                "sigma_alpha": r"Population Level: $\sigma_{\alpha}$",
                "alpha_std": r'Model Specific: $\alpha$',
                "mu_beta_language": r'Population Level: $\mu_{\beta 1}$',
                "mu_beta_task": r'Population Level: $\mu_{\beta 2}$',
                "sigma_beta_language": r'Population Level: $\sigma_{\beta}$',
                "sigma_beta_task": r'Population Level: $\sigma_{\beta}$',
                "beta_language_std": r'Model Specific: $\beta 1$',
                "beta_task_std": r'Model Specific: $\beta 2$',
                "phi_alpha": r'Population Level: $\phi_{\alpha}$',
                "beta_task_phi": r'Population Level: $\phi_{\beta}$'
            }

            fig, ax = plt.subplots(figsize=(8, 6))
            t_val = []
            est = []

            # Case 1: Each true parameter is a single number (or a 1-element vector)
            if isinstance(true_params[0], (float, int)):
                # Expect each posterior sample to be of shape (n_samples, K),
                # where K could be >1 but we plot all points on a single axis.
                K = all_posteriors[0].shape[1]
                n_runs = len(all_posteriors)
                for run_idx, (true_val, samples) in enumerate(zip(true_params, all_posteriors)):
                    for k in range(K):
                        # For scalar parameters K will be 1; for vector parameters this loops over each element.
                        if point_estimate == "mean":
                            est.append(np.mean(samples[:, k]))
                        elif point_estimate == "median":
                            est = np.median(samples[:, k])
                        else:
                            raise ValueError(f"Unknown point estimate: {point_estimate}")
                        # If parameter is scalar, true_val may be a number; if vector, expect a list-like
                        t_val.append(true_val if K == 1 else true_val[k])
                        # sns.scatterplot(
                        #     x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                        # sns.kdeplot(x=[t_val], y=[est], fill=True, cmap="inferno", ax=ax)
                        # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

            # Case 2: Each true parameter is a 1D array (vector)
            elif len(true_params[0].shape) == 1:
                K = all_posteriors[0].shape[1]
                n_runs = len(all_posteriors)
                for run_idx, (true_arr, samples) in enumerate(zip(true_params, all_posteriors)):
                    for k in range(K):
                        if point_estimate == "mean":
                            est.append(np.mean(samples[:, k]))
                        elif point_estimate == "median":
                            est = np.median(samples[:, k])
                        else:
                            raise ValueError(f"Unknown point estimate: {point_estimate}")
                        t_val.append(true_arr[k])
                        # sns.scatterplot(
                        #     x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                        # sns.kdeplot(x=[t_val], y=[est], fill=True, cmap="inferno", ax=ax)
                        # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

            # Case 3: Each true parameter is a 2D array (e.g. for matrices)
            else:
                n_participants, var_dim = true_params[0].shape
                n_runs = len(all_posteriors)
                # Expect the posterior samples to have columns = n_participants * var_dim
                expected_cols = var_dim * n_participants
                if all_posteriors[0].shape[1] != expected_cols:
                    raise ValueError(f"Expected {expected_cols} columns but got {all_posteriors[0].shape[1]}")
                for run_idx, (true_mat, samples) in enumerate(zip(true_params, all_posteriors)):
                    for lang_idx in range(var_dim):
                        for part_idx in range(n_participants):
                            col_idx = lang_idx * n_participants + part_idx
                            if point_estimate == "mean":
                                est.append(np.mean(samples[:, col_idx]))
                            elif point_estimate == "median":
                                est = np.median(samples[:, col_idx])
                            else:
                                raise ValueError(f"Unknown point estimate: {point_estimate}")
                            t_val.append(true_mat[part_idx, lang_idx])
                            # sns.scatterplot(
                            #     x=[t_val], y=[est], marker='o', s=30, color='tab:blue', ax=ax)
                            # sns.kdeplot(x=[t_val], y=[est], fill=True, cmap="inferno", ax=ax)
                            # ax.plot(t_val, est, marker='o', linestyle='none', markersize=6, color='tab:blue')

            sns.kdeplot(x=t_val, y=est, fill=True, cmap="inferno", ax=ax)
            # Identity line and labels
            ax.set_title(f'Parameter Recovery: {title_mappings.get(parameter_name)}', fontsize=20)
            ax.set_xlabel('True Value', fontsize=18)
            ax.set_ylabel('Posterior Mean', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            # if parameter_name contains sigma use 0 to 3, else use -3 to 3
            if "sigma" in parameter_name:
                ax.plot([0, 4], [0, 4], '--', color='gray')
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 4)
            elif "phi_alpha" in parameter_name:
                ax.plot([0, 4], [0, 4], '--', color='gray')
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 4)
            else:
                ax.plot([-3, 3], [-3, 3], '--', color='gray')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
            # plt.suptitle(f'Parameter Recovery: {point_estimate.capitalize()}')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
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
                                                     "seed": 123,
                                                    #  "output_dir": "output/stan_fits",
                                                     "adapt_delta": 0.95,
                                                    #  "show_console": True,
                                                     }):
        
        all_posteriors = {param: [] for param in parameter_names}
        true_params = {param: [] for param in parameter_names}
        failed_fits = []
        failed_fits_parameters = {param: [] for param in parameter_names}

        if os.path.exists(os.path.join(output_path_data)):
            print(f"Parameter recovery already run.")
            # read from csv
            for i in range(len(data)):
                try:
                    fit = cmdstanpy.from_csv(path=os.path.join(output_path_data, f"fit_{i}"))
                    for j in parameter_names:
                        posterior_samples = fit.draws_pd(j)
                        all_posteriors[j].append(posterior_samples.values)
                        true_params[j].append(data[i]['param_combo'][j])
                except Exception as e:
                    print(f"Error loading model for dataset {j}: {e}")
                    continue
        else:
            for i in range(len(data)):
            # for i in range(2):
                # if fit_i exists, skip
                # if os.path.exists(os.path.join(output_path_data, f"fit_{i}")):
                #     print(f"Skipping dataset {i}, already fitted.")
                #     continue
                os.makedirs(output_path_data, exist_ok=True)

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

                    for i in parameter_names:
                        posterior_samples = fit.draws_pd(i)
                        all_posteriors[i].append(posterior_samples.values)
                        true_params[i].append(data_['param_combo'][i])
                except Exception as e:
                    print(f"Error fitting model for dataset {i}: {e}")
                    failed_fits.append(i)
                    for param in parameter_names:
                        failed_fits_parameters[param].append(data_[f'param_combo'][param])
                    continue
            
            # Save failed fits information
            if failed_fits:
                # save as json
                failed_fits_info = {
                    "failed_fits": failed_fits,
                    "failed_fits_parameters": failed_fits_parameters
                }
                with open(os.path.join(output_path_data, "failed_fits.json"), "w") as f:
                    json.dump(failed_fits_info, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)


        for i in parameter_names:
            # Check if the directory exists, if not create it
            os.makedirs(os.path.join(output_path_figures, 'mean'), exist_ok=True)
            os.makedirs(os.path.join(output_path_figures, 'kde'), exist_ok=True)
            # os.makedirs(os.path.join(output_path_figures, 'median'), exist_ok=True)
            self._generate_figures_recovery_simple(all_posteriors = all_posteriors[i], 
                                                true_params = true_params[i], 
                                                parameter_name = i, 
                                                output_path = os.path.join(output_path_figures, "mean"),
                                                point_estimate = "mean")
            # self._generate_figures_recovery(all_posteriors = all_posteriors[i],
            #                                     true_params = true_params[i], 
            #                                     parameter_name = i, 
            #                                     output_path = os.path.join(output_path_figures, "median"),
            #                                     point_estimate = "median")
            self._generate_figures_recovery_kde(all_posteriors = all_posteriors[i],
                                                true_params = true_params[i], 
                                                parameter_name = i, 
                                                output_path = os.path.join(output_path_figures, "kde"),
                                                point_estimate = "mean")


        print(f"Model fit completed for {len(data)} datasets.")
        
        # Load the fitted models and extract the posterior samples.
        
        # for i in parameter_names:
        #     if os.path.exists(os.path.join(output_path_figures, f"{i}_recovery.png")):
        #         print(f"Skipping parameter {i}, already recovered.")
        #         continue

        #     all_posteriors = []
        #     true_params = []
        #     for j in range(len(data)):
        #     # for j in range(10):
        #         try:
        #             fit = cmdstanpy.from_csv(os.path.join(output_path_data, f"fit_{j}"))
        #             posterior_samples = fit.draws_pd(i)
        #             all_posteriors.append(posterior_samples.values)
        #             true_params.append(data[j]['param_combo'][i])
        #         except Exception as e:
        #             print(f"Error loading posterior samples for dataset {j}: {e}")
        #             continue

        #     self._generate_figures_recovery(all_posteriors = all_posteriors, 
        #                                         true_params = true_params, 
        #                                         parameter_name = i, 
        #                                         output_path = output_path_figures)

        pass