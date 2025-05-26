import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from cmdstanpy import CmdStanModel
import cmdstanpy
import numpy as np
import arviz as az

class ParameterEstimation():
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
                        row["test_index"] = idx + 1
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

        self.data = pd.DataFrame(rows)

    def preprocess_data(self, top_n = None, top_n_path = None, test_index_n = None):
        """
        Preprocess the loaded data.
        """
        df_full = self.data

        # remove task == speed
        remove_list = ['test_speed', 'test_speed_short', 'test_runtime', 'test_samples_per_second', 'test_steps_per_second', 'epoch', 'test_micro_f1_no_misc', 'test_micro_f1', 'test_em', 'test_f1', 'test_accuracy', 'test_loss', 'test_mcc', 'test_macro_f1']
        df_full = df_full[~df_full['metric'].isin(remove_list)]

        # remove few_shot == False
        df_full = df_full[df_full['few_shot'] == True]

        # use only the first n test_index
        if test_index_n is not None:
            df_full = df_full[df_full['test_index'] <= test_index_n]


        # remove value scores equal to or below 0
        # df_full = df_full[df_full['value'] > 0]

        # make a column called language, that takes the first of the list that is in dataset_languages
        def get_language(row):
            if row['dataset_languages']:
                # if nb, nn or no, just return 'no'
                if row['dataset_languages'][0] in ['nb', 'nn', 'no']:
                    return 'no'
                # if there are multiple languages, return the first one
                elif len(row['dataset_languages']) > 1:
                    print(f"Multiple languages found: {row['dataset_languages']}")
                else:
                    return row['dataset_languages'][0]

                return row['dataset_languages'][0]
            return None
        df_full['language'] = df_full.apply(get_language, axis=1)

        # mcc goes from -1 to 1. This should be transformed to 0 to 1
        def transform_mcc(value):
            if value == -1:
                return 0
            elif value == 1:
                return 1
            elif value < -1 or value > 1:
                raise ValueError(f"Invalid MCC value: {value}")
            else:
                return (value + 1) / 2

        # em goes from 0 to 100. This should be transformed to 0 to 1
        def transform_em(value):
            if value < 0 or value > 100:
                raise ValueError(f"Invalid EM value: {value}")
            else:
                return value / 100
            
        # f1 goes from 0 to 100. This should be transformed to 0 to 1
        def transform_f1(value):
            if value < 0 or value > 100:
                raise ValueError(f"Invalid F1 value: {value}")
            else:
                return value / 100

        # Apply the transformation to the 'value' column for 'mcc' metrics
        df_full.loc[df_full['metric'].str.lower() == 'mcc', 'value'] = df_full.loc[df_full['metric'].str.lower() == 'mcc', 'value'].apply(transform_mcc)

        # Apply the transformation to the 'value' column for 'em' metrics
        df_full.loc[df_full['metric'].str.lower() == 'em', 'value'] = df_full.loc[df_full['metric'].str.lower() == 'em', 'value'].apply(transform_em)

        # Apply the transformation to the 'value' column for 'f1' metrics
        df_full.loc[df_full['metric'].str.lower() == 'f1', 'value'] = df_full.loc[df_full['metric'].str.lower() == 'f1', 'value'].apply(transform_f1)

        # choose only the primary metric for each task
        primary_metrics = {
            'common-sense-reasoning': 'mcc',
            'knowledge': 'mcc',
            'linguistic-acceptability': 'mcc',
            'named-entity-recognition': 'micro_f1_no_misc',
            'reading-comprehension': 'em',
            'sentiment-classification': 'mcc',
            'summarization': 'bertscore'
        }

        # Filter the DataFrame to keep only the primary metric for each task
        def filter_primary_metric(row):
            task = row['task']
            metric = row['metric'].lower()
            if task in primary_metrics:
                return metric == primary_metrics[task]
            return False
        
        # remove where df_full["generative"] is false
        df_full = df_full[df_full["generative"] == True]

        # Apply the filter to the DataFrame
        df_full = df_full[df_full.apply(filter_primary_metric, axis=1)]

        if top_n is not None:
            df_euro = pd.read_csv(top_n_path)
            df_euro = df_euro[~df_euro['model'].str.contains('zero-shot', na=False)]
            df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot\)', '', regex=True)
            df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot, val\)', '', regex=True)
            df_euro_100 = df_euro.nsmallest(top_n, 'rank')
            df_full = df_full[df_full['model'].isin(df_euro_100['model'].tolist())]



        print(f"Unique tasks: {df_full['task'].unique()}")
        print(f"Unique metrics: {df_full['metric'].unique()}")
        print(f"Unique languages: {df_full['language'].unique()}")
        print(f"Number of unique models: {len(df_full['model'].unique())}")
        print(f"Number of observations: {len(df_full)}")
        
        self.data = df_full
    
    def data_description_plots(self):
        """
        Create data description plots.
        """
        os.makedirs(self.output_path_figures, exist_ok=True)

        df = self.data.copy()

        # if value == 0 add 1e-5
        df['value'] = df['value'].replace(0, 1e-5)
        # if value == 1 subtract 1e-5
        df['value'] = df['value'].replace(1, 1 - 1e-5)

        # Group the data by 'task' and 'metric'
        groups = list(df.groupby(['task', 'metric']))
        n_plots = len(groups)
        ncols = 3  # You can adjust the number of columns as needed
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for ax, ((task, metric), group) in zip(axes, groups):
            group['value'].plot.hist(bins=30, edgecolor='black', ax=ax)
            ax.set_title(f"Task: {task}, Metric: {metric}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        # Remove unused subplots if there are any
        for ax in axes[len(groups):]:
            ax.remove()

        plt.tight_layout()
        # Save the plot to file
        output_file = os.path.join(self.output_path_figures, "data_histograms.png")
        plt.savefig(output_file)
        plt.close()
        pass

    def prepare_data_for_stan(self, limit_models=None):
        """
        Prepare data for Stan model.
        """
        df = self.data.copy()

        # if value == 0 add 1e-5
        df['value'] = df['value'].replace(0, 1e-5)
        # if value == 1 subtract 1e-5
        df['value'] = df['value'].replace(1, 1 - 1e-5)

        # limit the number of models
        if limit_models is not None:
            # change name of models to numbers starting from 1
            df['model_id'] = df['model'].map({name: i + 1 for i, name in enumerate(df['model'].unique())})
            model_list = list(range(1, limit_models))
            df = df[df['model_id'].isin(model_list)]


        # Convert the DataFrame to a dictionary format suitable for Stan
        stan_data = {
            'N': len(df),
            'M': len(df['model'].unique()),
            'n_language': len(df['language'].unique()),
            'n_task': len(df['task'].unique()),
            'group': df['model'].map({name: i + 1 for i, name in enumerate(df['model'].unique())}).tolist(),
            'language': df['language'].map({name: i + 1 for i, name in enumerate(df['language'].unique())}).tolist(),
            'task': df['task'].map({name: i + 1 for i, name in enumerate(df['task'].unique())}).tolist(),
            'y': [val / 100 if val > 1 else val for val in df['value'].tolist()]
        }

        model_mapping = {name: i + 1 for i, name in enumerate(df['model'].unique())}
        task_mapping = {name: i + 1 for i, name in enumerate(df['task'].unique())}
        language_mapping = {name: i + 1 for i, name in enumerate(df['language'].unique())}

        # get length of each element in stan_data
        for key, value in stan_data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)}")
            else:
                print(f"{key}: {value}")

        self.stan_data = stan_data
        self.model_mapping = model_mapping
        self.task_mapping = task_mapping
        self.language_mapping = language_mapping

        # Save the mappings to CSV files
        os.makedirs(self.output_path_data, exist_ok=True)
        pd.DataFrame(list(model_mapping.items()), columns=['model', 'model_id']).to_csv(os.path.join(self.output_path_data, "model_mapping.csv"), index=False)
        pd.DataFrame(list(task_mapping.items()), columns=['task', 'task_id']).to_csv(os.path.join(self.output_path_data, "task_mapping.csv"), index=False)
        pd.DataFrame(list(language_mapping.items()), columns=['language', 'language_id']).to_csv(os.path.join(self.output_path_data, "language_mapping.csv"), index=False)
    
    def estimate_parameters(self,
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
                    data=self.stan_data,
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

    def summary_plots(self):
        """
        Generate summary plots.
        """
        print("Generating summary plots...")
        # Load the fit data
        # fit = self.fit

        coords = {"obs_id": np.arange(self.stan_data['N']),}
        dims = {"y": ["obs_id"], "y_pred": ["obs_id"]}

        cmdstanpy_data = az.from_cmdstanpy(
            posterior=self.fit,
            observed_data={'y': self.stan_data['y']},
            coords=coords,
            dims=dims,
            posterior_predictive="y_pred",
            )

        # Create a directory for the summary plots if it doesn't exist
        os.makedirs(self.output_path_figures, exist_ok=True)

        if not os.path.exists(os.path.join(self.output_path_figures, "overall_ppc.png")):

            az.plot_ppc(cmdstanpy_data, 
                data_pairs={"y":"y_pred"})
            plt.xlabel('y')

            plt.savefig(os.path.join(self.output_path_figures, "overall_ppc.png"))
            plt.close()

        if not os.path.exists(os.path.join(self.output_path_figures, "language_ppc.png")):

            print("Generating language summary plots...")

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()

            for i in range(1, self.stan_data["n_language"] + 1):
                az.plot_ppc(
                    cmdstanpy_data, 
                    data_pairs={"y": "y_pred"}, 
                    coords={"obs_id": np.where(np.array(self.stan_data["language"]) == i)[0]},
                    ax=axes[i-1],
                    legend=False
                )
                axes[i-1].set_xlabel('y')
                axes[i-1].set_title(f"Language: {next((k for k, v in self.language_mapping.items() if v == i), "Unknown")}")
            
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
            plt.suptitle(f'Posterior Predictive Check: Language')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path_figures, "language_ppc.png"))
            plt.close()

        if not os.path.exists(os.path.join(self.output_path_figures, "task_ppc.png")):

            print("Generating task summary plots...")

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.flatten()

            for i in range(1, self.stan_data["n_task"] + 1):
                az.plot_ppc(
                    cmdstanpy_data, 
                    data_pairs={"y": "y_pred"}, 
                    coords={"obs_id": np.where(np.array(self.stan_data["task"]) == i)[0]},
                    ax=axes[i-1],
                    legend=False
                )
                axes[i-1].set_xlabel('y')
                axes[i-1].set_title(f"Task: {next((k for k, v in self.task_mapping.items() if v == i), "Unknown")}")

            for ax in axes[self.stan_data["n_task"]:]:
                ax.remove()

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
            plt.suptitle(f'Posterior Predictive Check: Task')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path_figures, "task_ppc.png"))
            plt.close()

        if not os.path.exists(os.path.join(self.output_path_figures, "model_ppc.png")):

            print("Generating model summary plots...")

            # models_to_show = [1,10, 20, 30, 40, 50]
            plots_per_fig = 25
            n_figs = 4
            rows, cols = 5, 5

            for fig_idx in range(n_figs):
                fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
                axes = axes.flatten()
                start = fig_idx * plots_per_fig
                end   = start + plots_per_fig

                for i in range(start, end):
                    az.plot_ppc(
                        cmdstanpy_data, 
                        data_pairs={"y": "y_pred"}, 
                        coords={"obs_id": np.where(np.array(self.stan_data["group"]) == i + 1)[0]},
                        ax=axes[i-start],
                        legend=False
                    )
                    axes[i-start].set_xlabel('y')
                    axes[i-start].set_title(f"LLM {i + 1}")

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.03))
                plt.suptitle(f'Posterior Predictive Check: LLM', y=0.95)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path_figures, f"model_ppc_{fig_idx + 1}.png"))
                plt.close()


        pass

    def posterior_distribution_plots(self):
        language_map = {
            'en': 'English',
            'de': 'German',
            'es': 'Spanish',
            'fr': 'French',
            'it': 'Italian',
            'da': 'Danish',
            'no': 'Norwegian',
            'sv': 'Swedish',
            'fi': 'Finnish',
            'nl': 'Dutch',
            'fo': 'Faroese',
            'is': 'Icelandic',
        }

        coords = {"language_coord": [language_map.get(i, 'Unknown') for i in self.language_mapping.keys()],
                    "task_coord": [i for i in list(self.task_mapping.keys())],
                  }

        dims = {"mu_beta_language": ["language_coord"],
                "sigma_beta_language": ["language_coord"],
                "mu_beta_task": ["task_coord"],
                "sigma_beta_task": ["task_coord"],
                "beta_task_phi": ["task_coord"],}

        cmdstanpy_data = az.from_cmdstanpy(
            posterior=self.fit,
            observed_data={'y': self.stan_data['y']},
            coords=coords,
            dims=dims,
            posterior_predictive="y_pred",
            )
        
        variable_mappings = {
            "mu_alpha": r"$\mu_{\alpha}$",
            "sigma_alpha": r"$\sigma_{\alpha}$",
            "alpha_std": r'Model $\alpha$',
            "mu_beta_language": r'Language: $\mu_{\beta}$',
            "mu_beta_task": r'Task: $\mu_{\beta}$',
            "sigma_beta_language": r'Language: $\sigma_{\beta}$',
            "sigma_beta_task": r'Task: $\sigma_{\beta}$',
            "beta_language_std": r'Model Language $\beta$',
            "beta_task_std": r'Model Task $\beta$',
            "phi_alpha": r'$\phi_{\alpha}$',
            "beta_task_phi": r'$\phi_{\beta}$'
        }

        cmdstanpy_data = cmdstanpy_data.rename_vars(variable_mappings)

        az.style.use(["arviz-whitegrid", "arviz-viridish"])
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()

        # Plot alpha parameters: mu_alpha, sigma_alpha, and alpha_std
        az.plot_forest(
            cmdstanpy_data,
            var_names=[
                variable_mappings.get("mu_alpha"),
                variable_mappings.get("sigma_alpha"),
            ],
            colors="C1",
            hdi_prob=0.95,
            ax=axes[0]
        )
        axes[0].set_title("Intercept Parameters")

        # Plot language parameters: mu_beta_language, sigma_beta_language, and beta_language_std
        az.plot_forest(
            cmdstanpy_data,
            var_names=[
                variable_mappings.get("mu_beta_language"),
                variable_mappings.get("sigma_beta_language")
            ],
            colors="C2",
            hdi_prob=0.95,
            ax=axes[1]
        )
        axes[1].set_title("Language Parameters")

        # Plot task parameters: mu_beta_task, sigma_beta_task, and beta_task_std
        az.plot_forest(
            cmdstanpy_data,
            var_names=[
                variable_mappings.get("mu_beta_task"),
                variable_mappings.get("sigma_beta_task")
            ],
            colors="C3",
            hdi_prob=0.95,
            ax=axes[2]
        )
        axes[2].set_title("Task Parameters")

        # Plot phi parameters: phi_alpha and beta_task_phi
        az.plot_forest(
            cmdstanpy_data,
            var_names=[
                variable_mappings.get("phi_alpha"),
                variable_mappings.get("beta_task_phi")
            ],
            colors="C4",
            hdi_prob=0.95,
            ax=axes[3]
        )
        axes[3].set_title(r"$\phi$ Parameters")

        plt.tight_layout()
        # Save the plot to file
        output_file = os.path.join(self.output_path_figures, "posterior_distribution_plots.png")
        plt.savefig(output_file)
        plt.close()


    def rank_comparison(self, rank_path, output_path_data, output_path_figures):
        
        # read summary csv
        df = pd.read_csv(os.path.join(output_path_data, "overview/summary.csv"))
        df = df[df["Unnamed: 0"].str.contains("alpha_std")]
        # only keep columns Unnamed: 0, Mean, 5%, 50%, 95%
        df = df[["Unnamed: 0", "Mean", "5%", "50%", "95%"]]
        # extract the number in the Unnamed: 0 column
        df["model_id"] = df["Unnamed: 0"].str.extract(r'(\d+)').astype(int)

        # read in model mapping
        model_mapping = pd.read_csv(os.path.join(self.output_path_data, "model_mapping.csv"))
        # from model extract the last string after a / and before </a>
        model_mapping["model_name"] = model_mapping["model"].str.extract(r'\/([^\/]+)<\/a>')[0]
        # join df and model_mapping on model_id
        df = df.merge(model_mapping[["model_id", "model_name"]], on="model_id")
        # sort according to mean
        df = df.sort_values(by="Mean", ascending=False)
        df.reset_index(drop=True, inplace=True)

        df_euro = pd.read_csv(rank_path)
        df_euro = df_euro[~df_euro['model'].str.contains('zero-shot', na=False)]
        df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot\)', '', regex=True)
        df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot, val\)', '', regex=True)
        df_euro["model_name"] = df_euro["model"].str.extract(r'\/([^\/]+)<\/a>')[0]
        df_euro_100 = df_euro.nsmallest(100, 'rank')
        df_euro_100.reset_index(drop=True, inplace=True)

        list1 = df_euro_100["model_name"]  # Your first ranked list
        list2 = df["model_name"]  # Your second ranked list

        rank1 = {item: idx for idx, item in enumerate(list1)}
        rank2 = {item: idx for idx, item in enumerate(list2)}

        x = [rank1[item] for item in list1]
        y = [rank2[item] for item in list1]

        plt.scatter(x, y)
        plt.plot([0, 100], [0, 100], 'r--')  # Ideal diagonal
        plt.xlabel("Rank on EuroEval")
        plt.ylabel("Rank according to Model")
        plt.title("Rank Comparison")

        # Save the plot to file
        output_file = os.path.join(output_path_figures, "rank_comparison.png")
        plt.savefig(output_file)
        plt.close()