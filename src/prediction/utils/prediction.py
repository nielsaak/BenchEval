import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from cmdstanpy import CmdStanModel
import cmdstanpy
import numpy as np
import arviz as az

class Prediction():
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
        self.language_map = {
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

    def _iterative_data_generation(self, df):
        """
        Run preprocess data with different combinations of langauge and task removed
        """

        # Get the unique languages and tasks
        unique_languages = df['language'].unique()
        unique_tasks = df['task'].unique()

        # Create a list to store the dataframes
        dfs = []

        # Iterate over each language and task
        # for language in unique_languages:
        for language in ["en", "da", "is"]:
            for task in unique_tasks:
                # Filter the dataframe for the current language and task
                df_train = df[~((df['language'] == language) & (df['task'] == task))]
                df_test = df[(df['language'] == language) & (df['task'] == task)]
                
                
                # Append the filtered dataframe to the list
                dfs.append({"train": df_train, "test": df_test})

                # print unique combination of language and task in df_train and df_test
                
        # dfs[0]["train"].to_csv(os.path.join(self.output_path_data, "train.csv"), index=False)
        # dfs[0]["test"].to_csv(os.path.join(self.output_path_data, "test.csv"), index=False)

        print("Iterative data generation completed.")
        # print len of list
        print(f"Number of dataframes generated: {len(dfs)}")

        return dfs

    def prepare_data_for_stan(self):
        """
        Prepare data for Stan model.
        """

        df = self.data.copy()

        # if value == 0 add 1e-5
        df['value'] = df['value'].replace(0, 1e-5)
        # if value == 1 subtract 1e-5
        df['value'] = df['value'].replace(1, 1 - 1e-5)

        # change name of models to numbers starting from 1
        df['model'] = df['model'].map({name: i + 1 for i, name in enumerate(df['model'].unique())})
        # change name of tasks to numbers starting from 1
        df['task_id'] = df['task'].map({name: i + 1 for i, name in enumerate(df['task'].unique())})
        # change name of languages to numbers starting from 1
        df['language_id'] = df['language'].map({name: i + 1 for i, name in enumerate(df['language'].unique())})

        # save number of unique languages and tasks
        n_languages = len(df['language'].unique())
        n_tasks = len(df['task'].unique())


        dfs = self._iterative_data_generation(df)

        # Create a list to store the standata
        stan_data_list = []

        for i in range(len(dfs)):

            df = dfs[i]['train']
            df_test = dfs[i]['test']

            # Convert the DataFrame to a dictionary format suitable for Stan
            stan_data = {
                'N': df.shape[0],
                'M': len(df['model'].unique()),
                'n_language': n_languages,
                'n_task': n_tasks,
                'group': df['model'].tolist(),
                'language': df['language_id'].tolist(),
                'task': df['task_id'].tolist(),
                'y': df['value'].tolist(),
                'N_test': df_test.shape[0],
                'group_test': df_test['model'],
                'language_test': df_test['language_id'].tolist(),
                'task_test': df_test['task_id'].tolist(),
                'y_test': df_test['value'].tolist()
            }

            # Append the stan_data to the list
            stan_data_list.append({'stan_data': stan_data,
                                   'language': df_test['language'].unique(),
                                   'task': df_test['task'].unique(),
                                   'task_id': df_test['task_id'].unique(),})

            model_mapping = {name: i + 1 for i, name in enumerate(df['model'].unique())}
            task_mapping = {name: i + 1 for i, name in enumerate(df['task'].unique())}
            language_mapping = {name: i + 1 for i, name in enumerate(df['language'].unique())}

            # get length of each element in stan_data
            # for key, value in stan_data.items():
            #     if isinstance(value, list):
            #         print(f"{key}: {len(value)}")
            #     else:
            #         print(f"{key}: {value}")

            print(f"Stan data prepared for prediction on {df_test['language'].unique()} and {df_test['task'].unique()}.")

        print(f"Stan data prepared for {len(dfs)} iterations.")

        self.stan_data_list = stan_data_list
        self.model_mapping = model_mapping
        self.task_mapping = task_mapping
        self.language_mapping = language_mapping
    
    def make_predictions(self,
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

        # limit number of elements in stan_data_list to 2
        # self.stan_data_list = self.stan_data_list[:4]

        # dict to store mae
        mae_dict = {}

        for i, stan_data in enumerate(self.stan_data_list):

            # Create a directory for the output data if it doesn't exist
            os.makedirs(output_path_data, exist_ok=True)

            # Create a directory for the output figures if it doesn't exist
            os.makedirs(output_path_figures, exist_ok=True)

            print(f"Stan data prepared for prediction on {stan_data['language']} and {stan_data['task']}.")

            # Compile the Stan model using cmdstanpy.
            try:
                model = CmdStanModel(stan_file=stan_file)

                fit = model.sample(
                        data=stan_data['stan_data'],
                        **model_fit_params,
                    )
                
                print("Fitting completed.")

                # print("Generate Summary...")

                # Generate summary statistics
                # summary = fit.summary()
                # summary_df = pd.DataFrame(summary)
                # summary_df.to_csv(os.path.join(output_path_data, f"summary_{stan_data['language']}_{stan_data['task']}.csv"))

                print("Generating prediction plot...")

                # df_full['language'] = df_full['language'].replace(language_map)

                stan_data_language = self.language_map[stan_data['language'][0]]

                diff = fit.y_pred_test.mean(axis=0) - stan_data['stan_data']["y_test"]
                plt.figure()
                plt.scatter(np.arange(len(diff)), diff)
                plt.axhline(0, color='red', linestyle='--')
                plt.xlabel("Index")
                plt.ylabel("Difference")
                plt.title(f"Prediction Differences for: {stan_data['task'][0]} in {stan_data_language}")
                plt.savefig(os.path.join(output_path_figures, f"prediction_difference_{stan_data['language'][0]}_{stan_data['task'][0]}.png"))
                plt.close()

                print("Plot saved.")
                print("Calculating Mean Absolute Error...")

                mae = np.mean(np.abs(diff))
                print(f"Mean Absolute Error: {mae}")

                # save mae to dict
                mae_dict[f"{stan_data['language']}_{stan_data['task']}"] = mae
            
            except Exception as e:
                print(f"Error fitting model: {e}")
                continue
        
        # create a new dict that has all languages as keys and a list of mae values as values
        mae_language_dict = {}
        mae_task_dict = {}
        for key, value in mae_dict.items():
            language, task = key.split('_')
            if language not in mae_language_dict:
                mae_language_dict[language] = []
            if task not in mae_task_dict:
                mae_task_dict[task] = []
            mae_language_dict[language].append(value)
            mae_task_dict[task].append(value)
        # calculate mean mae for each language
        # for key, value in mae_language_dict.items():
        #     mae_language_dict[key] = np.mean(value)
        # # calculate mean mae for each task
        # for key, value in mae_task_dict.items():
        #     mae_task_dict[key] = np.mean(value)

        # save mae dict to csv
        mae_df = pd.DataFrame(mae_dict.items(), columns=['language_task', 'mae'])
        mae_df.to_csv(os.path.join(output_path_data, "mae.csv"), index=False)

        # save grand average mae to csv
        grand_average_mae = np.mean(list(mae_dict.values()))
        grand_average_mae_df = pd.DataFrame({'grand_average_mae': [grand_average_mae]})
        grand_average_mae_df.to_csv(os.path.join(output_path_data, "grand_average_mae.csv"), index=False)

        # save mae dict for language to csv
        mae_language_df = pd.DataFrame(mae_language_dict.items(), columns=['language', 'mae'])
        mae_language_df.to_csv(os.path.join(output_path_data, "mae_language.csv"), index=False)

        # save mae dict for task to csv
        mae_task_df = pd.DataFrame(mae_task_dict.items(), columns=['task', 'mae'])
        mae_task_df.to_csv(os.path.join(output_path_data, "mae_task.csv"), index=False)

        # calculate mean mae for each language
        for key, value in mae_language_dict.items():
            mae_language_dict[key] = np.mean(value)
        # calculate mean mae for each task
        for key, value in mae_task_dict.items():
            mae_task_dict[key] = np.mean(value)
        # save mae dict to csv
        mae_language_df = pd.DataFrame(mae_language_dict.items(), columns=['language', 'mae'])
        mae_language_df.to_csv(os.path.join(output_path_data, "mae_language_mean.csv"), index=False)
        mae_task_df = pd.DataFrame(mae_task_dict.items(), columns=['task', 'mae'])
        mae_task_df.to_csv(os.path.join(output_path_data, "mae_task_mean.csv"), index=False)



        pass

    def baseline_predictions(self, output_path_data, output_path_figures):
        """
        Generate baseline predictions.
        """

        # Get the unique languages and tasks
        unique_languages = self.data['language'].unique()
        unique_tasks = self.data['task'].unique()
        # Get the unique models
        unique_models = self.data['model'].unique()

        # calculate mean for each task and use for baseline prediction
        # create a dict to store the mean for each task
        task_means = {}


        mae_dict = {}

        for i, stan_data in enumerate(self.stan_data_list):

            # Create a directory for the output data if it doesn't exist
            os.makedirs(output_path_data, exist_ok=True)

            # Create a directory for the output figures if it doesn't exist
            os.makedirs(output_path_figures, exist_ok=True)

            print(f"Stan data prepared for prediction on {stan_data['language']} and {stan_data['task']}.")

            # Compile the Stan model using cmdstanpy.
            try:
                
                df = pd.DataFrame({
                    'y': stan_data['stan_data']["y"],
                    'group': stan_data['stan_data']["group"],
                    'task': stan_data['stan_data']["task"],
                })

                # 1) filter to only the "jumping" rows
                df_task = df[df['task'] == stan_data['task_id'][0]]

                # 2) group by participant and take the mean of y
                mean_per_llm = df_task.groupby('group')['y'].mean()

                # print(f"Mean per LLM for {stan_data['task'][0]}: {mean_per_llm}")

                # print("Generating prediction plot...")

                stan_data_language = self.language_map[stan_data['language'][0]]

                df_test = pd.DataFrame({
                    'y_test': stan_data['stan_data']["y_test"],
                    'group_test': stan_data['stan_data']["group_test"],
                })

                # 3) merge the two dataframes on group
                df_test = df_test.merge(mean_per_llm, left_on='group_test', right_index=True, how='left')

                diff = df_test['y'] - df_test['y_test']
                plt.figure()
                plt.scatter(np.arange(len(diff)), diff)
                plt.axhline(0, color='red', linestyle='--')
                plt.xlabel("Index")
                plt.ylabel("Difference")
                plt.title(f"Prediction Differences for: {stan_data['task'][0]} in {stan_data_language}")
                plt.savefig(os.path.join(output_path_figures, f"prediction_difference_{stan_data['language'][0]}_{stan_data['task'][0]}.png"))
                plt.close()

                print("Plot saved.")
                print("Calculating Mean Absolute Error...")

                mae = np.mean(np.abs(diff))
                print(f"Mean Absolute Error: {mae}")

                # save mae to dict
                mae_dict[f"{stan_data['language']}_{stan_data['task']}"] = mae
            
            except Exception as e:
                print(f"Error fitting model: {e}")
                continue
            
         # create a new dict that has all languages as keys and a list of mae values as values
        mae_language_dict = {}
        mae_task_dict = {}
        for key, value in mae_dict.items():
            language, task = key.split('_')
            if language not in mae_language_dict:
                mae_language_dict[language] = []
            if task not in mae_task_dict:
                mae_task_dict[task] = []
            mae_language_dict[language].append(value)
            mae_task_dict[task].append(value)
        # calculate mean mae for each language
        # for key, value in mae_language_dict.items():
        #     mae_language_dict[key] = np.mean(value)
        # # calculate mean mae for each task
        # for key, value in mae_task_dict.items():
        #     mae_task_dict[key] = np.mean(value)

        # save mae dict to csv
        mae_df = pd.DataFrame(mae_dict.items(), columns=['language_task', 'mae'])
        mae_df.to_csv(os.path.join(output_path_data, "mae.csv"), index=False)

        # save grand average mae to csv
        grand_average_mae = np.mean(list(mae_dict.values()))
        grand_average_mae_df = pd.DataFrame({'grand_average': "True", 'grand_average_mae': [grand_average_mae]})
        grand_average_mae_df.to_csv(os.path.join(output_path_data, "grand_average_mae.csv"), index=False)


        # save mae dict for language to csv
        mae_language_df = pd.DataFrame(mae_language_dict.items(), columns=['language', 'mae'])
        mae_language_df.to_csv(os.path.join(output_path_data, "mae_language.csv"), index=False)

        # save mae dict for task to csv
        mae_task_df = pd.DataFrame(mae_task_dict.items(), columns=['task', 'mae'])
        mae_task_df.to_csv(os.path.join(output_path_data, "mae_task.csv"), index=False)

        # calculate mean mae for each language
        for key, value in mae_language_dict.items():
            mae_language_dict[key] = np.mean(value)
        # calculate mean mae for each task
        for key, value in mae_task_dict.items():
            mae_task_dict[key] = np.mean(value)
        # save mae dict to csv
        mae_language_df = pd.DataFrame(mae_language_dict.items(), columns=['language', 'mae'])
        mae_language_df.to_csv(os.path.join(output_path_data, "mae_language_mean.csv"), index=False)
        mae_task_df = pd.DataFrame(mae_task_dict.items(), columns=['task', 'mae'])
        mae_task_df.to_csv(os.path.join(output_path_data, "mae_task_mean.csv"), index=False)



    def comparison(self, output_path_data):
        # Create a directory for the output data if it doesn't exist
        os.makedirs(os.path.join(output_path_data, "comparison"), exist_ok=True)

        # load in the previous mae results and calculate difference between 'thesis' and 'baseline'
        list_of_csv_names = ["mae.csv", "mae_language_mean.csv", "mae_task_mean.csv", "grand_average_mae.csv"]
        # create a dict to store the dataframes
        dataframes_thesis = {}
        # create a dict to store the dataframes
        dataframes_baseline = {}
        # load in the dataframes
        for csv_name in list_of_csv_names:
            dataframes_thesis[csv_name] = pd.read_csv(os.path.join(os.path.join(output_path_data, "thesis"), csv_name))
            dataframes_baseline[csv_name] = pd.read_csv(os.path.join(os.path.join(output_path_data, "baseline"), csv_name))
        # create a dict to store the differences
        differences = {}
        # calculate the differences
        for csv_name in list_of_csv_names:
            # merge the two dataframes on the first column
            df = pd.merge(
                dataframes_thesis[csv_name],
                dataframes_baseline[csv_name],
                on=dataframes_thesis[csv_name].columns[0],
                suffixes=('_thesis', '_baseline')
            )
            # calculate the difference between the two columns
            if csv_name == "grand_average_mae.csv":
                print(df)
                df['difference'] = df['grand_average_mae_thesis'] - df['grand_average_mae_baseline']
            else:
                df['difference'] = df.iloc[:, 1] - df.iloc[:, 2]
            # save the dataframe to a csv file
            df.to_csv(os.path.join(output_path_data, f"comparison/difference_{csv_name}"), index=False)

        return