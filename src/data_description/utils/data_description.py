import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

class DataDescription():
    def __init__(self, output_path_figures, output_path_data):
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

    def preprocess_data(self, top_n = None, top_n_path = None):
        """
        Preprocess the loaded data.
        """
        df_full = self.data

        # remove task == speed
        remove_list = ['test_speed', 'test_speed_short', 'test_runtime', 'test_samples_per_second', 'test_steps_per_second', 'epoch', 'test_micro_f1_no_misc', 'test_micro_f1', 'test_em', 'test_f1', 'test_accuracy', 'test_loss', 'test_mcc', 'test_macro_f1']
        df_full = df_full[~df_full['metric'].isin(remove_list)]

        # remove few_shot == False
        df_full = df_full[df_full['few_shot'] == True]

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

        # Change languages from abbreviation to full language name
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

        if top_n is not None:
            df_euro = pd.read_csv(top_n_path)
            df_euro = df_euro[~df_euro['model'].str.contains('zero-shot', na=False)]
            df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot\)', '', regex=True)
            df_euro['model'] = df_euro['model'].str.replace(r' \(few-shot, val\)', '', regex=True)
            df_euro_100 = df_euro.nsmallest(top_n, 'rank')
            df_full = df_full[df_full['model'].isin(df_euro_100['model'].tolist())]

        df_full['language'] = df_full['language'].replace(language_map)

        print(f"Unique tasks: {df_full['task'].unique()}")
        print(f"Unique metrics: {df_full['metric'].unique()}")
        print(f"Unique languages: {df_full['language'].unique()}")
        print(f"Number of unique models: {len(df_full['model'].unique())}")
        print(f"Number of observations: {len(df_full)}")
        
        self.data = df_full

        # save the unique models to a latex file
        unique_models = pd.DataFrame(sorted(self.data['model'].unique()), columns=['model'])
        # extract the mode link that is inside ''
        unique_models['model_link'] = unique_models['model'].str.extract(r"'(.*?)'", expand=False)
        # remove the HTML tags by extracting the text between > and </a>
        unique_models['model'] = unique_models['model'].str.extract(r"<a href='.*'>(.*?)</a>", expand=False).fillna(unique_models['model'])
        output_path = os.path.join(self.output_path_data, "unique_models.txt")
        with open(output_path, "w") as txt_file:
            txt_file.write(unique_models.to_latex(index=False, caption="Unique Models", label="tab:unique_models"))
        print(f"Unique models saved to {output_path}")

    def generate_data_description(self):
        """
        Generate a description of the data including means and standard deviations across languages and tasks and save it to a file.
        """
        # Group by language and task, then calculate mean and std for each group
        grouped_language = self.data.groupby(['language']).agg(
            mean_value=('value', 'mean'),
            std_value=('value', 'std'),
            count=('value', 'count')
        ).reset_index().sort_values(by='mean_value')

        grouped_task = self.data.groupby(['task']).agg(
            mean_value=('value', 'mean'),
            std_value=('value', 'std'),
            count=('value', 'count')
        ).reset_index().sort_values(by='mean_value')

        # Save the grouped data to a CSV file
        output_path = os.path.join(self.output_path_data, "data_description_language.csv")
        grouped_language.to_csv(output_path, index=False)
        output_path_txt = os.path.join(self.output_path_data, "data_description_language.txt")
        with open(output_path_txt, "w") as txt_file:
            txt_file.write(grouped_language.to_latex(index=False, caption="Language Data Description", label="tab:language_data"))
        print(f"Data description by language saved to {output_path_txt}")

        print(f"Data description by language saved to {output_path}")
        output_path = os.path.join(self.output_path_data, "data_description_task.csv")
        grouped_task.to_csv(output_path, index=False)
        output_path_txt = os.path.join(self.output_path_data, "data_description_task.txt")
        with open(output_path_txt, "w") as txt_file:
            txt_file.write(grouped_task.to_latex(index=False, caption="Task Data Description", label="tab:task_data"))
        print(f"Data description by task saved to {output_path}")

    def plot_histograms(self, group_by: str):
        df = self.data.copy()

        # Group the data by the specified column
        groups = list(df.groupby([group_by]))
        n_plots = len(groups)
        ncols = 3  # You can adjust the number of columns as needed
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()

        for ax, (val, group) in zip(axes, groups):
            group['value'].plot.hist(bins=30, edgecolor='black', ax=ax)
            ax.set_title(f"{group_by.capitalize()}: {val[0].capitalize()}", fontsize=16)
            ax.set_xlabel("Value", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            ax.set_xlim(0, 1)

        # Remove unused subplots if there are any
        for ax in axes[len(groups):]:
            ax.remove()

        # Add an overall title to the full figure
        fig.suptitle(f"Histograms by {group_by.capitalize()}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(self.output_path_figures, f"histograms_{group_by}.png")
        plt.savefig(output_path)
        plt.close()
    
    def plot_histograms_model_counts(self):

        model_counts = self.data['model'].value_counts()
        plt.hist(model_counts, bins=20, edgecolor='black')
        plt.title("Model Observations Histogram", fontsize=20)
        plt.xlabel("Number of Observations", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # Save the figure
        output_path = os.path.join(self.output_path_figures, "model_observations_histogram.png")
        plt.savefig(output_path)
        plt.close()