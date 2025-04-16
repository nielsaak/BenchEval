import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataSimulation():
    def __init__(self):
        self.num_samples = 0
        self.num_features = 0
        self.noise_level = 0.0
        self.data = None

    def set_parameters(self, n_tasks, n_lang, n_models, n_repetitions, n_metrics, noise_level):
        # insert specification of hyperparameters here that are relevant for the data simulation
        self.n_tasks = n_tasks
        self.n_lang = n_lang
        self.n_models = n_models
        self.n_repetitions = n_repetitions
        self.n_metrics = n_metrics
        self.noise_level = noise_level

    def generate_data_single(self):
        # insert code for generating data here
        import numpy as np
        import pandas as pd
        from scipy.special import expit as inv_logit
        from scipy.stats import beta, norm

        # Settings
        N = 1000           # Total observations
        M = 5              # Number of groups
        n_language = 4
        n_metric = 3
        n_task = 6

        # Hyperparameters (true values)
        mu_alpha = 0.0
        sigma_alpha = 1.0

        mu_beta_language = np.random.normal(0, 0.5, n_language)
        sigma_beta_language = 0.3

        mu_beta_metric = np.random.normal(0, 0.5, n_metric)
        sigma_beta_metric = 0.3

        mu_beta_task = np.random.normal(0, 0.5, n_task)
        sigma_beta_task = 0.3

        mu_phi = 10.0
        sigma_phi = 2.0

        # Sample group-level parameters
        alpha = np.random.normal(mu_alpha, sigma_alpha, M)
        beta_language = np.random.normal(mu_beta_language, sigma_beta_language, (M, n_language))
        beta_metric = np.random.normal(mu_beta_metric, sigma_beta_metric, (M, n_metric))
        beta_task = np.random.normal(mu_beta_task, sigma_beta_task, (M, n_task))
        phi = np.random.normal(mu_phi, sigma_phi, M)

        # Sample observations
        data = []
        for i in range(N):
            g = np.random.randint(0, M)  # group index (0-based)
            l = np.random.randint(0, n_language)
            m = np.random.randint(0, n_metric)
            t = np.random.randint(0, n_task)

            eta = alpha[g] + beta_language[g, l] + beta_metric[g, m] + beta_task[g, t]
            mu = inv_logit(eta)
            a = mu * phi[g]
            b = (1 - mu) * phi[g]
            y = beta.rvs(a, b)

            data.append({
                'group': g + 1,
                'language': l + 1,
                'metric': m + 1,
                'task': t + 1,
                'y': y
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df.head())

        return data
    
    def generate_data_multi(self):
        # insert code for generating data here
        data = None

        self.data = data
        return data

    def generate_figures(self, filename: str):
        # insert code for relevant plots here
        return

    def save_data(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)