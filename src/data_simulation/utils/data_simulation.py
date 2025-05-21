import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
import itertools
import pickle
import os

class DataSimulation():
    def __init__(self):
        self.parameters = None
        self.data = None

    def set_parameters(self, L, K_1, K_2, n_repetitions):
        def _generate_zero_sum_array(n, m=0):
            """
            Generate an array of length n (or matrix of shape (m, n)) whose rows sum to zero.
            """
            if m == 0:
                if n == 1:
                    return np.array([0.])
                arr = np.random.uniform(-1, 1, size=n-1)
                return np.append(arr, -arr.sum())
            else:
                arr = np.random.uniform(-1, 1, size=(m, n-1))
                last_col = -arr.sum(axis=1, keepdims=True)
                return np.hstack([arr, last_col])

        # insert specification of hyperparameters here that are relevant for the data simulation
        self.parameters = {
            "mu_alpha": [-0.5,-0.2,-0.1,0.0,0.1,0.2,0.5],
            "sigma_alpha": [0.1, 0.2, 0.5, 0.7, 1.0],
            "alpha_std": np.random.normal(0, 1, size=L),

            "mu_beta_language": _generate_zero_sum_array(K_1),
            "mu_beta_task": _generate_zero_sum_array(K_2),
            "sigma_beta_language": np.random.lognormal(-1, 1, size=(K_1)),
            "sigma_beta_task": np.random.lognormal(-1, 1, size=(K_2)),
            "beta_language_std": _generate_zero_sum_array(K_1, L),
            "beta_task_std": _generate_zero_sum_array(K_2, L),

            "phi_alpha": [0.1, 1, 2, 3, 4, 5, 10],
            "beta_task_phi": _generate_zero_sum_array(K_2),
        }

        self.L = L
        self.K_1 = K_1
        self.K_2 = K_2
        self.n_repetitions = n_repetitions
        
    def simulate_from_params(self):
        """
        Simulate beta-distributed data based on the provided parameter dictionary.

        params: dict
            A dictionary containing keys for:
            - 'alpha_mu_mu', 'alpha_mu_sigma', 'alpha_mu_std'
            - 'beta_mu_mu_k_1', 'beta_mu_mu_k_2', 'beta_mu_sigma_1', 'beta_mu_sigma_2',
                'beta_mu_std_1', 'beta_mu_std_2'
            - 'alpha_phi', 'beta_phi'
            Some values in params may be lists; those will be varied in a Cartesian product.
        K1, K2, L: int
            Dimensions for language (K1), metric (K2), and number of participants (L).
        n_repetitions: int
            Number of repetitions per participant.

        Returns
        -------
        results: list of dicts
            Each dict contains:
            - 'param_combo': dict, the hyperparameter values used
            - 'data_dict': dict, containing N, K, K1, K2, J, L, X_subject, Z, idx, y
            - 'fig': matplotlib.figure.Figure, histogram of y
        """
        # Unpack parameters
        params = self.parameters
        L = self.L
        K1 = self.K_1
        K2 = self.K_2
        n_repetitions = self.n_repetitions

        # Identify keys to loop over (those with list-like values)
        loop_keys = [k for k, v in params.items() if isinstance(v, (list, np.ndarray)) and not isinstance(v, np.ndarray) or isinstance(v, list)]
        # Build list of values for product
        loop_values = [params[k] for k in loop_keys]
        results = []
        iter = 0

        for combo in itertools.product(*loop_values):
            # Build a dict of current hyperparameters
            combo_dict = params.copy()
            for k, val in zip(loop_keys, combo):
                combo_dict[k] = val

            # Unpack hyperparameters
            mu_alpha = combo_dict['mu_alpha']
            sigma_alpha = combo_dict['sigma_alpha']
            alpha_std = np.asarray(combo_dict['alpha_std'])
            mu_beta_language = np.asarray(combo_dict['mu_beta_language'])
            mu_beta_task = np.asarray(combo_dict['mu_beta_task'])
            sigma_beta_language = np.asarray(combo_dict['sigma_beta_language'])
            sigma_beta_task = np.asarray(combo_dict['sigma_beta_task'])
            beta_language_std = np.asarray(combo_dict['beta_language_std'])
            beta_task_std = np.asarray(combo_dict['beta_task_std'])
            phi_alpha = combo_dict['phi_alpha']
            beta_task_phi = np.asarray(combo_dict['beta_task_phi'])

            # Build data frame
            df = pd.DataFrame(
                [(participant, ti, c, a)
                for participant in range(L)
                for ti in range(n_repetitions)
                for c in range(K1)
                for a in range(K2)],
                columns=['model', 'n_repetitions', 'language', 'task']
            )

            # Compute population parameters
            alpha_mu = mu_alpha + sigma_alpha * alpha_std
            beta_mu_1 = mu_beta_language + sigma_beta_language * beta_language_std
            beta_mu_2 = mu_beta_task + sigma_beta_task * beta_task_std

            # Index by participant
            beta_mu_1_idx = beta_mu_1[df['model']]
            beta_mu_2_idx = beta_mu_2[df['model']]
            alpha_mu_idx = alpha_mu[df['model']]

            # Linear predictors and beta draws
            eta = [alpha_mu_idx[i] + beta_mu_1_idx[i][df["language"][i]] + beta_mu_2_idx[i][df["task"][i]] + np.random.normal(0,0.5) for i in range(df.shape[0])]
        
            mu = expit(eta)
            eta_phi = [phi_alpha + beta_task_phi[df["task"][i]] + np.random.normal(0,0.5) for i in range(df.shape[0])]
            phi = np.exp(eta_phi)
            phi = np.clip(phi, 1e-5, None)
            A = mu * phi
            B = (1 - mu) * phi
            y = beta.rvs(A, B)

            # Package data
            data_dict = {
                'N': df.shape[0],
                'M': L,
                'n_language': K1,
                'n_task': K2,
                'group': df['model'] + 1,
                'language': df['language'] + 1,
                'task': df['task'] + 1,
                'y': y
            }

            # Plot histogram
            # fig, ax = plt.subplots()
            # ax.hist(y, bins=20, edgecolor='black')
            # ax.set_title(f"Histogram of y (combo: {dict(zip(loop_keys, combo))})")
            # ax.set_xlabel('y')
            # ax.set_ylabel('Frequency')

            results.append({'param_combo': combo_dict,
                            'data_dict': data_dict})
            # plt.close(fig)

            iter += 1
            if iter % 10 == 0:
                print(f"Completed {iter} iterations.")

        self.results = results

        return
    
    def save_data(self, path):
        # Save to a file
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(self.results, f)