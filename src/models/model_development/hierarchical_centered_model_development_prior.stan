
data {
  int<lower=0> N;                       // Number of observations
  int<lower=0> M;                       // Number of groups/models
  int<lower=0> n_language;              // Number of languages
  int<lower=0> n_task;                  // Number of tasks

  array[N] int<lower=1,upper=M>       group;     // Group index per observation
  array[N] int<lower=1,upper=n_language> language; // Language index per observation
  array[N] int<lower=1,upper=n_task>    task;     // Task index per observation
}

generated quantities {

  real mu_alpha    = normal_rng(0, 2);
  real<lower=0> sigma_alpha = exponential_rng(1);

  vector[n_language] mu_beta_language_unc;
  for (j in 1:n_language) {
    mu_beta_language_unc[j] = normal_rng(0, 2);
  }

  real mean_mu_beta_language = mean(mu_beta_language_unc);
  vector[n_language] mu_beta_language;
  for (j in 1:n_language) {
    mu_beta_language[j] = mu_beta_language_unc[j] - mean_mu_beta_language;
  }

  vector[n_language] sigma_beta_language;
  for (j in 1:n_language) {
    sigma_beta_language[j] = exponential_rng(1);
  }


  vector[n_task] mu_beta_task_unc;
  for (t in 1:n_task) {
    mu_beta_task_unc[t] = normal_rng(0, 2);
  }

  real mean_mu_beta_task = mean(mu_beta_task_unc);
  vector[n_task] mu_beta_task;
  for (t in 1:n_task) {
    mu_beta_task[t] = mu_beta_task_unc[t] - mean_mu_beta_task;
  }

  vector[n_task] sigma_beta_task;
  for (t in 1:n_task) {
    sigma_beta_task[t] = exponential_rng(1);
  }

  real phi_alpha = normal_rng(0, 2);

  vector[n_task] beta_task_phi_unc;
  for (t in 1:n_task) {
    beta_task_phi_unc[t] = normal_rng(0, 2);
  }
  real mean_beta_task_phi = mean(beta_task_phi_unc);
  vector[n_task] beta_task_phi;
  for (t in 1:n_task) {
    beta_task_phi[t] = beta_task_phi_unc[t] - mean_beta_task_phi;
  }

  vector[M] alpha_std;
  for (m in 1:M) {
    alpha_std[m] = normal_rng(mu_alpha, sigma_alpha);
  }

  array[M] vector[n_language] beta_language_std_unc;
  array[M] vector[n_language] beta_language_std; 
  for (m in 1:M) {
    for (j in 1:n_language) {
      beta_language_std_unc[m][j] = normal_rng(mu_beta_language[j],
                                               sigma_beta_language[j]);
    }
    // Center row m:
    real row_mean_lang = mean(beta_language_std_unc[m]);
    for (j in 1:n_language) {
      beta_language_std[m][j] = beta_language_std_unc[m][j] - row_mean_lang;
    }
  }

  array[M] vector[n_task] beta_task_std_unc;
  array[M] vector[n_task] beta_task_std;
  for (m in 1:M) {
    for (t in 1:n_task) {
      beta_task_std_unc[m][t] = normal_rng(mu_beta_task[t],
                                           sigma_beta_task[t]);
    }

    real row_mean_task = mean(beta_task_std_unc[m]);
    for (t in 1:n_task) {
      beta_task_std[m][t] = beta_task_std_unc[m][t] - row_mean_task;
    }
  }

  vector[N] y_pred;
  for (i in 1:N) {
    int   g_i = group[i];
    int   lang_i = language[i];
    int   t_i = task[i];

    real eta_mu_i = alpha_std[g_i]
                    + beta_language_std[g_i][lang_i]
                    + beta_task_std[g_i][t_i];

    real mu_i = inv_logit(eta_mu_i);

    real eta_phi_i = phi_alpha + beta_task_phi[t_i];
    real phi_i     = exp(eta_phi_i);

    real a_i = mu_i * phi_i;
    real b_i = (1 - mu_i) * phi_i;

    y_pred[i] = beta_rng(a_i, b_i);
  }
}
