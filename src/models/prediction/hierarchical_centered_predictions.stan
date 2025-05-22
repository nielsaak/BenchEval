data {
  int<lower=0> N;                       // Number of observations
  int<lower=0> M;                       // Number of groups/models
  int<lower=0> n_language;              // Number of languages
  int<lower=0> n_task;                  // Number of tasks

  array[N] int<lower=1,upper=M> group;  // Group indicator for each observation
  array[N] int<lower=1,upper=n_language> language; // Language indicator for each observation
  array[N] int<lower=1,upper=n_task> task; // Task indicator for each observation

  vector<lower=0,upper=1>[N] y; // Output value for each observation

  // include test data
  int<lower=0> N_test;                       // Number of test observations
  // int<lower=0> M_test;                       // Number of test groups/models
  // int<lower=0> n_language_test;              // Number of test languages
  // int<lower=0> n_task_test;                  // Number of test tasks
  array[N_test] int<lower=1,upper=M> group_test;  // Group indicator for each test observation
  array[N_test] int<lower=1,upper=n_language> language_test; // Language indicator for each test observation
  array[N_test] int<lower=1,upper=n_task> task_test; // Task indicator for each test observation
  // vector<lower=0,upper=1>[N_test] y_test; // Output value for each test observation
}

parameters {
  // Hyperpriors
  real mu_alpha;
  real<lower=0> sigma_alpha;

  sum_to_zero_vector[n_language] mu_beta_language;
  vector<lower=0>[n_language] sigma_beta_language;

  sum_to_zero_vector[n_task] mu_beta_task;
  vector<lower=0>[n_task] sigma_beta_task;

  // Group-level parameters
  vector[M] alpha_std;
  array[M] sum_to_zero_vector[n_language] beta_language_std;
  array[M] sum_to_zero_vector[n_task] beta_task_std;

  // Phi parameters
  real phi_alpha;
  sum_to_zero_vector[n_task] beta_task_phi;
}

transformed parameters {
  // vector[M] alpha = mu_alpha + sigma_alpha * alpha_std;
  // array[M] row_vector[n_language] beta_language;
  // array[M] row_vector[n_task] beta_task;

  // for (m in 1:M) {
  //   for (k in 1:n_language) {
  //     beta_language[m,k] = mu_beta_language[k] + sigma_beta_language[k] * beta_language_std[m,k];
  //   }
  //   for (k in 1:n_task) {
  //     beta_task[m,k] = mu_beta_task[k] + sigma_beta_task[k] * beta_task_std[m,k];
  //   }
  // }

  vector[N] eta_mu;
  for (i in 1:N) {
    eta_mu[i] = alpha_std[group[i]]
                + beta_language_std[group[i], language[i]]
                + beta_task_std[group[i], task[i]];
  }
  vector[N] mu = inv_logit(eta_mu);

  vector[N] eta_phi;
  vector[N] beta_task_phi_idx = beta_task_phi[task];
  eta_phi = phi_alpha + beta_task_phi_idx;
  vector[N] phi = exp(eta_phi);
}

model {
  beta_task_phi ~ normal(0, 2);
  phi_alpha ~ normal(0, 2);

  // Hyperpriors
  mu_alpha ~ normal(0, 2);
  sigma_alpha ~ exponential(1);
  alpha_std ~ normal(mu_alpha, sigma_alpha);

  mu_beta_language ~ normal(0, 2);
  sigma_beta_language ~ exponential(1);

  mu_beta_task ~ normal(0, 2);
  sigma_beta_task ~ exponential(1);


  for (m in 1:M) {
    beta_language_std[m] ~ normal(mu_beta_language, sigma_beta_language);
    beta_task_std[m] ~ normal(mu_beta_task, sigma_beta_task);
  }

  // Likelihood
  vector[N] a = mu .* phi;
  vector[N] b = (1 - mu) .* phi;
  y ~ beta(a, b);
}

generated quantities {
  vector[N] y_pred;

  for (i in 1:N) {
    real a = mu[i] * phi[i];
    real b = (1 - mu[i]) * phi[i];
    y_pred[i] = beta_rng(a, b);
  }

  // Test data
  vector[N_test] y_pred_test;
  for (i in 1:N_test) {
    real mu_test = inv_logit(alpha_std[group_test[i]]
                + beta_language_std[group_test[i], language_test[i]]
                + beta_task_std[group_test[i], task_test[i]]);
    real phi_test = exp(phi_alpha + beta_task_phi[task_test[i]]);
    real a_test = mu_test * phi_test;
    real b_test = (1 - mu_test) * phi_test;
    y_pred_test[i] = beta_rng(a_test, b_test);
  }
}