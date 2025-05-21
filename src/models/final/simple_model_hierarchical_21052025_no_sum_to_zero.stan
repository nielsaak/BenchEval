data {
  int<lower=0> N;                       // Number of observations
  int<lower=0> M;                       // Number of groups/models
  int<lower=0> n_language;              // Number of languages
  int<lower=0> n_task;                  // Number of tasks

  array[N] int<lower=1,upper=M> group;  // Group indicator for each observation
  array[N] int<lower=1,upper=n_language> language; // Language indicator for each observation
  array[N] int<lower=1,upper=n_task> task; // Task indicator for each observation

  vector<lower=0,upper=1>[N] y; // Output value for each observation
}

parameters {
  // Hyperpriors
  real mu_alpha;
  real<lower=0> sigma_alpha;

  vector[n_language-1] mu_beta_language_raw;
  vector<lower=0>[n_language-1] sigma_beta_language;

  vector[n_task-1] mu_beta_task;
  vector<lower=0>[n_task-1] sigma_beta_task;

  // Group-level parameters
  vector[M] alpha_std;
  array[M] vector[n_language-1] beta_language_std;
  array[M] vector[n_task-1] beta_task_std;

  // Phi parameters
  real phi_alpha;
  vector[n_task-1] beta_task_phi;
}

transformed parameters {
  vector[n_language] mu_beta_language;
  mu_beta_language[1] = 0;
  for (k in 2:n_language) {
    mu_beta_language[k] = mu_beta_language_raw[k-1];
  }


  vector[M] alpha = mu_alpha + sigma_alpha * alpha_std;
  array[M] row_vector[n_language-1] beta_language;
  array[M] row_vector[n_task-1] beta_task;

  for (m in 1:M) {
    for (k in 1:n_language-1) {
      beta_language[m,k] = mu_beta_language[k] + sigma_beta_language[k] * beta_language_std[m,k];
    }
    for (k in 1:n_task-1) {
      beta_task[m,k] = mu_beta_task[k] + sigma_beta_task[k] * beta_task_std[m,k];
    }
  }

  vector[N] eta_mu;
  for (i in 1:N) {
    eta_mu[i] = alpha[group[i]]
                + beta_language[group[i], language[i]]
                + beta_task[group[i], task[i]];
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
  alpha_std ~ std_normal();

  mu_beta_language ~ normal(0, 2);
  sigma_beta_language ~ exponential(1);

  mu_beta_task ~ normal(0, 2);
  sigma_beta_task ~ exponential(1);


  for (m in 1:M) {
    beta_language_std[m] ~ std_normal();
    beta_task_std[m] ~ std_normal();
  }

  // Likelihood
  vector[N] a = mu .* phi + 1e-9;
  vector[N] b = (1 - mu) .* phi + 1e-9;
  y ~ beta(a, b);
}

generated quantities {
  vector[N] y_pred;

  for (i in 1:N) {
    real a = mu[i] * phi[i] + 1e-9;
    real b = (1 - mu[i]) * phi[i] + 1e-9;
    y_pred[i] = beta_rng(a, b);
  }
}