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
  vector[M] alpha = mu_alpha + sigma_alpha * alpha_std;
  array[M] row_vector[n_language] beta_language;
  array[M] row_vector[n_task] beta_task;

  for (m in 1:M) {
    for (k in 1:n_language) {
      beta_language[m,k] = mu_beta_language[k] + sigma_beta_language[k] * beta_language_std[m,k];
    }
    for (k in 1:n_task) {
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
  phi_alpha ~ normal(log(10), 0.5);
  beta_task_phi ~ normal(0, 0.5);


  // Hyperpriors
  // mu_alpha ~ std_normal();
  // sigma_alpha ~ exponential(1);
  // alpha_std ~ std_normal();

  // mu_beta_language ~ std_normal();
  // sigma_beta_language ~ exponential(1);

  // mu_beta_task ~ std_normal();
  // sigma_beta_task ~ exponential(1);

  // 4.1 Coefficients: Student-t(3,0,0.5)
  mu_alpha        ~ student_t(3, 0, 0.5);
  sigma_alpha     ~ exponential(2);
  alpha_std       ~ normal(0, 1);

  mu_beta_language~ student_t(3, 0, 0.5);
  sigma_beta_language~ exponential(2);
  mu_beta_task    ~ student_t(3, 0, 0.5);
  sigma_beta_task ~ exponential(2);

  alpha_std       ~ normal(0, 1);
  for (m in 1:M) {
    beta_language_std[m] ~ normal(0, 1);
    beta_task_std[m]     ~ normal(0, 1);
  }

  // 4.2 φ ~ Gamma(5,1) via transformed parameters
  phi_alpha       ~ normal(log(5), 0.3);   // centers φ≈5
  beta_task_phi   ~ normal(0, 0.5);

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
    // real y_pred_temp = beta_proportion_rng(mu[i], phi[i]);
    // if below 0.01, set to 0.01
    // if (y_pred_temp < 0.01) {
    //   y_pred_temp = 0.01;
    // }
    // // if above 0.99, set to 0.99
    // if (y_pred_temp > 0.99) {
    //   y_pred_temp = 0.99;
    // }
    // y_pred[i] = y_pred_temp;
  }
}