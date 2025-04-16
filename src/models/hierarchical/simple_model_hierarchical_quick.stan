data {
  int<lower=1> N;                       // Number of observations
  int<lower=1> M;                       // Number of groups/models
  int<lower=1> n_language;
  int<lower=1> n_metric;
  int<lower=1> n_task;

  array[N] int<lower=1,upper=M> group;  // Group indicator
  array[N] int<lower=1,upper=n_language> language;
  array[N] int<lower=1,upper=n_metric> metric;
  array[N] int<lower=1,upper=n_task> task;

  vector<lower=0,upper=1>[N] y;
}

parameters {
  // Hyperpriors
  real mu_alpha;
  real<lower=0> sigma_alpha;

  vector[n_language] mu_beta_language;
  real<lower=0> sigma_beta_language;

  vector[n_metric] mu_beta_metric;
  real<lower=0> sigma_beta_metric;

  vector[n_task] mu_beta_task;
  real<lower=0> sigma_beta_task;

  real<lower=0> mu_phi;
  real<lower=0> sigma_phi;

  // Group-level parameters
  vector[M] alpha;
  matrix[M, n_language] beta_language;
  matrix[M, n_metric] beta_metric;
  matrix[M, n_task] beta_task;
  vector<lower=0>[M] phi;
}

transformed parameters {
  vector<lower=0, upper=1>[N] mu;
  for (i in 1:N) {
    real eta = alpha[group[i]]
                + beta_language[group[i], language[i]]
                + beta_metric[group[i], metric[i]]
                + beta_task[group[i], task[i]];
    mu[i] = inv_logit(eta);
  }
}

model {
  // Hyperpriors
  mu_alpha ~ normal(0, 2);
  sigma_alpha ~ exponential(1);

  mu_beta_language ~ normal(0, 2);
  sigma_beta_language ~ exponential(1);

  mu_beta_metric ~ normal(0, 2);
  sigma_beta_metric ~ exponential(1);

  mu_beta_task ~ normal(0, 2);
  sigma_beta_task ~ exponential(1);

  mu_phi ~ normal(0, 2);
  sigma_phi ~ exponential(1);

  // Group-level priors
  alpha ~ normal(mu_alpha, sigma_alpha);

  for (m in 1:M) {
    beta_language[m] ~ normal(mu_beta_language, sigma_beta_language);
    beta_metric[m] ~ normal(mu_beta_metric, sigma_beta_metric);
    beta_task[m] ~ normal(mu_beta_task, sigma_beta_task);
    phi[m] ~ normal(mu_phi, sigma_phi);
  }

  // Likelihood
  for (i in 1:N) {
    y[i] ~ beta_proportion(mu[i], phi[group[i]]);
  }
}
