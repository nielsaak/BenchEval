data {
  int<lower=1> N;                       // Number of observations
  int<lower=1> n_language;              // Number of language levels
  int<lower=1> n_metric;                // Number of metric levels
  int<lower=1> n_task;                  // Number of task levels

  // Categorical predictors
  array[N] int<lower=1,upper=n_language> language;
  array[N] int<lower=1,upper=n_metric> metric;
  array[N] int<lower=1,upper=n_task> task;

  // Outcome variable (continuous, between 0 and 1)
  vector<lower=0,upper=1>[N] y;
}

parameters {
  real alpha;                           // Overall intercept

  // Coefficients for each categorical predictor
  vector[n_language] beta_language;
  vector[n_metric] beta_metric;
  vector[n_task] beta_task;

  // Precision parameter for the beta distribution
  // Controls the variance; higher phi means lower variance around the mean.
  real<lower=0> phi;
}

transformed parameters {
  // The linear predictor is transformed via the inverse logit to lie in (0,1)
  vector<lower=0,upper=1>[N] mu;
  for (i in 1:N)
    mu[i] = inv_logit(alpha + beta_language[language[i]] +
                      beta_metric[metric[i]] +
                      beta_task[task[i]]);
}

model {
  // Priors
  alpha ~ normal(0, 5);
  beta_language ~ normal(0, 2);
  beta_metric ~ normal(0, 2);
  beta_task ~ normal(0, 2);
  phi ~ gamma(2, 0.1);

  // Likelihood: beta regression with mean mu and precision phi.
  // The beta distribution is parameterized here by (mu*phi, (1-mu)*phi).
  y ~ beta(mu * phi, (1 - mu) * phi);
}
