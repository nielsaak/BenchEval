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

generated quantities {
   // Prior predictive samples
   real alpha_prior;
   real beta_language_prior;
   real beta_metric_prior;
   real beta_task_prior;
   real phi_prior;
   
   // Posterior and prior predictions
   // array[trials] int prior_preds;
   // array[trials] int posterior_preds;

   // Generate prior samples
   alpha_prior = normal_rng(0, 5);
   beta_language_prior = normal_rng(0, 2);
   beta_metric_prior = normal_rng(0, 2);
   beta_task_prior = normal_rng(0, 2);
   phi_prior = gamma_rng(2, 0.1);

   //array[N] real<lower=0, upper=1> mu_prior;
   //for (i in 1:N){
   // mu_prior[i] = inv_logit(alpha_prior + beta_language_prior + beta_metric_prior + beta_task_prior);
   //}
   real<lower=0, upper=1> mu_prior;
   mu_prior = inv_logit(alpha_prior + beta_language_prior + beta_metric_prior + beta_task_prior);
   
   array[N] real<lower=0, upper=1> mu_posterior;
   for (i in 1:N){
    mu_posterior[i] = inv_logit(alpha + beta_language[language[i]] + beta_metric[metric[i]] + beta_task[task[i]]);
   }
   
}
