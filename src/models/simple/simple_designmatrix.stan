data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix

  // Outcome variable (continuous, between 0 and 1)
  vector<lower=0,upper=1>[N] y;
}

parameters {
  vector[K] beta;
  
  // Precision parameter for the beta distribution
  // Controls the variance; higher phi means lower variance around the mean.
  real<lower=0> phi;
}

transformed parameters {
  
  // The linear predictor is transformed via the inverse logit to lie in (0,1)
  vector<lower=0,upper=1>[N] mu;
  
  mu = inv_logit(x*beta);
}

model {
  // Priors
  beta ~ normal(0, 5);
  phi ~ gamma(2, 0.1);

  // Likelihood: beta regression with mean mu and precision phi.
  // The beta distribution is parameterized here by (mu*phi, (1-mu)*phi).
  y ~ beta(mu * phi, (1 - mu) * phi);
}