data {
  int<lower = 1> n; //number of sample obs
  real lwt[n]; //log of weights for sample obs
  int<lower=1> m; //number of areas
  int<lower=1, upper=m> grp[n]; //area number of each sample obs
  int<lower=1> p; //number of covariates
  matrix[n, p] X; //Fixed effect design matrix for sample
  int<lower=0, upper=1> y[n]; //Binary response for sample
  int<lower=1> n_pred;  //number of cells in each county
  int<lower=1, upper=m> grp_pred[n_pred]; //area number for each prediction
  matrix[n_pred, p] X_pred; //Fixed effects design matrix for predictions
  matrix[m, n_pred] N_MAT; //Population size matrix for each cell
}
parameters {
  vector[p] beta;
  vector[m] phi;
  vector[p] alpha;
  real a;
  real<lower=0> tau;
  real<lower=0> tau_w;
}
transformed parameters {
  vector[n] v;
  vector[n_pred] v_pred;
  v=phi[grp];
  v_pred=phi[grp_pred];
}
model {
  y ~ bernoulli_logit(X*beta + v);
  for(i in 1:n){
    lwt[i] ~ normal(X[i, ]*alpha + y[i]*a, tau_w);
  }
  phi ~ normal(0, tau);
  beta ~ normal(0, sqrt(10));
  alpha ~ normal(0, sqrt(10));
  a ~ normal(0, sqrt(10));
  tau ~ cauchy(0, 5);
  tau_w ~ cauchy(0, 5);
}
generated quantities {
  vector[n_pred] p_pred;
  vector[n_pred] y_pred;
  vector[n_pred] num1;
  vector[n_pred] den1;
  vector[n_pred] den2;
  vector[m] y_pred_cty;
  p_pred=inv_logit(X_pred*beta + v_pred);
  num1=exp(X_pred*alpha + a + tau_w/2) - 1;
  den1=exp(X_pred*alpha + a + tau_w/2) .* p_pred;
  den2=exp(X_pred*alpha + tau_w/2) .* (1 - p_pred) -1;
  y_pred=(num1).*(p_pred)./(den1 + den2);
  y_pred_cty=N_MAT*y_pred;
}
