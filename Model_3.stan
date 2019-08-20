data {
  int<lower = 1> n; //number of obs
  real lwt[n]; //log of weights
  int<lower=1> m;
  int<lower=1, upper=m> grp[n];
  int<lower=1> p;
  matrix[n, p] X;
  int<lower=0, upper=1> y[n];
  int<lower=1> n_pred;
  int<lower=1, upper=m> grp_pred[n_pred];
  matrix[n_pred, p] X_pred;
  matrix[m, n_pred] N_MAT;
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
  beta ~ normal(0, 10);
  alpha ~ normal(0, 10);
  a ~ normal(0, 10);
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
