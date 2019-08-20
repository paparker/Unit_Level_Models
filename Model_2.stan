data {
  int<lower=0> popSize;
  int<lower=0> ss;
  int<lower=0> r; //number of counties
  int<lower=0> J; //number of cells
  int s[r]; // number of cells in each county
  int<lower=0> nk[r]; //sample count by county
  int<lower=0> Nk[r]; //population count by county
  int county[J]; //county index for each obs
  int<lower=0> y[J]; //response
  int<lower=0> vec_n[J];
  real w[J]; //weights
  int<lower=0> n_Uni;
  real uni_W[n_Uni]; //unique weights that define the cells (sorted)
  int<lower=1, upper=n_Uni> uni_W_Ind[J]; //unique weight indicator (for sorted weights)
  matrix<lower=0, upper=1>[r, r] W; //county adjacency matrix
}
transformed data {
  real delta = 1e-9;
  vector[r] zeros;
  vector[J] vec_w;
  //vector[N] lW;
  //real lW_uni[J];
  matrix<lower=0>[r, r] D;
  {
    vector[r] W_rowsums;
    for (i in 1:r) {
      W_rowsums[i] = sum(W[i, ]);
    }
    D = diag_matrix(W_rowsums);
  }
  zeros = rep_vector(0, r);
  vec_w = to_vector(w);
  //lW = log(vec_w);
  //lW_uni = log(uni_W);
}
parameters {
  vector[r] u; //spatial random effects
  vector[r] v; // iid random effects (area level)
  real<lower=0> precU; //ICAR precision parameter
  real<lower=0> sigV; //iid RE standard deviation
  real<lower=-1, upper=1> alpha; //CAR correlation
  vector<lower=0, upper=1>[J] cellPop;

  real<lower=0> alphaGP;
  real<lower=0> rho;
  vector[n_Uni] eta;
}
transformed parameters {
  vector[n_Uni] f;
  matrix[n_Uni, n_Uni] L_K;
  matrix[n_Uni, n_Uni] K=cov_exp_quad(uni_W, alphaGP, rho);
  for(n in 1:n_Uni){
    K[n, n] = K[n, n] + delta;
  }
  L_K = cholesky_decompose(K);
  f = L_K * eta;
}
model {
  vector[J] cellExp;
  int pos;
  pos=1;
  for(c in 1:r){
    vector[s[c]] t1 = cellPop[pos:(pos+s[c]-1)];
    vector[s[c]] t2 = vec_w[pos:(pos+s[c]-1)];
    cellExp[pos:(pos+s[c]-1)] = nk[c] * (t1 ./ t2) / sum(t1 ./ t2);
    pos = pos + s[c];
  }
  vec_n ~ poisson(cellExp);

  y ~ binomial_logit(vec_n, f[uni_W_Ind] + v[county] + u[county]);
  u ~ multi_normal_prec(zeros, precU*(D - alpha*W)); //CAR prior on u
  v ~ normal(0, sigV); // iid prior on v
  precU ~ normal(0, 100); //prior on precU
  sigV ~ normal(0, 100); //prior on sigV
  eta ~ normal(0,1);
  alpha ~ normal(0,1);
  rho ~ inv_gamma(5,5);
}
generated quantities {
  vector[J] cellPopNorm;
  vector[J] cellMean;
  int pos2;
  pos2=1;
  for(c in 1:r){
    cellPopNorm[pos2:(pos2+s[c]-1)] = Nk[c]*(cellPop[pos2:(pos2+s[c]-1)])/sum(cellPop[pos2:(pos2+s[c]-1)]);
    pos2 = pos2 + s[c];
  }
  cellMean=rep_vector(1,J) ./ (1.0 + exp(-1*(  f[uni_W_Ind] + v[county] + u[county])));
}
