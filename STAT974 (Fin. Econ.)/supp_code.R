packages <- c("midasr","quantmod","xts","rugarch","stochvol","rstan")
to_install <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(to_install)>0) install.packages(to_install)

library(midasr); library(quantmod); library(xts)
library(rugarch); library(stochvol); library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = 10)
set.seed(1)

# ---------- PF helper functions 
logsumexp <- function(a){
  m <- max(a)
  m + log(sum(exp(a - m)))
}

resample_systematic <- function(w){
  N <- length(w)
  u0 <- runif(1) / N
  u <- u0 + (0:(N-1)) / N
  cdf <- cumsum(w)
  idx <- integer(N)
  j <- 1
  for(i in 1:N){
    while(u[i] > cdf[j]) j <- j + 1
    idx[i] <- j
  }
  idx
}

# ---------- Data
data("rvsp500", package="midasr")
rv_dates <- as.Date(as.character(rvsp500$DateID), format="%Y%m%d")
rv_xts <- xts(as.numeric(rvsp500$SPX2.rv), order.by=rv_dates)
colnames(rv_xts) <- "RV"

getSymbols("^GSPC", src="yahoo", from="2000-01-01", to="2013-12-31", auto.assign=TRUE)
px <- Ad(GSPC)
r_xts <- diff(log(px))
colnames(r_xts) <- "r"
index(r_xts) <- as.Date(index(r_xts))

dat <- merge(r_xts, rv_xts, join="inner")
dat <- dat[is.finite(dat$r) & is.finite(dat$RV) & !is.na(dat$r) & !is.na(dat$RV) & dat$RV > 0, ]
dat$logRV <- log(dat$RV)

stopifnot(
  NROW(dat) > 0,
  all(is.finite(coredata(dat$r))),
  all(is.finite(coredata(dat$RV))),
  all(coredata(dat$RV) > 0)
)

dates <- index(dat)
n <- NROW(dat)
test_n <- max(1, floor(0.2*n))
train_n <- n - test_n

train <- dat[1:train_n]
test  <- dat[(train_n+1):n]
split_date <- dates[train_n]

r_all  <- as.numeric(dat$r)
x_all  <- as.numeric(dat$RV)

r_train  <- as.numeric(train$r)
r_test   <- as.numeric(test$r)
x_train  <- as.numeric(train$RV)
x_test   <- as.numeric(test$RV)
lx_train <- as.numeric(train$logRV)
lx_test  <- as.numeric(test$logRV)

H <- length(r_test)

# ---------- Metrics
qlike <- function(y, hhat) mean(log(hhat) + y/hhat, na.rm=TRUE)
mse   <- function(y, yhat) mean((y - yhat)^2, na.rm=TRUE)
mae   <- function(y, yhat) mean(abs(y - yhat), na.rm=TRUE)

gauss_logscore <- function(r, mu, h){
  mean(-0.5*(log(2*pi) + log(h) + (r - mu)^2/h), na.rm=TRUE)
}

# ---------- GARCH(1,1)
garch_spec <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model = list(armaOrder=c(0,0), include.mean=TRUE),
  distribution.model = "norm"
)
garch_fit <- ugarchfit(spec=garch_spec, data=r_train, solver="hybrid")

cf_g <- coef(garch_fit)
omega_g <- unname(cf_g["omega"])
alpha_g <- unname(cf_g["alpha1"])
beta_g  <- unname(cf_g["beta1"])
mu_g    <- if ("mu" %in% names(cf_g)) unname(cf_g["mu"]) else mean(r_train)

h_g_train <- as.numeric(sigma(garch_fit))^2
h_last_g  <- tail(h_g_train, 1)

h_g_test <- rep(NA_real_, H)
h_prev <- as.numeric(h_last_g)
r_prev <- tail(r_train, 1)
for(i in 1:H){
  eps_prev <- (if(i==1) r_prev else r_test[i-1]) - mu_g
  h_prev <- omega_g + alpha_g*(eps_prev^2) + beta_g*h_prev
  h_g_test[i] <- h_prev
}

h_g_all <- c(h_g_train, h_g_test)
mu_g_all <- rep(mu_g, n)

# ---------- SV fit (stochvol) + in-sample variance
r_train_mu <- mean(r_train)
r_train_dm <- r_train - r_train_mu

sv_fit <- svsample(r_train_dm, draws=4000, burnin=1000, quiet=TRUE)

pars_sv <- as.matrix(sv_fit$para)
mu_sv_med  <- median(pars_sv[,"mu"])
phi_sv_med <- median(pars_sv[,"phi"])
sig_sv_med <- median(pars_sv[,"sigma"])

latent_sv <- as.matrix(sv_fit$latent)  # draws x T
h_sv_train <- apply(exp(latent_sv), 2, median)

# ---------- SV particle filter on test (fixed params; no refit)
mu_h_pf    <- mu_sv_med
phi_pf     <- phi_sv_med
sig_eta_pf <- sig_sv_med

Np_sv <- 5000

hT_sv <- latent_sv[, ncol(latent_sv)]
h_particles <- sample(hT_sv, size=Np_sv, replace=TRUE)

h_filt_test_sv <- numeric(H)

for(t in 1:H){
  h_particles <- mu_h_pf + phi_pf*(h_particles - mu_h_pf) + rnorm(Np_sv, 0, sig_eta_pf)

  rt_dm <- r_test[t] - r_train_mu
  loglik_r <- dnorm(rt_dm, mean=0, sd=exp(h_particles/2), log=TRUE)

  logw <- loglik_r
  logw <- logw - logsumexp(logw)
  w <- exp(logw)

  h_filt_test_sv[t] <- sum(w * h_particles)

  idx <- resample_systematic(w)
  h_particles <- h_particles[idx]
}

h_sv_test <- exp(h_filt_test_sv)
h_sv_all <- c(h_sv_train, h_sv_test)
mu_sv_all <- rep(r_train_mu, n)

# ---------- Realized GARCH (custom likelihood)
rgarch_nll <- function(par, r, x){
  omega <- exp(par[1])
  beta  <- 1/(1+exp(-par[2]))
  gamma <- exp(par[3])
  xi    <- par[4]
  delta <- par[5]
  su    <- exp(par[6])
  mu    <- par[7]

  n <- length(r)
  h <- rep(var(r), n)
  ll <- 0

  for(t in 2:n){
    h[t] <- omega + beta*h[t-1] + gamma*x[t-1]
    if(h[t] <= 0 || !is.finite(h[t])) return(1e12)

    eps <- r[t] - mu
    ll_r <- -0.5*(log(2*pi) + log(h[t]) + (eps^2)/h[t])

    m <- xi + delta*log(h[t])
    lx <- log(x[t])
    if(!is.finite(m) || !is.finite(lx)) return(1e12)

    ll_x <- -0.5*(log(2*pi) + 2*log(su) + ((lx - m)^2)/(su^2))
    if(!is.finite(ll_x)) return(1e12)

    ll <- ll + ll_r + ll_x
    if(!is.finite(ll)) return(1e12)
  }
  -ll
}

mu_hat <- mean(r_train)
rg_init <- c(log(0.01), qlogis(0.9), log(0.1), 0, 1, log(0.2), mu_hat)
rg_fit <- optim(rg_init, rgarch_nll, r=r_train, x=x_train, method="BFGS",
                control=list(maxit=800))
rg_par <- rg_fit$par

omega_rg <- exp(rg_par[1])
beta_rg  <- 1/(1+exp(-rg_par[2]))
gamma_rg <- exp(rg_par[3])
xi_rg    <- rg_par[4]
delta_rg <- rg_par[5]
su_rg    <- exp(rg_par[6])
mu_rg    <- rg_par[7]

h_rg_train <- rep(var(r_train), length(r_train))
for(t in 2:length(r_train)){
  h_rg_train[t] <- omega_rg + beta_rg*h_rg_train[t-1] + gamma_rg*x_train[t-1]
}

# one-step-ahead: lagged RV only
h_rg_test <- rep(NA_real_, H)
h_prev <- tail(h_rg_train, 1)
x_prev <- tail(x_train, 1)
for(i in 1:H){
  x_lag <- if(i==1) x_prev else x_test[i-1]
  h_prev <- omega_rg + beta_rg*h_prev + gamma_rg*x_lag
  h_rg_test[i] <- h_prev
}

h_rg_all <- c(h_rg_train, h_rg_test)
mu_rg_all <- rep(mu_rg, n)

# ---------- Realized SV in Stan (fit on train)
stan_code <- "
data {
  int<lower=1> T;
  vector[T] r;
  vector[T] logx;
}
parameters {
  real mu;
  real mu_h;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_eta;
  real alpha;
  real<lower=0> sigma_u;
  vector[T] h;
}
model {
  mu ~ normal(0, 1);
  mu_h ~ normal(0, 1);
  phi ~ normal(0, 0.5);
  sigma_eta ~ cauchy(0, 0.5);
  alpha ~ normal(0, 1);
  sigma_u ~ cauchy(0, 0.5);

  h[1] ~ normal(mu_h, sigma_eta / sqrt(1 - phi*phi));
  for(t in 2:T)
    h[t] ~ normal(mu_h + phi*(h[t-1]-mu_h), sigma_eta);

  r ~ normal(mu, exp(h/2));
  logx ~ normal(alpha + h, sigma_u);
}
"

stan_dat <- list(T=length(r_train), r=r_train, logx=lx_train)
sm <- stan_model(model_code=stan_code)

rsv_fit <- sampling(
  object=sm,
  data=stan_dat,
  chains=4,
  iter=2500,
  warmup=1500,
  seed=1,
  refresh=100,
  control=list(adapt_delta=0.99, max_treedepth=15)
)

post <- rstan::extract(rsv_fit, pars=c("mu","mu_h","phi","sigma_eta","alpha","sigma_u","h"))

mu_rsv_med    <- median(post$mu)
mu_h_med      <- median(post$mu_h)
phi_med       <- median(post$phi)
sig_eta_med   <- median(post$sigma_eta)
alpha_med     <- median(post$alpha)
sig_u_med     <- median(post$sigma_u)

h_draws <- post$h
h_rsv_train <- apply(exp(h_draws), 2, median)
hT_draw2 <- h_draws[, ncol(h_draws)]

# ---------- Realized SV particle filter on test
Np_rsv <- 5000

mu_pf      <- mu_rsv_med
mu_h_pf2   <- mu_h_med
phi_pf2    <- phi_med
sig_eta_pf2<- sig_eta_med
alpha_pf   <- alpha_med
sig_u_pf2  <- sig_u_med

h_particles <- sample(hT_draw2, size=Np_rsv, replace=TRUE)
h_filt_test_rsv <- numeric(H)

for(t in 1:H){
  h_particles <- mu_h_pf2 + phi_pf2*(h_particles - mu_h_pf2) + rnorm(Np_rsv, 0, sig_eta_pf2)

  rt  <- r_test[t]
  lxt <- lx_test[t]

  loglik_r <- dnorm(rt, mean=mu_pf, sd=sqrt(exp(h_particles)), log=TRUE)
  loglik_x <- dnorm(lxt, mean=alpha_pf + h_particles, sd=sig_u_pf2, log=TRUE)

  logw <- loglik_r + loglik_x
  logw <- logw - logsumexp(logw)
  w <- exp(logw)

  h_filt_test_rsv[t] <- sum(w * h_particles)

  idx <- resample_systematic(w)
  h_particles <- h_particles[idx]
}

h_rsv_test <- exp(h_filt_test_rsv)
h_rsv_all  <- c(h_rsv_train, h_rsv_test)
mu_rsv_all <- rep(mu_rsv_med, n)

# ---------- Metric table (test)
rv_test <- x_test
models <- c("GARCH(1,1)", "SV", "RealizedGARCH", "RealizedSV")

rv_tbl <- data.frame(
  model = models,
  QLIKE_RV = c(
    qlike(rv_test, h_g_test),
    qlike(rv_test, h_sv_test),
    qlike(rv_test, h_rg_test),
    qlike(rv_test, h_rsv_test)
  ),
  MSE_RV = c(
    mse(rv_test, h_g_test),
    mse(rv_test, h_sv_test),
    mse(rv_test, h_rg_test),
    mse(rv_test, h_rsv_test)
  ),
  MAE_RV = c(
    mae(rv_test, h_g_test),
    mae(rv_test, h_sv_test),
    mae(rv_test, h_rg_test),
    mae(rv_test, h_rsv_test)
  )
)

ret_tbl <- data.frame(
  model = models,
  LogScore_ret = c(
    gauss_logscore(r_test, mu_g, h_g_test),
    gauss_logscore(r_test, r_train_mu, h_sv_test),
    gauss_logscore(r_test, mu_rg, h_rg_test),
    gauss_logscore(r_test, mu_rsv_med, h_rsv_test)
  ),
  MSE_mean = c(
    mse(r_test, rep(mu_g, length(r_test))),
    mse(r_test, rep(r_train_mu, length(r_test))),
    mse(r_test, rep(mu_rg, length(r_test))),
    mse(r_test, rep(mu_rsv_med, length(r_test)))
  ),
  MAE_mean = c(
    mae(r_test, rep(mu_g, length(r_test))),
    mae(r_test, rep(r_train_mu, length(r_test))),
    mae(r_test, rep(mu_rg, length(r_test))),
    mae(r_test, rep(mu_rsv_med, length(r_test)))
  )
)

metric_tab <- merge(rv_tbl, ret_tbl, by="model", sort=FALSE)
print(metric_tab, row.names=FALSE, digits=6)

# ---------- Save per-model plots (Volatility and Returns)

save_vol_plot <- function(fname, title, vol_model, vol_real, dates, split_date){
  png(filename=fname, width=1200, height=700, res=150)
  plot(dates, vol_real, type="l", col="black", lwd=2,
       xlab="Date", ylab="Volatility",
       main=title)
  lines(dates, vol_model, col="blue", lwd=1.8)
  abline(v=split_date, lty=2)
  legend("topleft",
         legend=c("sqrt(RV)","Model-implied volatility","Train/Test split"),
         col=c("black","blue","black"),
         lty=c(1,1,2),
         lwd=c(2,1.8,1),
         bty="n",
         cex=0.9)
  dev.off()
}

save_ret_plot <- function(fname, title, r_all, mu_all, h_all, dates, split_date){
  png(filename=fname, width=1200, height=700, res=150)
  plot(dates, r_all, type="l", col="black",
       xlab="Date", ylab="Return",
       main=title)
  lines(dates, mu_all + 2*sqrt(h_all), col="blue", lty=3, lwd=1.4)
  lines(dates, mu_all - 2*sqrt(h_all), col="blue", lty=3, lwd=1.4)
  abline(v=split_date, lty=2)
  legend("topright",
         legend=c("Returns","Model pm 2*sigma","Train/Test split"),
         col=c("black","blue","black"),
         lty=c(1,3,2),
         lwd=c(1,1.4,1),
         bty="n",
         cex=0.9)
  dev.off()
}

# ---------- Volatility series
vol_real <- sqrt(x_all)

vol_g   <- sqrt(h_g_all)
vol_sv  <- sqrt(h_sv_all)
vol_rg  <- sqrt(h_rg_all)
vol_rsv <- sqrt(h_rsv_all)

# ---------- Volatility plots
save_vol_plot("vol_garch.png",
              "Volatility: sqrt(RV) vs GARCH(1,1)",
              vol_g, vol_real, dates, split_date)

save_vol_plot("vol_sv.png",
              "Volatility: sqrt(RV) vs SV (filtered)",
              vol_sv, vol_real, dates, split_date)

save_vol_plot("vol_rgarch.png",
              "Volatility: sqrt(RV) vs Realized GARCH",
              vol_rg, vol_real, dates, split_date)

save_vol_plot("vol_rsv.png",
              "Volatility: sqrt(RV) vs Realized SV (filtered)",
              vol_rsv, vol_real, dates, split_date)

# ---------- Return plots
save_ret_plot("ret_garch.png",
              "Returns with 2*sigma band: GARCH(1,1)",
              r_all, mu_g_all, h_g_all, dates, split_date)

save_ret_plot("ret_sv.png",
              "Returns with 2*sigma band: SV (filtered)",
              r_all, mu_sv_all, h_sv_all, dates, split_date)

save_ret_plot("ret_rgarch.png",
              "Returns with 2*sigma band: Realized GARCH",
              r_all, mu_rg_all, h_rg_all, dates, split_date)

save_ret_plot("ret_rsv.png",
              "Returns with 2*sigma band: Realized SV (filtered)",
              r_all, mu_rsv_all, h_rsv_all, dates, split_date)

cat("Saved plots:\n",
    "vol_garch.png, vol_sv.png, vol_rgarch.png, vol_rsv.png\n",
    "ret_garch.png, ret_sv.png, ret_rgarch.png, ret_rsv.png\n")
    
# ---------- Standardized residuals
z_garch <- (r_test - mu_g) / sqrt(h_g_test)
z_sv    <- (r_test - r_train_mu) / sqrt(h_sv_test)
z_rg    <- (r_test - mu_rg) / sqrt(h_rg_test)
z_rsv   <- (r_test - mu_rsv_med) / sqrt(h_rsv_test)

# ---------- Residual plots
save_resid_plot <- function(fname, title, z, dates){
  png(fname, width=1200, height=700, res=150)
  plot(dates, z, type="l", col="black",
       main=title, xlab="Date", ylab="Standardized residual")
  abline(h=c(-2,0,2), lty=c(2,1,2), col=c("gray","black","gray"))
  dev.off()
}

save_resid_plot("resid_garch.png", "Standardized residuals: GARCH(1,1)", z_garch, index(test))
save_resid_plot("resid_sv.png",    "Standardized residuals: SV", z_sv, index(test))
save_resid_plot("resid_rgarch.png","Standardized residuals: Realized GARCH", z_rg, index(test))
save_resid_plot("resid_rsv.png",   "Standardized residuals: Realized SV", z_rsv, index(test))

# Residual ACF plot
png("resid_acf_sq.png", width=1200, height=700, res=150)
par(mfrow=c(2,2))
acf(z_garch^2, main="ACF(z^2): GARCH")
acf(z_sv^2,    main="ACF(z^2): SV")
acf(z_rg^2,    main="ACF(z^2): Realized GARCH")
acf(z_rsv^2,   main="ACF(z^2): Realized SV")
dev.off()
