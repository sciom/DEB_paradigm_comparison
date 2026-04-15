## ============================================================
## Reproducible benchmark: Classical DEB vs Bayesian DEB vs ML
## for Daphnia magna body length growth
## ============================================================
##
## Supplementary code for:
## "Dynamic Energy Budget Theory, Bayesian Inference, and
##  Machine Learning: A Comparative Review with a Decision
##  Framework for Ecological Modelling"
##
## Authors: B.K. Hackenberger & T. Djerdj
## Journal: Ecological Modelling
##
## Requirements:
##   install.packages(c("deSolve", "BayesianTools", "randomForest",
##                      "xgboost", "ggplot2", "patchwork", "dplyr", "tidyr"))
##
## Usage:
##   source("benchmark.R")
##   # Results saved to benchmark_results.csv and benchmark_figure.pdf
##
## ============================================================

library(deSolve)
library(BayesianTools)
library(randomForest)
library(xgboost)
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)

set.seed(42)

## ============================================================
## 1. SIMULATE "TRUE" DAPHNIA MAGNA GROWTH DATA
## ============================================================
## We use a simplified DEB model (DEBkiss-like von Bertalanffy
## growth) with published AmP parameters for D. magna as ground
## truth, then add realistic measurement noise.

# True DEB parameters (based on AmP entry for Daphnia magna)
true_params <- list(
  Linf  = 4.8,    # mm, maximum structural length
  rB    = 0.15,   # d^-1, von Bertalanffy growth rate
  L0    = 0.8,    # mm, length at birth (neonates)
  TA    = 8000,   # K, Arrhenius temperature
  Tref  = 293.15, # K, reference temperature (20 C)
  sigma = 0.12    # mm, observation noise (sd)
)

# DEB-derived von Bertalanffy growth function
deb_growth <- function(t, params, temp_C = 20) {
  TC <- exp(params$TA / params$Tref - params$TA / (temp_C + 273.15))
  rB_corr <- params$rB * TC
  L <- params$Linf - (params$Linf - params$L0) * exp(-rB_corr * t)
  return(L)
}

# Generate training data: 20 C, ad libitum
times_train <- seq(0, 21, by = 1.5)  # 15 time points, 0-21 days
n_rep <- 5  # replicates per time point

train_data <- expand.grid(
  time = times_train,
  rep = 1:n_rep
) %>%
  mutate(
    temp = 20,
    L_true = deb_growth(time, true_params, temp_C = 20),
    L_obs = L_true + rnorm(n(), 0, true_params$sigma),
    L_obs = pmax(L_obs, 0.3)  # enforce positivity
  )

# Generate extrapolation data: 25 C (independent validation)
test_data <- expand.grid(
  time = times_train,
  rep = 1:n_rep
) %>%
  mutate(
    temp = 25,
    L_true = deb_growth(time, true_params, temp_C = 25),
    L_obs = L_true + rnorm(n(), 0, true_params$sigma),
    L_obs = pmax(L_obs, 0.3)
  )

cat("Training data: n =", nrow(train_data), "(20 C)\n")
cat("Test data:     n =", nrow(test_data), "(25 C)\n\n")

## ============================================================
## 2. APPROACH 1: CLASSICAL DEB (Least-squares optimisation)
## ============================================================

# Objective: minimise sum of squared residuals
classical_obj <- function(par, data) {
  Linf <- par[1]
  rB   <- par[2]
  L0   <- par[3]
  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params$TA, Tref = true_params$Tref)
  pred <- deb_growth(data$time, params_local, temp_C = data$temp[1])
  return(sum((data$L_obs - pred)^2))
}

cat("--- CLASSICAL DEB ---\n")
t_class_start <- Sys.time()

fit_classical <- optim(
  par = c(Linf = 5.0, rB = 0.12, L0 = 0.9),
  fn = classical_obj,
  data = train_data,
  method = "Nelder-Mead",
  control = list(maxit = 5000)
)

t_class_end <- Sys.time()
t_class <- as.numeric(difftime(t_class_end, t_class_start, units = "secs"))

params_classical <- list(
  Linf = fit_classical$par[1],
  rB   = fit_classical$par[2],
  L0   = fit_classical$par[3],
  TA   = true_params$TA,
  Tref = true_params$Tref
)

# Predictions
train_data$pred_classical <- deb_growth(train_data$time, params_classical, 20)
test_data$pred_classical  <- deb_growth(test_data$time, params_classical, 25)

cat("  Linf =", round(params_classical$Linf, 3), "mm\n")
cat("  rB   =", round(params_classical$rB, 4), "d^-1\n")
cat("  L0   =", round(params_classical$L0, 3), "mm\n")
cat("  Runtime:", round(t_class, 3), "s\n\n")

## ============================================================
## 3. APPROACH 2: BAYESIAN DEB (MCMC via BayesianTools)
## ============================================================

# Log-likelihood function
log_likelihood <- function(par) {
  Linf  <- par[1]
  rB    <- par[2]
  L0    <- par[3]
  sigma <- par[4]

  if (Linf <= 0 | rB <= 0 | L0 <= 0 | sigma <= 0) return(-Inf)
  if (L0 >= Linf) return(-Inf)

  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params$TA, Tref = true_params$Tref)
  pred <- deb_growth(train_data$time, params_local, temp_C = 20)
  ll <- sum(dnorm(train_data$L_obs, mean = pred, sd = sigma, log = TRUE))
  return(ll)
}

# Log-prior: informative priors from AmP database (Daphniidae)
log_prior <- function(par) {
  Linf  <- par[1]
  rB    <- par[2]
  L0    <- par[3]
  sigma <- par[4]

  lp <- dnorm(log(Linf), log(4.8), 0.2, log = TRUE) +   # AmP: Linf ~ 4-6 mm
        dnorm(log(rB), log(0.15), 0.3, log = TRUE) +     # AmP: rB ~ 0.08-0.25
        dnorm(log(L0), log(0.8), 0.2, log = TRUE) +      # neonate size ~ 0.6-1.0
        dunif(sigma, 0.01, 1.0, log = TRUE)               # weakly informative
  return(lp)
}

# BayesianTools setup
bayesian_setup <- createBayesianSetup(
  likelihood = log_likelihood,
  prior = createPrior(
    density = log_prior,
    sampler = function() c(
      exp(rnorm(1, log(4.8), 0.2)),
      exp(rnorm(1, log(0.15), 0.3)),
      exp(rnorm(1, log(0.8), 0.2)),
      runif(1, 0.05, 0.5)
    ),
    lower = c(1, 0.01, 0.1, 0.01),
    upper = c(10, 1.0, 3.0, 1.0)
  ),
  names = c("Linf", "rB", "L0", "sigma")
)

cat("--- BAYESIAN DEB ---\n")
cat("  Running MCMC (DEzs, 3 chains x 10000 iterations)...\n")
t_bayes_start <- Sys.time()

fit_bayesian <- runMCMC(
  bayesianSetup = bayesian_setup,
  sampler = "DEzs",
  settings = list(
    iterations = 10000,
    burnin = 3000,
    thin = 2,
    nrChains = 3,
    message = FALSE
  )
)

t_bayes_end <- Sys.time()
t_bayes <- as.numeric(difftime(t_bayes_end, t_bayes_start, units = "secs"))

# Extract posterior summary
posterior <- getSample(fit_bayesian, parametersOnly = TRUE)
post_summary <- apply(posterior, 2, function(x)
  c(mean = mean(x), sd = sd(x),
    q025 = quantile(x, 0.025), q975 = quantile(x, 0.975)))

cat("  Posterior summary:\n")
print(round(post_summary, 4))
cat("  Runtime:", round(t_bayes, 1), "s\n")

# Posterior predictive: generate predictions from posterior samples
n_post_samples <- min(500, nrow(posterior))
idx <- sample(nrow(posterior), n_post_samples)

# Interpolation (20 C) posterior predictive
pred_matrix_train <- matrix(NA, nrow = nrow(train_data), ncol = n_post_samples)
for (i in 1:n_post_samples) {
  p <- posterior[idx[i], ]
  params_i <- list(Linf = p[1], rB = p[2], L0 = p[3],
                   TA = true_params$TA, Tref = true_params$Tref)
  pred_matrix_train[, i] <- deb_growth(train_data$time, params_i, 20) +
    rnorm(nrow(train_data), 0, p[4])
}

# Extrapolation (25 C) posterior predictive
pred_matrix_test <- matrix(NA, nrow = nrow(test_data), ncol = n_post_samples)
for (i in 1:n_post_samples) {
  p <- posterior[idx[i], ]
  params_i <- list(Linf = p[1], rB = p[2], L0 = p[3],
                   TA = true_params$TA, Tref = true_params$Tref)
  pred_matrix_test[, i] <- deb_growth(test_data$time, params_i, 25) +
    rnorm(nrow(test_data), 0, p[4])
}

# Point predictions (posterior mean)
params_bayesian <- list(
  Linf = post_summary["mean", "Linf"],
  rB   = post_summary["mean", "rB"],
  L0   = post_summary["mean", "L0"],
  TA   = true_params$TA,
  Tref = true_params$Tref
)

train_data$pred_bayesian <- deb_growth(train_data$time, params_bayesian, 20)
test_data$pred_bayesian  <- deb_growth(test_data$time, params_bayesian, 25)

# Credible intervals
train_data$bayes_lo <- apply(pred_matrix_train, 1, quantile, 0.025)
train_data$bayes_hi <- apply(pred_matrix_train, 1, quantile, 0.975)
test_data$bayes_lo  <- apply(pred_matrix_test, 1, quantile, 0.025)
test_data$bayes_hi  <- apply(pred_matrix_test, 1, quantile, 0.975)

# Coverage
coverage_train <- mean(train_data$L_obs >= train_data$bayes_lo &
                       train_data$L_obs <= train_data$bayes_hi)
coverage_test  <- mean(test_data$L_obs >= test_data$bayes_lo &
                       test_data$L_obs <= test_data$bayes_hi)

cat("  95% CI coverage (20 C):", round(coverage_train, 3), "\n")
cat("  95% CI coverage (25 C):", round(coverage_test, 3), "\n\n")

## ============================================================
## 4. APPROACH 3: RANDOM FOREST
## ============================================================

cat("--- RANDOM FOREST ---\n")
t_rf_start <- Sys.time()

rf_model <- randomForest(
  L_obs ~ time + temp,
  data = train_data,
  ntree = 500,
  mtry = 1,
  importance = TRUE
)

t_rf_end <- Sys.time()
t_rf <- as.numeric(difftime(t_rf_end, t_rf_start, units = "secs"))

train_data$pred_rf <- predict(rf_model, train_data)
test_data$pred_rf  <- predict(rf_model, test_data)

cat("  OOB R-squared:", round(1 - rf_model$mse[500] / var(train_data$L_obs), 3), "\n")
cat("  Runtime:", round(t_rf, 3), "s\n")
cat("  Variable importance:\n")
print(round(importance(rf_model), 2))
cat("\n")

## ============================================================
## 5. APPROACH 4: XGBOOST (Gradient Boosting)
## ============================================================

cat("--- XGBOOST ---\n")
t_xgb_start <- Sys.time()

# Create DMatrix objects with features (time, temp)
dtrain_xgb <- xgb.DMatrix(
  data = as.matrix(train_data[, c("time", "temp")]),
  label = train_data$L_obs
)
dtest_xgb <- xgb.DMatrix(
  data = as.matrix(test_data[, c("time", "temp")]),
  label = test_data$L_obs
)

# Train XGBoost model
xgb_model <- xgb.train(
  params = list(
    max_depth = 4,
    eta = 0.1,
    objective = "reg:squarederror"
  ),
  data = dtrain_xgb,
  nrounds = 200,
  verbose = 0
)

t_xgb_end <- Sys.time()
t_xgb <- as.numeric(difftime(t_xgb_end, t_xgb_start, units = "secs"))

train_data$pred_xgb <- predict(xgb_model, dtrain_xgb)
test_data$pred_xgb  <- predict(xgb_model, dtest_xgb)

cat("  Runtime:", round(t_xgb, 3), "s\n\n")

## ============================================================
## 6. COMPUTE PERFORMANCE METRICS
## ============================================================

calc_metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  r2   <- 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  mae  <- mean(abs(obs - pred))
  return(c(RMSE = rmse, R2 = r2, MAE = mae))
}

results <- data.frame(
  Paradigm = rep(c("Classical DEB", "Bayesian DEB", "Random Forest", "XGBoost"), each = 2),
  Task = rep(c("Interpolation (20 C)", "Extrapolation (25 C)"), 4),
  RMSE = NA, R2 = NA, MAE = NA,
  Coverage_95 = NA, Runtime_s = NA
)

# Classical DEB
m <- calc_metrics(train_data$L_obs, train_data$pred_classical)
results[1, c("RMSE", "R2", "MAE")] <- m
results[1, "Runtime_s"] <- t_class
m <- calc_metrics(test_data$L_obs, test_data$pred_classical)
results[2, c("RMSE", "R2", "MAE")] <- m

# Bayesian DEB
m <- calc_metrics(train_data$L_obs, train_data$pred_bayesian)
results[3, c("RMSE", "R2", "MAE")] <- m
results[3, "Runtime_s"] <- t_bayes
results[3, "Coverage_95"] <- coverage_train
m <- calc_metrics(test_data$L_obs, test_data$pred_bayesian)
results[4, c("RMSE", "R2", "MAE")] <- m
results[4, "Coverage_95"] <- coverage_test

# Random Forest
m <- calc_metrics(train_data$L_obs, train_data$pred_rf)
results[5, c("RMSE", "R2", "MAE")] <- m
results[5, "Runtime_s"] <- t_rf
m <- calc_metrics(test_data$L_obs, test_data$pred_rf)
results[6, c("RMSE", "R2", "MAE")] <- m

# XGBoost
m <- calc_metrics(train_data$L_obs, train_data$pred_xgb)
results[7, c("RMSE", "R2", "MAE")] <- m
results[7, "Runtime_s"] <- t_xgb
m <- calc_metrics(test_data$L_obs, test_data$pred_xgb)
results[8, c("RMSE", "R2", "MAE")] <- m

# Print results
cat("============================================================\n")
cat("BENCHMARK RESULTS\n")
cat("============================================================\n\n")
results_print <- results
results_print[, 3:7] <- round(results_print[, 3:7], 4)
print(results_print, row.names = FALSE)

# Save to CSV
write.csv(results, "benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to benchmark_results.csv\n\n")

## ============================================================
## 7. GENERATE FIGURE
## ============================================================

# Prepare prediction curves for plotting
time_fine <- seq(0, 21, by = 0.1)

curve_classical_20 <- data.frame(
  time = time_fine,
  pred = deb_growth(time_fine, params_classical, 20),
  method = "Classical DEB"
)
curve_classical_25 <- data.frame(
  time = time_fine,
  pred = deb_growth(time_fine, params_classical, 25),
  method = "Classical DEB"
)
curve_bayesian_20 <- data.frame(
  time = time_fine,
  pred = deb_growth(time_fine, params_bayesian, 20),
  method = "Bayesian DEB"
)
curve_bayesian_25 <- data.frame(
  time = time_fine,
  pred = deb_growth(time_fine, params_bayesian, 25),
  method = "Bayesian DEB"
)

# Bayesian credible band (mean prediction)
bayes_band_20 <- data.frame(time = time_fine)
bayes_band_25 <- data.frame(time = time_fine)
band_matrix_20 <- matrix(NA, length(time_fine), n_post_samples)
band_matrix_25 <- matrix(NA, length(time_fine), n_post_samples)
for (i in 1:n_post_samples) {
  p <- posterior[idx[i], ]
  params_i <- list(Linf = p[1], rB = p[2], L0 = p[3],
                   TA = true_params$TA, Tref = true_params$Tref)
  band_matrix_20[, i] <- deb_growth(time_fine, params_i, 20)
  band_matrix_25[, i] <- deb_growth(time_fine, params_i, 25)
}
bayes_band_20$lo <- apply(band_matrix_20, 1, quantile, 0.025)
bayes_band_20$hi <- apply(band_matrix_20, 1, quantile, 0.975)
bayes_band_20$med <- apply(band_matrix_20, 1, median)
bayes_band_25$lo <- apply(band_matrix_25, 1, quantile, 0.025)
bayes_band_25$hi <- apply(band_matrix_25, 1, quantile, 0.975)
bayes_band_25$med <- apply(band_matrix_25, 1, median)

# RF predictions on fine grid (only works for 20 C training domain)
rf_fine_20 <- data.frame(time = time_fine, temp = 20)
rf_fine_25 <- data.frame(time = time_fine, temp = 25)
rf_fine_20$pred <- predict(rf_model, rf_fine_20)
rf_fine_25$pred <- predict(rf_model, rf_fine_25)

# XGBoost predictions on fine grid
xgb_fine_20 <- xgb.DMatrix(data = as.matrix(data.frame(time = time_fine, temp = 20)))
xgb_fine_25 <- xgb.DMatrix(data = as.matrix(data.frame(time = time_fine, temp = 25)))
xgb_fine_20_pred <- predict(xgb_model, xgb_fine_20)
xgb_fine_25_pred <- predict(xgb_model, xgb_fine_25)
xgb_curve_20 <- data.frame(time = time_fine, pred = xgb_fine_20_pred)
xgb_curve_25 <- data.frame(time = time_fine, pred = xgb_fine_25_pred)

# True curves
true_20 <- data.frame(time = time_fine,
                       L = deb_growth(time_fine, true_params, 20))
true_25 <- data.frame(time = time_fine,
                       L = deb_growth(time_fine, true_params, 25))

# Panel A: Interpolation (20 C)
p1 <- ggplot() +
  geom_ribbon(data = bayes_band_20,
              aes(x = time, ymin = lo, ymax = hi),
              fill = "darkorange", alpha = 0.2) +
  geom_line(data = true_20, aes(x = time, y = L),
            colour = "grey50", linetype = "dashed", linewidth = 0.6) +
  geom_line(data = curve_classical_20, aes(x = time, y = pred),
            colour = "steelblue", linewidth = 0.8) +
  geom_line(data = bayes_band_20, aes(x = time, y = med),
            colour = "darkorange", linewidth = 0.8) +
  geom_line(data = rf_fine_20, aes(x = time, y = pred),
            colour = "forestgreen", linewidth = 0.8) +
  geom_line(data = xgb_curve_20, aes(x = time, y = pred),
            colour = "purple", linewidth = 0.8) +
  geom_point(data = train_data, aes(x = time, y = L_obs),
             size = 1, alpha = 0.4) +
  labs(x = "Time (days)", y = "Body length (mm)",
       title = "A) Interpolation (20 C)") +
  annotate("text", x = 1, y = max(train_data$L_obs) * 0.98,
           label = "Dashed: true model", size = 2.5, hjust = 0) +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10))

# Panel B: Extrapolation (25 C)
p2 <- ggplot() +
  geom_ribbon(data = bayes_band_25,
              aes(x = time, ymin = lo, ymax = hi),
              fill = "darkorange", alpha = 0.2) +
  geom_line(data = true_25, aes(x = time, y = L),
            colour = "grey50", linetype = "dashed", linewidth = 0.6) +
  geom_line(data = curve_classical_25, aes(x = time, y = pred),
            colour = "steelblue", linewidth = 0.8) +
  geom_line(data = bayes_band_25, aes(x = time, y = med),
            colour = "darkorange", linewidth = 0.8) +
  geom_line(data = rf_fine_25, aes(x = time, y = pred),
            colour = "forestgreen", linewidth = 0.8) +
  geom_line(data = xgb_curve_25, aes(x = time, y = pred),
            colour = "purple", linewidth = 0.8) +
  geom_point(data = test_data, aes(x = time, y = L_obs),
             size = 1, alpha = 0.4) +
  labs(x = "Time (days)", y = "Body length (mm)",
       title = "B) Extrapolation (25 C)") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10))

# Panel C: RMSE comparison
results_plot <- results %>%
  select(Paradigm, Task, RMSE) %>%
  mutate(Task = factor(Task, levels = c("Interpolation (20 C)",
                                         "Extrapolation (25 C)")))

p3 <- ggplot(results_plot, aes(x = Paradigm, y = RMSE, fill = Task)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = c("grey70", "grey30")) +
  labs(x = "", y = "RMSE (mm)", title = "C) Performance comparison",
       fill = "") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10),
        legend.position = "bottom",
        axis.text.x = element_text(angle = 15, hjust = 1))

# Combine panels
fig <- (p1 | p2) / p3 +
  plot_annotation(
    caption = paste0(
      "Blue: Classical DEB. Orange: Bayesian DEB (band = 95% credible interval). ",
      "Green: Random Forest. Purple: XGBoost. Dashed grey: true generating model."
    ),
    theme = theme(plot.caption = element_text(size = 8, hjust = 0))
  )

ggsave("benchmark_figure.pdf", fig, width = 8, height = 7, dpi = 300)
ggsave("benchmark_figure.png", fig, width = 8, height = 7, dpi = 300)

cat("Figure saved to benchmark_figure.pdf and benchmark_figure.png\n")

## ============================================================
## 8. CONVERGENCE DIAGNOSTICS (Bayesian)
## ============================================================

cat("\n--- MCMC DIAGNOSTICS ---\n")
cat("Gelman-Rubin R-hat:\n")
gel <- gelmanDiagnostics(fit_bayesian)
print(round(gel$psrf, 4))

cat("\nEffective sample size:\n")
if (requireNamespace("coda", quietly = TRUE)) {
  ess <- coda::effectiveSize(getSample(fit_bayesian, parametersOnly = TRUE, coda = TRUE))
  print(round(ess))
} else {
  cat("  (coda package not available, skipping ESS computation)\n")
}

## ============================================================
## 9. SCENARIO 2: MULTI-TEMPERATURE TRAINING
## ============================================================
## Does richer training data (multiple temperatures) improve ML
## extrapolation to unseen temperatures?  We train on 15, 20, 25 C
## and test at 28 C (outside all training temperatures).

cat("\n============================================================\n")
cat("SCENARIO 2: Multi-temperature training\n")
cat("============================================================\n\n")

# Generate additional training data at 15 C and 25 C
train_15 <- expand.grid(time = times_train, rep = 1:n_rep) %>%
  mutate(
    temp = 15,
    L_true = deb_growth(time, true_params, temp_C = 15),
    L_obs = L_true + rnorm(n(), 0, true_params$sigma),
    L_obs = pmax(L_obs, 0.3)
  )

train_25 <- expand.grid(time = times_train, rep = 1:n_rep) %>%
  mutate(
    temp = 25,
    L_true = deb_growth(time, true_params, temp_C = 25),
    L_obs = L_true + rnorm(n(), 0, true_params$sigma),
    L_obs = pmax(L_obs, 0.3)
  )

# Combine into multi-temperature training set
train_multi <- bind_rows(train_data, train_15, train_25)

# Test extrapolation at 28 C (outside all training temperatures)
test_28 <- expand.grid(time = times_train, rep = 1:n_rep) %>%
  mutate(
    temp = 28,
    L_true = deb_growth(time, true_params, temp_C = 28),
    L_obs = L_true + rnorm(n(), 0, true_params$sigma),
    L_obs = pmax(L_obs, 0.3)
  )

cat("Multi-temp training: n =", nrow(train_multi),
    "(15, 20, 25 C combined)\n")
cat("Multi-temp test:     n =", nrow(test_28), "(28 C)\n\n")

# --- Classical DEB on multi-temperature data ---
cat("--- CLASSICAL DEB (multi-temp) ---\n")
classical_obj_multi <- function(par, data) {
  Linf <- par[1]
  rB   <- par[2]
  L0   <- par[3]
  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params$TA, Tref = true_params$Tref)
  pred <- numeric(nrow(data))
  for (tc in unique(data$temp)) {
    idx_tc <- data$temp == tc
    pred[idx_tc] <- deb_growth(data$time[idx_tc], params_local, temp_C = tc)
  }
  return(sum((data$L_obs - pred)^2))
}

t_class_m_start <- Sys.time()
fit_classical_multi <- optim(
  par = c(Linf = 5.0, rB = 0.12, L0 = 0.9),
  fn = classical_obj_multi,
  data = train_multi,
  method = "Nelder-Mead",
  control = list(maxit = 5000)
)
t_class_m_end <- Sys.time()
t_class_m <- as.numeric(difftime(t_class_m_end, t_class_m_start, units = "secs"))

params_classical_m <- list(
  Linf = fit_classical_multi$par[1],
  rB   = fit_classical_multi$par[2],
  L0   = fit_classical_multi$par[3],
  TA   = true_params$TA,
  Tref = true_params$Tref
)

train_multi$pred_classical <- NA
for (tc in unique(train_multi$temp)) {
  idx_tc <- train_multi$temp == tc
  train_multi$pred_classical[idx_tc] <- deb_growth(
    train_multi$time[idx_tc], params_classical_m, temp_C = tc)
}
test_28$pred_classical <- deb_growth(test_28$time, params_classical_m, 28)

cat("  Linf =", round(params_classical_m$Linf, 3), "mm\n")
cat("  rB   =", round(params_classical_m$rB, 4), "d^-1\n")
cat("  L0   =", round(params_classical_m$L0, 3), "mm\n")
cat("  Runtime:", round(t_class_m, 3), "s\n\n")

# --- Bayesian DEB on multi-temperature data ---
cat("--- BAYESIAN DEB (multi-temp) ---\n")
cat("  Running MCMC (DEzs, 3 chains x 10000 iterations)...\n")

log_likelihood_multi <- function(par) {
  Linf  <- par[1]
  rB    <- par[2]
  L0    <- par[3]
  sigma <- par[4]

  if (Linf <= 0 | rB <= 0 | L0 <= 0 | sigma <= 0) return(-Inf)
  if (L0 >= Linf) return(-Inf)

  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params$TA, Tref = true_params$Tref)
  pred <- numeric(nrow(train_multi))
  for (tc in unique(train_multi$temp)) {
    idx_tc <- train_multi$temp == tc
    pred[idx_tc] <- deb_growth(train_multi$time[idx_tc], params_local, temp_C = tc)
  }
  ll <- sum(dnorm(train_multi$L_obs, mean = pred, sd = sigma, log = TRUE))
  return(ll)
}

bayesian_setup_multi <- createBayesianSetup(
  likelihood = log_likelihood_multi,
  prior = createPrior(
    density = log_prior,
    sampler = function() c(
      exp(rnorm(1, log(4.8), 0.2)),
      exp(rnorm(1, log(0.15), 0.3)),
      exp(rnorm(1, log(0.8), 0.2)),
      runif(1, 0.05, 0.5)
    ),
    lower = c(1, 0.01, 0.1, 0.01),
    upper = c(10, 1.0, 3.0, 1.0)
  ),
  names = c("Linf", "rB", "L0", "sigma")
)

t_bayes_m_start <- Sys.time()
fit_bayesian_multi <- runMCMC(
  bayesianSetup = bayesian_setup_multi,
  sampler = "DEzs",
  settings = list(
    iterations = 10000,
    burnin = 3000,
    thin = 2,
    nrChains = 3,
    message = FALSE
  )
)
t_bayes_m_end <- Sys.time()
t_bayes_m <- as.numeric(difftime(t_bayes_m_end, t_bayes_m_start, units = "secs"))

posterior_m <- getSample(fit_bayesian_multi, parametersOnly = TRUE)
post_summary_m <- apply(posterior_m, 2, function(x)
  c(mean = mean(x), sd = sd(x),
    q025 = quantile(x, 0.025), q975 = quantile(x, 0.975)))

cat("  Posterior summary:\n")
print(round(post_summary_m, 4))
cat("  Runtime:", round(t_bayes_m, 1), "s\n")

params_bayesian_m <- list(
  Linf = post_summary_m["mean", "Linf"],
  rB   = post_summary_m["mean", "rB"],
  L0   = post_summary_m["mean", "L0"],
  TA   = true_params$TA,
  Tref = true_params$Tref
)

train_multi$pred_bayesian <- NA
for (tc in unique(train_multi$temp)) {
  idx_tc <- train_multi$temp == tc
  train_multi$pred_bayesian[idx_tc] <- deb_growth(
    train_multi$time[idx_tc], params_bayesian_m, temp_C = tc)
}
test_28$pred_bayesian <- deb_growth(test_28$time, params_bayesian_m, 28)

# Bayesian coverage at 28 C
n_post_samples_m <- min(500, nrow(posterior_m))
idx_m <- sample(nrow(posterior_m), n_post_samples_m)
pred_matrix_28 <- matrix(NA, nrow = nrow(test_28), ncol = n_post_samples_m)
for (i in 1:n_post_samples_m) {
  p <- posterior_m[idx_m[i], ]
  params_i <- list(Linf = p[1], rB = p[2], L0 = p[3],
                   TA = true_params$TA, Tref = true_params$Tref)
  pred_matrix_28[, i] <- deb_growth(test_28$time, params_i, 28) +
    rnorm(nrow(test_28), 0, p[4])
}
test_28$bayes_lo <- apply(pred_matrix_28, 1, quantile, 0.025)
test_28$bayes_hi <- apply(pred_matrix_28, 1, quantile, 0.975)
coverage_28 <- mean(test_28$L_obs >= test_28$bayes_lo &
                    test_28$L_obs <= test_28$bayes_hi)
cat("  95% CI coverage (28 C):", round(coverage_28, 3), "\n\n")

# --- Random Forest on multi-temperature data ---
cat("--- RANDOM FOREST (multi-temp) ---\n")
t_rf_m_start <- Sys.time()
rf_model_multi <- randomForest(
  L_obs ~ time + temp,
  data = train_multi,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)
t_rf_m_end <- Sys.time()
t_rf_m <- as.numeric(difftime(t_rf_m_end, t_rf_m_start, units = "secs"))

train_multi$pred_rf <- predict(rf_model_multi, train_multi)
test_28$pred_rf     <- predict(rf_model_multi, test_28)

cat("  OOB R-squared:", round(1 - rf_model_multi$mse[500] / var(train_multi$L_obs), 3), "\n")
cat("  Runtime:", round(t_rf_m, 3), "s\n\n")

# --- XGBoost on multi-temperature data ---
cat("--- XGBOOST (multi-temp) ---\n")
t_xgb_m_start <- Sys.time()

dtrain_multi_xgb <- xgb.DMatrix(
  data = as.matrix(train_multi[, c("time", "temp")]),
  label = train_multi$L_obs
)
dtest_28_xgb <- xgb.DMatrix(
  data = as.matrix(test_28[, c("time", "temp")]),
  label = test_28$L_obs
)

xgb_model_multi <- xgb.train(
  params = list(
    max_depth = 4,
    eta = 0.1,
    objective = "reg:squarederror"
  ),
  data = dtrain_multi_xgb,
  nrounds = 200,
  verbose = 0
)

t_xgb_m_end <- Sys.time()
t_xgb_m <- as.numeric(difftime(t_xgb_m_end, t_xgb_m_start, units = "secs"))

train_multi$pred_xgb <- predict(xgb_model_multi, dtrain_multi_xgb)
test_28$pred_xgb     <- predict(xgb_model_multi, dtest_28_xgb)

cat("  Runtime:", round(t_xgb_m, 3), "s\n\n")

# --- Compute multi-temp metrics ---
results_multi <- data.frame(
  Paradigm = rep(c("Classical DEB", "Bayesian DEB", "Random Forest", "XGBoost"), each = 2),
  Task = rep(c("Interpolation (15+20+25 C)", "Extrapolation (28 C)"), 4),
  RMSE = NA, R2 = NA, MAE = NA,
  Coverage_95 = NA, Runtime_s = NA
)

# Classical DEB
m <- calc_metrics(train_multi$L_obs, train_multi$pred_classical)
results_multi[1, c("RMSE", "R2", "MAE")] <- m
results_multi[1, "Runtime_s"] <- t_class_m
m <- calc_metrics(test_28$L_obs, test_28$pred_classical)
results_multi[2, c("RMSE", "R2", "MAE")] <- m

# Bayesian DEB
m <- calc_metrics(train_multi$L_obs, train_multi$pred_bayesian)
results_multi[3, c("RMSE", "R2", "MAE")] <- m
results_multi[3, "Runtime_s"] <- t_bayes_m
m <- calc_metrics(test_28$L_obs, test_28$pred_bayesian)
results_multi[4, c("RMSE", "R2", "MAE")] <- m
results_multi[4, "Coverage_95"] <- coverage_28

# Random Forest
m <- calc_metrics(train_multi$L_obs, train_multi$pred_rf)
results_multi[5, c("RMSE", "R2", "MAE")] <- m
results_multi[5, "Runtime_s"] <- t_rf_m
m <- calc_metrics(test_28$L_obs, test_28$pred_rf)
results_multi[6, c("RMSE", "R2", "MAE")] <- m

# XGBoost
m <- calc_metrics(train_multi$L_obs, train_multi$pred_xgb)
results_multi[7, c("RMSE", "R2", "MAE")] <- m
results_multi[7, "Runtime_s"] <- t_xgb_m
m <- calc_metrics(test_28$L_obs, test_28$pred_xgb)
results_multi[8, c("RMSE", "R2", "MAE")] <- m

# Print multi-temp results
cat("============================================================\n")
cat("SCENARIO 2: MULTI-TEMPERATURE BENCHMARK RESULTS\n")
cat("Training: 15, 20, 25 C | Testing: 28 C\n")
cat("============================================================\n\n")
results_multi_print <- results_multi
results_multi_print[, 3:7] <- round(results_multi_print[, 3:7], 4)
print(results_multi_print, row.names = FALSE)

# Save multi-temp results
write.csv(results_multi, "benchmark_results_multitemp.csv", row.names = FALSE)
cat("\nResults saved to benchmark_results_multitemp.csv\n")

# --- Compare Scenario 1 vs Scenario 2 for ML approaches ---
cat("\n============================================================\n")
cat("COMPARISON: Effect of multi-temperature training on ML\n")
cat("============================================================\n\n")

# Extract extrapolation RMSE from both scenarios
scen1_rf   <- results$RMSE[results$Paradigm == "Random Forest" &
                            results$Task == "Extrapolation (25 C)"]
scen1_xgb  <- results$RMSE[results$Paradigm == "XGBoost" &
                            results$Task == "Extrapolation (25 C)"]
scen2_rf   <- results_multi$RMSE[results_multi$Paradigm == "Random Forest" &
                                  results_multi$Task == "Extrapolation (28 C)"]
scen2_xgb  <- results_multi$RMSE[results_multi$Paradigm == "XGBoost" &
                                  results_multi$Task == "Extrapolation (28 C)"]

cat("Random Forest extrapolation RMSE:\n")
cat("  Scenario 1 (train 20 C -> test 25 C):", round(scen1_rf, 4), "mm\n")
cat("  Scenario 2 (train 15+20+25 C -> test 28 C):", round(scen2_rf, 4), "mm\n")
cat("  Change:", ifelse(scen2_rf < scen1_rf, "IMPROVED", "WORSENED"),
    "(", round(abs(scen2_rf - scen1_rf) / scen1_rf * 100, 1), "% )\n\n")

cat("XGBoost extrapolation RMSE:\n")
cat("  Scenario 1 (train 20 C -> test 25 C):", round(scen1_xgb, 4), "mm\n")
cat("  Scenario 2 (train 15+20+25 C -> test 28 C):", round(scen2_xgb, 4), "mm\n")
cat("  Change:", ifelse(scen2_xgb < scen1_xgb, "IMPROVED", "WORSENED"),
    "(", round(abs(scen2_xgb - scen1_xgb) / scen1_xgb * 100, 1), "% )\n\n")

cat("Key insight: Multi-temperature training allows ML approaches to learn\n")
cat("temperature-dependence from data, potentially improving extrapolation\n")
cat("performance to novel temperatures.\n")

## ============================================================
## 10. SCENARIO 3: MODEL MISSPECIFICATION
## ============================================================
## This scenario demonstrates that when the mechanistic model is
## misspecified (true data-generating process violates constant-kappa
## assumption), ML methods can outperform DEB because they are not
## constrained by incorrect assumptions.
##
## True model: kappa (allocation fraction) decreases linearly with
## age from 0.85 to 0.55 over 21 days, simulating an ontogenetic
## shift in energy allocation.  The standard DEB model assumes
## kappa is constant, so fitting it to this data is misspecified.

cat("\n============================================================\n")
cat("SCENARIO 3: Model misspecification (variable kappa)\n")
cat("============================================================\n\n")

set.seed(42)

# True model: kappa decreases with age (simulating ontogenetic shift)
deb_growth_variable_kappa <- function(t, params, temp_C = 20) {
  TC <- exp(params$TA / params$Tref - params$TA / (temp_C + 273.15))
  # kappa decreases from 0.85 to 0.55 over 21 days
  kappa_t <- 0.85 - 0.30 * (t / 21)
  # This modulates Linf dynamically
  Linf_t <- kappa_t * params$Linf_max
  rB_corr <- params$rB * TC
  L <- Linf_t - (Linf_t - params$L0) * exp(-rB_corr * t)
  return(L)
}

# Parameters for the variable-kappa generating model
true_params_misspec <- list(
  Linf_max = 6.5,    # mm, maximum structural length (at kappa=1)
  rB       = 0.15,   # d^-1, von Bertalanffy growth rate
  L0       = 0.8,    # mm, length at birth (neonates)
  TA       = 8000,   # K, Arrhenius temperature
  Tref     = 293.15, # K, reference temperature (20 C)
  sigma    = 0.12    # mm, observation noise (sd)
)

# Generate training data at 20 C (same structure as Scenario 1)
times_train_s3 <- seq(0, 21, by = 1.5)  # 15 time points, 0-21 days
n_rep_s3 <- 5

train_misspec <- expand.grid(
  time = times_train_s3,
  rep = 1:n_rep_s3
) %>%
  mutate(
    temp = 20,
    L_true = deb_growth_variable_kappa(time, true_params_misspec, temp_C = 20),
    L_obs = L_true + rnorm(n(), 0, true_params_misspec$sigma),
    L_obs = pmax(L_obs, 0.3)  # enforce positivity
  )

cat("Misspec training data: n =", nrow(train_misspec), "(20 C, variable kappa)\n\n")

# --- Classical DEB (misspecified: assumes constant kappa) ---
cat("--- CLASSICAL DEB (misspecified) ---\n")

classical_obj_misspec <- function(par, data) {
  Linf <- par[1]
  rB   <- par[2]
  L0   <- par[3]
  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params_misspec$TA,
                       Tref = true_params_misspec$Tref)
  pred <- deb_growth(data$time, params_local, temp_C = data$temp[1])
  return(sum((data$L_obs - pred)^2))
}

t_class_s3_start <- Sys.time()
fit_classical_misspec <- optim(
  par = c(Linf = 5.0, rB = 0.12, L0 = 0.9),
  fn = classical_obj_misspec,
  data = train_misspec,
  method = "Nelder-Mead",
  control = list(maxit = 5000)
)
t_class_s3_end <- Sys.time()
t_class_s3 <- as.numeric(difftime(t_class_s3_end, t_class_s3_start, units = "secs"))

params_classical_s3 <- list(
  Linf = fit_classical_misspec$par[1],
  rB   = fit_classical_misspec$par[2],
  L0   = fit_classical_misspec$par[3],
  TA   = true_params_misspec$TA,
  Tref = true_params_misspec$Tref
)

train_misspec$pred_classical <- deb_growth(train_misspec$time, params_classical_s3, 20)

cat("  Linf =", round(params_classical_s3$Linf, 3), "mm\n")
cat("  rB   =", round(params_classical_s3$rB, 4), "d^-1\n")
cat("  L0   =", round(params_classical_s3$L0, 3), "mm\n")
cat("  Runtime:", round(t_class_s3, 3), "s\n\n")

# --- Bayesian DEB (also misspecified: assumes constant kappa) ---
cat("--- BAYESIAN DEB (misspecified) ---\n")
cat("  Running MCMC (DEzs, 3 chains x 10000 iterations)...\n")

log_likelihood_misspec <- function(par) {
  Linf  <- par[1]
  rB    <- par[2]
  L0    <- par[3]
  sigma <- par[4]

  if (Linf <= 0 | rB <= 0 | L0 <= 0 | sigma <= 0) return(-Inf)
  if (L0 >= Linf) return(-Inf)

  params_local <- list(Linf = Linf, rB = rB, L0 = L0,
                       TA = true_params_misspec$TA,
                       Tref = true_params_misspec$Tref)
  pred <- deb_growth(train_misspec$time, params_local, temp_C = 20)
  ll <- sum(dnorm(train_misspec$L_obs, mean = pred, sd = sigma, log = TRUE))
  return(ll)
}

bayesian_setup_misspec <- createBayesianSetup(
  likelihood = log_likelihood_misspec,
  prior = createPrior(
    density = log_prior,
    sampler = function() c(
      exp(rnorm(1, log(4.8), 0.2)),
      exp(rnorm(1, log(0.15), 0.3)),
      exp(rnorm(1, log(0.8), 0.2)),
      runif(1, 0.05, 0.5)
    ),
    lower = c(1, 0.01, 0.1, 0.01),
    upper = c(10, 1.0, 3.0, 1.0)
  ),
  names = c("Linf", "rB", "L0", "sigma")
)

t_bayes_s3_start <- Sys.time()
fit_bayesian_misspec <- runMCMC(
  bayesianSetup = bayesian_setup_misspec,
  sampler = "DEzs",
  settings = list(
    iterations = 10000,
    burnin = 3000,
    thin = 2,
    nrChains = 3,
    message = FALSE
  )
)
t_bayes_s3_end <- Sys.time()
t_bayes_s3 <- as.numeric(difftime(t_bayes_s3_end, t_bayes_s3_start, units = "secs"))

posterior_s3 <- getSample(fit_bayesian_misspec, parametersOnly = TRUE)
post_summary_s3 <- apply(posterior_s3, 2, function(x)
  c(mean = mean(x), sd = sd(x),
    q025 = quantile(x, 0.025), q975 = quantile(x, 0.975)))

cat("  Posterior summary:\n")
print(round(post_summary_s3, 4))
cat("  Runtime:", round(t_bayes_s3, 1), "s\n")

params_bayesian_s3 <- list(
  Linf = post_summary_s3["mean", "Linf"],
  rB   = post_summary_s3["mean", "rB"],
  L0   = post_summary_s3["mean", "L0"],
  TA   = true_params_misspec$TA,
  Tref = true_params_misspec$Tref
)

train_misspec$pred_bayesian <- deb_growth(train_misspec$time, params_bayesian_s3, 20)
cat("\n")

# --- Random Forest (no structural assumptions) ---
cat("--- RANDOM FOREST (misspec) ---\n")
t_rf_s3_start <- Sys.time()

rf_model_misspec <- randomForest(
  L_obs ~ time + temp,
  data = train_misspec,
  ntree = 500,
  mtry = 1,
  importance = TRUE
)

t_rf_s3_end <- Sys.time()
t_rf_s3 <- as.numeric(difftime(t_rf_s3_end, t_rf_s3_start, units = "secs"))

train_misspec$pred_rf <- predict(rf_model_misspec, train_misspec)

cat("  OOB R-squared:", round(1 - rf_model_misspec$mse[500] / var(train_misspec$L_obs), 3), "\n")
cat("  Runtime:", round(t_rf_s3, 3), "s\n\n")

# --- XGBoost (no structural assumptions) ---
cat("--- XGBOOST (misspec) ---\n")
t_xgb_s3_start <- Sys.time()

dtrain_misspec_xgb <- xgb.DMatrix(
  data = as.matrix(train_misspec[, c("time", "temp")]),
  label = train_misspec$L_obs
)

xgb_model_misspec <- xgb.train(
  params = list(
    max_depth = 4,
    eta = 0.1,
    objective = "reg:squarederror"
  ),
  data = dtrain_misspec_xgb,
  nrounds = 200,
  verbose = 0
)

t_xgb_s3_end <- Sys.time()
t_xgb_s3 <- as.numeric(difftime(t_xgb_s3_end, t_xgb_s3_start, units = "secs"))

train_misspec$pred_xgb <- predict(xgb_model_misspec, dtrain_misspec_xgb)

cat("  Runtime:", round(t_xgb_s3, 3), "s\n\n")

# --- Compute misspecification metrics (interpolation only) ---
results_misspec <- data.frame(
  Paradigm = c("Classical DEB", "Bayesian DEB", "Random Forest", "XGBoost"),
  Task = rep("Interpolation (20 C, variable kappa)", 4),
  RMSE = NA, R2 = NA, MAE = NA,
  Runtime_s = NA
)

m <- calc_metrics(train_misspec$L_obs, train_misspec$pred_classical)
results_misspec[1, c("RMSE", "R2", "MAE")] <- m
results_misspec[1, "Runtime_s"] <- t_class_s3

m <- calc_metrics(train_misspec$L_obs, train_misspec$pred_bayesian)
results_misspec[2, c("RMSE", "R2", "MAE")] <- m
results_misspec[2, "Runtime_s"] <- t_bayes_s3

m <- calc_metrics(train_misspec$L_obs, train_misspec$pred_rf)
results_misspec[3, c("RMSE", "R2", "MAE")] <- m
results_misspec[3, "Runtime_s"] <- t_rf_s3

m <- calc_metrics(train_misspec$L_obs, train_misspec$pred_xgb)
results_misspec[4, c("RMSE", "R2", "MAE")] <- m
results_misspec[4, "Runtime_s"] <- t_xgb_s3

# Print misspec results
cat("============================================================\n")
cat("SCENARIO 3: MODEL MISSPECIFICATION BENCHMARK RESULTS\n")
cat("True DGP: variable kappa (0.85 -> 0.55 over 21 days)\n")
cat("DEB fit: standard constant-kappa von Bertalanffy (misspecified)\n")
cat("============================================================\n\n")
results_misspec_print <- results_misspec
results_misspec_print[, 3:6] <- round(results_misspec_print[, 3:6], 4)
print(results_misspec_print, row.names = FALSE)

# Save misspec results
write.csv(results_misspec, "benchmark_results_misspec.csv", row.names = FALSE)
cat("\nResults saved to benchmark_results_misspec.csv\n")

# --- Comparison: which approach wins under misspecification? ---
best_idx <- which.min(results_misspec$RMSE)
worst_idx <- which.max(results_misspec$RMSE)

cat("\n--- Misspecification comparison ---\n")
cat("Best  RMSE:", round(results_misspec$RMSE[best_idx], 4),
    "mm (", results_misspec$Paradigm[best_idx], ")\n")
cat("Worst RMSE:", round(results_misspec$RMSE[worst_idx], 4),
    "mm (", results_misspec$Paradigm[worst_idx], ")\n")

deb_rmse <- results_misspec$RMSE[results_misspec$Paradigm == "Classical DEB"]
rf_rmse  <- results_misspec$RMSE[results_misspec$Paradigm == "Random Forest"]
xgb_rmse <- results_misspec$RMSE[results_misspec$Paradigm == "XGBoost"]

cat("\nML vs Classical DEB improvement:\n")
cat("  Random Forest: ", round((deb_rmse - rf_rmse) / deb_rmse * 100, 1),
    "% reduction in RMSE\n")
cat("  XGBoost:       ", round((deb_rmse - xgb_rmse) / deb_rmse * 100, 1),
    "% reduction in RMSE\n")

cat("\nKey insight: When the mechanistic model is misspecified (constant-kappa\n")
cat("DEB fitted to variable-kappa data), ML methods can outperform DEB\n")
cat("because they are not constrained by incorrect structural assumptions.\n")

cat("\n============================================================\n")
cat("BENCHMARK COMPLETE\n")
cat("Files generated:\n")
cat("  benchmark_results.csv           - Scenario 1 results table\n")
cat("  benchmark_results_multitemp.csv - Scenario 2 results table\n")
cat("  benchmark_results_misspec.csv   - Scenario 3 results table\n")
cat("  benchmark_figure.pdf            - four-approach comparison figure\n")
cat("  benchmark_figure.png            - same, PNG format\n")
cat("============================================================\n")
