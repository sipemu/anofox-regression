# Reference values for solvers::eb_shrink (issue #58).
#
# DerSimonian-Laird random-effects meta-analysis and empirical-Bayes (BLUP)
# shrinkage of each group's estimate:
#   rma(yi, sei = sei, method = "DL")  -> mu, se(mu), tau^2, Q, I^2
#   blup(fit)                           -> shrunken estimates and their SEs
#   rma(..., method = "FE")             -> complete pooling (tau^2 = 0)
#   rma(..., tau2 = <fixed>)            -> fixed tau^2
#
# Note: metafor's blup() se is sqrt(w*se^2*(1-w)... ) including the uncertainty in
# mu; the crate (like the extension) reports the conditional posterior SD
# sqrt(1 / (1/se^2 + 1/tau^2)), which ignores the uncertainty of mu. The base-R
# lines below compute that quantity directly.
#
# Run: Rscript tests/r_scripts/generate_eb_shrink_validation.R

suppressPackageStartupMessages(library(metafor))
options(digits = 17)

show <- function(name, v) cat(name, "=", paste(sprintf("%.15e", v), collapse = ", "), "\n")

run <- function(label, yi, sei, method = "DL", tau2 = NULL) {
  cat("##", label, "\n")
  fit <- if (is.null(tau2)) rma(yi, sei = sei, method = method) else
    rma(yi, sei = sei, tau2 = tau2, method = "DL")
  fe <- rma(yi, sei = sei, method = "FE")
  show("mu", as.numeric(fit$b))
  show("mu_se", fit$se)
  show("tau2", fit$tau2)
  show("q", fe$QE)
  show("i2", fit$I2 / 100)
  t2 <- fit$tau2
  if (t2 > 0) {
    b <- blup(fit)
    show("shrunken", b$pred)
    show("weight", t2 / (t2 + sei^2))
    show("shrunken_se", sqrt(1 / (1 / sei^2 + 1 / t2)))
  }
}

yi  <- c(0.10, 0.30, 0.35, 0.65, 1.00)
sei <- c(0.30, 0.10, 0.50, 0.20, 0.40)
run("fixture DL", yi, sei)
run("fixture FE (TauMethod::None)", yi, sei, method = "FE")
run("fixture fixed tau2 = 0.05", yi, sei, tau2 = 0.05)

# BCG vaccine trials (metafor::dat.bcg), log risk ratios.
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
show("bcg_yi", dat$yi)
show("bcg_sei", sqrt(dat$vi))
run("bcg DL", dat$yi, sqrt(dat$vi))
