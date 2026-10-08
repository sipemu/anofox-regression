#!/usr/bin/env Rscript
# =============================================================================
# GLMM fixed-effect Wald inference references (issue: z / p / Wald CI upstream).
#
# Reads the scenario data from tests/fixtures/glmm_validation.rs (so the inputs
# are byte-identical to the Rust tests) and prints, per scenario, lme4's
#   z = estimate / SE, p = 2 * pnorm(-|z|)       (summary(glmer) "Pr(>|z|)")
#   confint(m, parm = "beta_", method = "Wald", level = 0.95 / 0.90)
#
#   Rscript tests/r_scripts/generate_glmm_inference_validation.R
# =============================================================================
suppressPackageStartupMessages(library(lme4))
src <- readLines("tests/fixtures/glmm_validation.rs")
arr <- function(name) {
  line <- grep(paste0("^const ", name, ":"), src, value = TRUE)
  as.numeric(strsplit(sub(".*= \\[(.*)\\];", "\\1", line), ",\\s*")[[1]])
}
show <- function(tag, m) {
  b <- fixef(m); se <- sqrt(diag(as.matrix(vcov(m))))
  z <- b / se; p <- 2 * pnorm(-abs(z))
  ci95 <- confint(m, parm = "beta_", method = "Wald", level = 0.95)
  ci90 <- confint(m, parm = "beta_", method = "Wald", level = 0.90)
  cat(sprintf("// %s  (lme4 %s)\n", tag, packageVersion("lme4")))
  f <- function(nm, v) cat(sprintf("const %s_%s: [f64; %d] = [%s];\n", tag, nm, length(v),
                                   paste(sprintf("%.12e", v), collapse = ", ")))
  f("Z", z); f("P", p)
  f("CI95_LO", ci95[, 1]); f("CI95_HI", ci95[, 2])
  f("CI90_LO", ci90[, 1]); f("CI90_HI", ci90[, 2])
}
d <- data.frame(x = arr("X_LMM_A"), y = arr("Y_LMM_A"), g = factor(arr("G_LMM_A")))
show("LMM_A", lmer(y ~ x + (1 | g), data = d, REML = TRUE,
                   control = lmerControl(calc.derivs = FALSE)))
d <- data.frame(x = arr("X_POIS_C"), y = arr("Y_POIS_C"), g = factor(arr("G_POIS_C")))
show("POIS_C", glmer(y ~ x + (1 | g), data = d, family = poisson, nAGQ = 0))
d <- data.frame(x = arr("X_BIN_D"), y = arr("Y_BIN_D"), g = factor(arr("G_BIN_D")))
show("BIN_D", glmer(y ~ x + (1 | g), data = d, family = binomial, nAGQ = 0))
