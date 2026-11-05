#!/usr/bin/env Rscript
# Validation data for `diagnostics::accuracy` (#102) against Hmisc::somers2,
# Hmisc::rcorr.cens, Hmisc::GiniMd, Hmisc::R2Measures,
# survival::concordancefit, rms::lrm (stats vector) and rms::val.prob.
#
#   Rscript tests/r_scripts/generate_accuracy_validation.R > tests/fixtures/accuracy_validation.rs

suppressMessages({ library(Hmisc); library(rms); library(survival) })

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  num <- as.numeric(v)
  txt <- ifelse(is.infinite(num), ifelse(num > 0, "f64::INFINITY", "f64::NEG_INFINITY"),
                sprintf("%.17e", num))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}
scalar <- function(name, v) cat(sprintf("const %s: f64 = %.17e;\n", name, as.numeric(v)))

set.seed(20261028)
n <- 200
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rbinom(n, 1, plogis(-0.3 + 1.2 * x1 - 0.8 * x2 + 0.4 * x1^2))
w <- round(runif(n, 0.5, 2), 2)
emit("X1", x1); emit("X2", x2); emit("Y", y); emit("W", w)

# lrm stats vector (linear predictor from the fit, 2 d.f.).
f <- lrm(y ~ x1 + x2)
lp <- f$linear.predictors
emit("LP", lp)
s <- f$stats
for (k in c("Model L.R.", "P", "C", "Dxy", "Gamma", "Tau-a", "R2", "Brier", "g", "gr", "gp"))
  scalar(paste0("LRM_", toupper(gsub("[^A-Za-z0-9]", "_", k))), s[k])
r2m <- s[grep("^R2\\(", names(s))]
emit("LRM_R2M", r2m)
# Weighted lrm.
fw <- lrm(y ~ x1 + x2, weights = w)
emit("LPW", fw$linear.predictors)
sw <- fw$stats
for (k in c("Model L.R.", "C", "Dxy", "Gamma", "Tau-a", "R2", "Brier"))
  scalar(paste0("LRMW_", toupper(gsub("[^A-Za-z0-9]", "_", k))), sw[k])
emit("LRMW_R2M", sw[grep("^R2\\(", names(sw))])

# somers2, weighted and not.
p <- plogis(lp)
emit("SOMERS2", somers2(p, y)[1:2])
emit("SOMERS2_W", somers2(p, y, weights = w)[1:2])

# concordancefit counts on a continuous y with ties in x and y, weighted.
yc <- round(x1 + rnorm(n), 1); xc <- round(x2 + x1, 1)
emit("YC", yc); emit("XC", xc)
a <- concordancefit(yc, xc, weights = w)
emit("CONC_W", c(a$count[c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")], a$concordance))
a <- concordancefit(yc, xc)
emit("CONC", c(a$count[c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")], a$concordance))

# rcorr.cens: right-censored with ties, and uncensored.
tt <- round(rexp(n, exp(0.5 * x1)), 1); ev <- rbinom(n, 1, 0.7)
px <- round(-x1 + rnorm(n, 0, 0.5), 1)
emit("TT", tt); emit("EV", ev); emit("PX", px)
emit("RCORR", rcorr.cens(px, Surv(tt, ev)))
emit("RCORR_UNC", rcorr.cens(px, tt))

# GiniMd and R2Measures.
scalar("GINI_LP", GiniMd(lp))
emit("R2M_ESS", R2Measures(25.3, 4, 200, 60))
emit("R2M_FREQ", R2Measures(25.3, 4, 200, c(60, 140)))
emit("R2M_NONE", R2Measures(25.3, 4, 200))

# val.prob on predictions from a deliberately miscalibrated model.
pv <- plogis(0.3 + 1.6 * lp)
emit("PV", pv)
v <- val.prob(pv, y, pl = FALSE)
emit("VALPROB", v)
cat(sprintf("const VALPROB_NAMES: [&str; %d] = [%s];\n", length(v),
            paste(sprintf("\"%s\"", names(v)), collapse = ", ")))

# Effective sample size behind the R2(ess) columns of lrm: R2Measures(...,
# sumwty) uses n (1 - sum(P_k^3)) of the (weighted) class counts.
ess <- function(cnt) sum(cnt) * (1 - sum((cnt / sum(cnt))^3))
scalar("LRM_ESS", ess(table(y)))
scalar("LRMW_ESS", ess(tapply(w, y, sum)))

# val.prob with probabilities of exactly 0 / 1. L01 is summed over all rows
# with na.rm = TRUE: p = 0 with y = 1 gives L01 = Inf (U = Inf, Q = -Inf);
# p = 1 with y = 1 and p = 0 / 1 with y = 0 give NaN terms that are dropped.
ones <- which(y == 1); zeros <- which(y == 0)
emit("VALPROB_EDGE_IDX", c(ones[1], ones[2], zeros[1], zeros[2]) - 1)
pe <- pv; pe[ones[1]] <- 1; pe[ones[2]] <- 0
emit("VALPROB_EDGE_INF", suppressWarnings(val.prob(pe, y, pl = FALSE)))
pe <- pv; pe[ones[1]] <- 1; pe[zeros[1]] <- 0; pe[zeros[2]] <- 1
emit("VALPROB_EDGE_FIN", suppressWarnings(val.prob(pe, y, pl = FALSE)))
# Constant p = 1: every L01 term is NaN, so L01 = 0.
emit("VALPROB_CONST_ONE", val.prob(rep(1, n), y, pl = FALSE))
