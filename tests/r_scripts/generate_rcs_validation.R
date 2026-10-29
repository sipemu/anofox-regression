#!/usr/bin/env Rscript
# Validation data for `splines` (#99) against Hmisc::rcspline.eval,
# Hmisc::rcspline.restate, rms::rcs, rms::lsp and rms::%ia%.
#
#   Rscript tests/r_scripts/generate_rcs_validation.R > tests/fixtures/rcs_validation.rs

suppressMessages({ library(Hmisc); library(rms) })

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261028)
datasets <- list(
  SMALL = rnorm(40),                                  # n < 100: outer knots moved to x[5], x[n-4]
  LARGE = rnorm(300, 10, 3),
  TIEDLOW = c(rep(0, 40), rexp(160)),                 # many ties at the minimum (fractied)
  TIEDBOTH = c(rep(0, 30), runif(140, 1, 9), rep(10, 30)),
  FEWU = sample(1:6, 100, replace = TRUE),            # few unique values: interior values
  INTS = sample(1:12, 150, replace = TRUE)
)

# Default knots for nk = 3..7 (NaN-padded to 7 entries; the count is emitted).
for (id in names(datasets)) {
  x <- datasets[[id]]
  emit(sprintf("X_%s", id), x)
  for (nk in 3:7) {
    k <- suppressWarnings(tryCatch(rcspline.eval(x, nk = nk, knots.only = TRUE),
                                   error = function(e) numeric(0)))
    emit(sprintf("KNOTS_%s_%d", id, nk), k)
  }
}

# Basis (inclx = TRUE, as rms::rcs) for each normalisation, row-major.
x <- datasets$SMALL
kn <- rcspline.eval(x, nk = 5, knots.only = TRUE)
for (norm in 0:2) {
  b <- rcspline.eval(x, knots = kn, inclx = TRUE, norm = norm)
  emit(sprintf("BASIS_SMALL_5_NORM%d", norm), t(b))
}
# rms::rcs uses the same default knots and basis.
emit("RMS_RCS_LARGE_4", t(unclass(rcs(datasets$LARGE, 4))[, ]))
emit("RMS_RCS_LARGE_4_KNOTS", attr(rcs(datasets$LARGE, 4), "parms"))

# Restated (unrestricted truncated-power) coefficients of a fitted rcs.
y <- sin(x) + 0.1 * x^2
f <- lm(y ~ rcspline.eval(x, knots = kn, inclx = TRUE))
emit("RESTATE_COEF_IN", coef(f))
r <- rcspline.restate(kn, coef(f))
emit("RESTATE_COEF_OUT", as.numeric(r))

# Linear spline.
emit("LSP_KNOTS", c(-1, 0, 1))
emit("LSP_SMALL", t(unclass(lsp(x, c(-1, 0, 1)))[, ]))

# Restricted interaction rcs(x1, 4) %ia% rcs(x2, 3).
x1 <- datasets$SMALL
x2 <- rnorm(40, 2)
emit("IA_X2", x2)
ia <- rcs(x1, 4) %ia% rcs(x2, 3)
cat(sprintf("const IA_NCOL: usize = %d;\n", ncol(ia)))
emit("IA", t(unclass(ia)[, ]))
cat(sprintf("const IA_NONLINEAR: [bool; %d] = [%s];\n", ncol(ia),
            paste(tolower(as.character(attr(ia, "nonlinear"))), collapse = ", ")))
