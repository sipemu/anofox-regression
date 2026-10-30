#!/usr/bin/env Rscript
# Validation data for `reduction::principal_components` and
# `cluster_scores` (#107, RMS §4.7) against stats::prcomp. Component signs
# are arbitrary; the Rust side aligns them before comparing.
#
#   Rscript tests/r_scripts/generate_pca_validation.R > tests/fixtures/pca_validation.rs

emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261110)
n <- 80
z1 <- rnorm(n); z2 <- rnorm(n)
X <- cbind(a = z1 + rnorm(n, sd = .3), b = 2 * z1 + rnorm(n, sd = .5), c = z2 * 10 + rnorm(n),
           d = z2 + z1 + rnorm(n, sd = .4), e = rnorm(n, 5, 2))
emit("X", t(X))
p <- prcomp(X, scale. = TRUE)
emit("SC_SDEV", p$sdev); emit("SC_ROT", p$rotation); emit("SC_SCORES", p$x)
q <- prcomp(X)
emit("UN_SDEV", q$sdev); emit("UN_ROT", q$rotation); emit("UN_SCORES", q$x)
nd <- X[1:3, ] + 0.5
emit("NEW", t(nd)); emit("SC_PRED", predict(p, nd)[, 1:2])
# cluster scores: clusters {a, b}, {c, d}, {e}; first PC of each (scaled)
cs <- cbind(prcomp(X[, c("a", "b")], scale. = TRUE)$x[, 1],
            prcomp(X[, c("c", "d")], scale. = TRUE)$x[, 1],
            prcomp(X[, "e", drop = FALSE], scale. = TRUE)$x[, 1])
emit("CL_SCORES", cs)
