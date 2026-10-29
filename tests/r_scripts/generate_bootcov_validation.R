#!/usr/bin/env Rscript
# Validation data for `validation::bootcov` (#104) against rms::bootcov
# (ordinary and cluster bootstrap); the resamples are regenerated with the
# seeds and sample() calls bootcov makes.
#
#   Rscript tests/r_scripts/generate_bootcov_validation.R > tests/fixtures/bootcov_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

set.seed(20261028)
n <- 120
id <- rep(1:30, each = 4)
u <- rnorm(30)[id]
x1 <- rnorm(n) + 0.3 * u; x2 <- runif(n)
y <- rbinom(n, 1, plogis(-0.2 + 0.9 * x1 - 0.7 * x2 + 0.6 * u))
emit("X1", x1); emit("X2", x2); emit("Y", y); emit_usize("ID", id - 1)
f <- lrm(y ~ x1 + x2, x = TRUE, y = TRUE)
B <- 50
set.seed(1); idx <- replicate(B, sample(1L:n, n, replace = TRUE))
emit_usize("BOOT_IDX", c(idx) - 1)
b1 <- bootcov(f, B = B, seed = 1)
emit("BOOT_COEF", t(b1$boot.Coef)); emit("BOOT_VAR", t(b1$var))
emit("BOOT_Q", apply(b1$boot.Coef, 2, quantile, probs = c(0.025, 0.975)))
cl <- as.character(id); clusters <- unique(cl); Obsno <- split(1:n, cl)
set.seed(2)
rows <- c(); lens <- c()
for (b in 1:40) { j <- sample(clusters, length(clusters), replace = TRUE); r <- unlist(Obsno[j]); rows <- c(rows, r); lens <- c(lens, length(r)) }
emit_usize("CL_ROWS", rows - 1); emit_usize("CL_LENS", lens)
b2 <- bootcov(f, cluster = id, B = 40, seed = 2)
emit("CL_VAR", t(b2$var))
