#!/usr/bin/env Rscript
# Validation data for `reduction` (#107, RMS §4.7) against Hmisc and stats:
# spearman2 (generalized Spearman rho^2), hoeffd (Hoeffding's D with
# P-values), varclus (similarity matrices + hclust) and stats::hclust for
# every linkage method, including ties.
#
#   Rscript tests/r_scripts/generate_reduction_validation.R > tests/fixtures/reduction_validation.rs

suppressMessages(library(Hmisc))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261102)
n <- 80
z1 <- rnorm(n); z2 <- rnorm(n)
x1 <- z1 + rnorm(n, sd = 0.3)
x2 <- z1^2 + rnorm(n, sd = 0.5)                 # nonmonotone in z1
x3 <- round(z2 + rnorm(n, sd = 0.4), 1)          # ties
x4 <- z2 + 0.5 * z1 + rnorm(n, sd = 0.6)
x5 <- rnorm(n)
x6 <- sample(1:3, n, TRUE)                        # few distinct values
X <- cbind(x1, x2, x3, x4, x5, x6)
y <- x1 + x2 - x4 + rnorm(n)
emit("X", t(X)); emit("Y", y)

# spearman2 of y on each column, p = 1 and p = 2, and on a factor
s1 <- t(sapply(1:6, function(j) spearman2(X[, j], y, p = 1)))
s2 <- t(sapply(1:6, function(j) spearman2(X[, j], y, p = 2)))
emit("SP1", t(s1)); emit("SP2", t(s2))
emit("SPF", spearman2(factor(x6), y))

h <- hoeffd(X)
emit("HOEFFD_D", h$D); emit("HOEFFD_P", ifelse(is.na(h$P), -1, h$P))

for (s in c("spearman", "pearson", "hoeffding")) {
  v <- varclus(X, similarity = s)
  tag <- toupper(s)
  emit(paste0("VC_", tag, "_SIM"), v$sim)
  emit(paste0("VC_", tag, "_MERGE"), t(v$hclust$merge))
  emit(paste0("VC_", tag, "_HEIGHT"), v$hclust$height)
  emit(paste0("VC_", tag, "_ORDER"), v$hclust$order)
}
v <- varclus(X, similarity = "spearman", trans = "abs", method = "average")
emit("VC_ABS_AVG_MERGE", t(v$hclust$merge)); emit("VC_ABS_AVG_HEIGHT", v$hclust$height)

# hclust on a dissimilarity with ties, every method
d <- dist(matrix(c(0, 1, 2, 4, 7, 11, 0, 0, 1, 1, 3, 3, 5, 2, 8, 0), ncol = 2))
emit("HC_DIST", as.matrix(d))
for (m in c("ward.D", "single", "complete", "average", "mcquitty", "median", "centroid", "ward.D2")) {
  hc <- hclust(d, method = m)
  tag <- toupper(gsub(".", "", m, fixed = TRUE))
  emit(paste0("HC_", tag, "_MERGE"), t(hc$merge))
  emit(paste0("HC_", tag, "_HEIGHT"), hc$height)
  emit(paste0("HC_", tag, "_ORDER"), hc$order)
}
