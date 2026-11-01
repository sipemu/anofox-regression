#!/usr/bin/env Rscript
# Validation data for `effects` (#116) against rms::datadist,
# summary.rms, contrast.rms and Predict.
#
#   Rscript tests/r_scripts/generate_effects_validation.R > tests/fixtures/effects_validation.rs

suppressMessages(library(rms))

emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}

set.seed(20261028)
n <- 300
x1 <- rnorm(n, 50, 10); x2 <- round(runif(n, 0, 10), 1)
g <- sample(c(1, 2, 3), n, replace = TRUE, prob = c(0.3, 0.45, 0.25))
lin <- -2 + 0.04 * x1 + 0.0015 * (x1 - 50)^2 + 0.1 * x2 + 0.005 * x1 * x2 + c(0, 0.5, -0.4)[g]
yc <- lin + rnorm(n)
yb <- rbinom(n, 1, plogis(lin))
emit("X1", x1); emit("X2", x2); emit("G", g); emit("YC", yc); emit("YB", yb)
d <- data.frame(x1, x2, g = factor(g), yc, yb)
dd <- datadist(d); options(datadist = "dd")
emit("LIM_X1", dd$limits$x1); emit("LIM_X2", dd$limits$x2)

for (m in c("ols", "lrm")) {
  f <- if (m == "ols") ols(yc ~ rcs(x1, 4) * x2 + g, data = d) else lrm(yb ~ rcs(x1, 4) * x2 + g, data = d)
  id <- toupper(m)
  emit(sprintf("%s_KNOTS", id), attr(rcs(x1, 4), "parms"))
  emit(sprintf("%s_COEF", id), coef(f))
  emit(sprintf("%s_VCOV", id), t(vcov(f)))
  s <- summary(f)
  # Effect rows only (skip lrm's odds-ratio rows), columns Low..Upper.
  eff <- s[s[, "Type"] == 1, c("Low", "High", "Diff.", "Effect", "S.E.", "Lower 0.95", "Upper 0.95"), drop = FALSE]
  emit(sprintf("%s_SUMMARY", id), t(eff))
  if (m == "lrm") {
    or <- s[s[, "Type"] == 2, c("Effect", "Lower 0.95", "Upper 0.95"), drop = FALSE]
    emit("LRM_OR", t(or))
  }
  k <- contrast(f, list(x1 = 60, x2 = 2), list(x1 = 45, x2 = 2))
  emit(sprintf("%s_CONTRAST", id), c(k$Contrast, k$SE, k$Lower, k$Upper, k$Z %||% NA, k$Pvalue))
  k2 <- contrast(f, list(x1 = 60, x2 = 8), list(x1 = 45, x2 = 8), list(x1 = 60, x2 = 2), list(x1 = 45, x2 = 2))
  emit(sprintf("%s_DOUBLE", id), c(k2$Contrast, k2$SE, k2$Lower, k2$Upper, k2$Pvalue))
  kg <- contrast(f, list(g = "3"), list(g = "2"))
  emit(sprintf("%s_CONTRAST_G", id), c(kg$Contrast, kg$SE, kg$Lower, kg$Upper, kg$Pvalue))
  p <- Predict(f, x1)
  emit(sprintf("%s_PRED_X1", id), p$x1); emit(sprintf("%s_PRED_YHAT", id), p$yhat)
  emit(sprintf("%s_PRED_LOWER", id), p$lower); emit(sprintf("%s_PRED_UPPER", id), p$upper)
}
# #210: a linear spline and a restricted interaction (%ia%)
for (m in c("LSP", "IA")) {
  f <- if (m == "LSP") ols(yc ~ rcs(x1, 4) + lsp(x2, c(3, 6)) + g, data = d) else
    ols(yc ~ rcs(x1, 4) + rcs(x2, 3) + g + rcs(x1, 4) %ia% rcs(x2, 3), data = d)
  emit(sprintf("%s_COEF", m), coef(f)); emit(sprintf("%s_VCOV", m), t(vcov(f)))
  if (m == "IA") emit("IA_KNOTS_X2", attr(rcs(x2, 3), "parms"))
  s <- summary(f)
  emit(sprintf("%s_SUMMARY", m), t(s[s[, "Type"] == 1, c("Low", "High", "Diff.", "Effect", "S.E.", "Lower 0.95", "Upper 0.95"), drop = FALSE]))
  k <- contrast(f, list(x1 = 60, x2 = 7), list(x1 = 45, x2 = 2))
  emit(sprintf("%s_CONTRAST", m), c(k$Contrast, k$SE, k$Lower, k$Upper))
  p <- Predict(f, x2)
  emit(sprintf("%s_PRED_X2", m), p$x2); emit(sprintf("%s_PRED_YHAT", m), p$yhat)
  emit(sprintf("%s_PRED_LOWER", m), p$lower); emit(sprintf("%s_PRED_UPPER", m), p$upper)
}
