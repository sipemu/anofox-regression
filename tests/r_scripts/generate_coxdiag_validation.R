#!/usr/bin/env Rscript
# Validation data for the Cox / AFT diagnostics (#115) against
# residuals.coxph, cox.zph and survreg.
#
#   Rscript tests/r_scripts/generate_coxdiag_validation.R > tests/fixtures/coxdiag_validation.rs

suppressMessages(library(survival))

emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", ifelse(is.infinite(v), ifelse(v > 0, "f64::INFINITY", "f64::NEG_INFINITY"), sprintf("%.17e", v)))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

d <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
d$status <- d$status - 1
emit("TIME", d$time); emit("STATUS", d$status)
emit("AGE", d$age); emit("SEX", d$sex); emit("ECOG", d$ph.ecog); emit("WTLOSS", d$wt.loss)
set.seed(20261028)
w <- round(runif(nrow(d), 0.5, 2), 2)
emit("W", w)
ctl <- coxph.control(eps = 1e-12, iter.max = 50)

res <- function(id, f) {
  emit(sprintf("%s_MART", id), residuals(f, "martingale"))
  emit(sprintf("%s_DEV", id), residuals(f, "deviance"))
  emit(sprintf("%s_SCH", id), t(residuals(f, "schoenfeld")))
  emit(sprintf("%s_SCSCH", id), t(residuals(f, "scaledsch")))
  emit(sprintf("%s_DFBETA", id), t(residuals(f, "dfbeta")))
  emit(sprintf("%s_DFBETAS", id), t(residuals(f, "dfbetas")))
}
zph <- function(id, f, tr) {
  z <- cox.zph(f, transform = tr)
  emit(sprintf("%s_ZPH_%s", id, toupper(tr)), t(z$table))
}
fe <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, control = ctl)
res("EFRON", fe)
for (tr in c("km", "rank", "identity", "log")) zph("EFRON", fe, tr)
z <- cox.zph(fe)
emit("EFRON_ZPH_X", z$x); emit("EFRON_ZPH_Y", t(z$y))
fb <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, ties = "breslow", control = ctl)
res("BRESLOW", fb)
zph("BRESLOW", fb, "km")
fw <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, weights = w, control = ctl, robust = FALSE)
res("WEIGHTED", fw)
zph("WEIGHTED", fw, "km")
fs <- coxph(Surv(time, status) ~ age + ph.ecog + wt.loss + strata(sex), data = d, control = ctl)
res("STRATA", fs)
zph("STRATA", fs, "km")

h <- heart
emit("H_START", h$start); emit("H_STOP", h$stop); emit("H_EVENT", h$event)
emit("H_AGE", h$age); emit("H_TRANSPLANT", as.numeric(h$transplant) - 1); emit("H_SURGERY", h$surgery)
fh <- coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = h, control = ctl)
res("HEART", fh)
zph("HEART", fh, "km")
zph("HEART", fh, "identity")

# AFT: standardized residuals (log t - lp) / sigma and Cox-Snell -log S(t | x).
fa <- survreg(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, dist = "weibull",
              control = survreg.control(rel.tolerance = 1e-12))
lp <- predict(fa, type = "lp")
zstd <- (log(d$time) - lp) / fa$scale
emit("AFT_STD", zstd)
emit("AFT_COXSNELL", exp(zstd))   # Weibull: H = exp(z)
