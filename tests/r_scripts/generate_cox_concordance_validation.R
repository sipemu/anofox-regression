# Reference values for tests/issue208_cox_concordance.rs (#208): coxph's
# concordance with case weights, offset, strata and (start, stop] data.
# Uses the data of generate_cox_validation.R (tests/fixtures/cox_validation.rs).
suppressMessages(library(survival))
d <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
d$status <- d$status - 1
set.seed(20261028)
w <- round(runif(nrow(d), 0.5, 2), 2); off <- round(rnorm(nrow(d), 0, 0.2), 3)
ctl <- coxph.control(eps = 1e-12, iter.max = 50)
cc <- function(name, f) cat(sprintf("const %s: f64 = %.17g;\n", name, f$concordance["concordance"]))
cc("CONC_WEIGHTED", coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, weights = w, control = ctl))
cc("CONC_WEIGHTED_OFFSET", coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss + offset(off), data = d, weights = w, control = ctl))
cc("CONC_STRATA", coxph(Surv(time, status) ~ age + ph.ecog + wt.loss + strata(sex), data = d, control = ctl))
cc("CONC_HEART", coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = heart, control = ctl))
cc("CONC_HEART_W", coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = heart, weights = rep(c(1, 2, 0.5), length.out = nrow(heart)), control = ctl))
# Binary covariate: many ties in the linear predictor
cc("CONC_TIES", coxph(Surv(time, status) ~ sex, data = d, control = ctl))
