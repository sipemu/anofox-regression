# Reference values for tests/r_validation_210_part17.rs (#210): lmDynamic
# with a binary response (distribution = "plogis" / "pnorm"). Inside
# lmDynamic greybox fits the candidates with alm(fast = TRUE), which stops
# short of the MLE (logLik -72.5502 vs glm's -72.5463 for y ~ x1 here), so
# the reference applies lmDynamic's combination, written out from its source
# (pAICc = 2k - 2 n l_t + 2k(k + 1)/(n - k - 1), AICc weights, lowess(f) of
# their logits, renormalised), to the converged glm(binomial) fits.
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2117); n <- 120
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rbinom(n, 1, plogis(-0.2 + ifelse(1:n > 60, 1.5, 0.3) * x1 + 0.4 * x2))
vec("X1", x1); vec("X2", x2); vec("Y", y)
forms <- list(y ~ 1, y ~ x1, y ~ x2, y ~ x1 + x2)
for (link in c("logit", "probit")) {
  fits <- lapply(forms, glm, family = binomial(link), control = glm.control(epsilon = 1e-14, maxit = 100))
  pic <- sapply(fits, function(g) {
    k <- length(coef(g)); p <- fitted(g)
    ll <- dbinom(y, 1, p, log = TRUE)
    2 * k - 2 * n * ll + 2 * k * (k + 1) / (n - k - 1)
  })
  w <- exp(-0.5 * (pic - apply(pic, 1, min))); w <- w / rowSums(w)
  wl <- log(w / (1 - w))
  s <- sapply(seq_len(ncol(wl)), function(j) lowess(wl[, j], f = 0.5)$y)
  s <- exp(s) / (1 + exp(s)); s <- s / rowSums(s)
  par <- t(sapply(fits, function(g) { b <- c(0, 0, 0); names(b) <- c("(Intercept)", "x1", "x2"); b[names(coef(g))] <- coef(g); b }))
  tag <- if (link == "logit") "PLOGIS" else "PNORM"
  vec(paste0(tag, "_WEIGHTS"), t(s)); vec(paste0(tag, "_DYNAMIC"), t(s %*% par))
}
# ALM with non-canonical links: for Poisson and Binomial the ALM likelihood
# is the GLM likelihood, so glm() with that link is the reference
# (greybox::alm fixes the link per distribution).
set.seed(2118); m <- 80
xc <- runif(m, 0.5, 3); yc <- rpois(m, (1 + 0.8 * xc)^2); yb <- rbinom(m, 1, pnorm(-1 + 0.7 * xc))
vec("XC", xc); vec("YC", yc); vec("YB", yb)
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
lk <- function(tag, f) vec(tag, c(coef(f), logLik(f)))
lk("POIS_SQRT", glm(yc ~ xc, family = poisson("sqrt"), control = ctl))
lk("POIS_IDENTITY", glm(yc ~ xc, family = poisson("identity"), control = ctl, start = c(1, 3)))
lk("BIN_PROBIT", glm(yb ~ xc, family = binomial("probit"), control = ctl))
lk("BIN_CLOGLOG", glm(yb ~ xc, family = binomial("cloglog"), control = ctl))
