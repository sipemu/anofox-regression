# Reference values for tests/r_validation_210_part15.rs (#210): the default
# (optimised) lowess span of greybox::lmDynamic. greybox's span criterion
# (fFinder) is written out from its source and evaluated on a grid, on the
# pointwise ICs of the candidate lm fits (pAIC / pAICc, model order ~1, ~x1,
# ~x2, ~x1 + x2, as lmDynamic(bruteforce = TRUE)).
suppressMessages(library(greybox))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
grid <- c(0.2, 0.35, 0.5, 0.65, 0.8, 1)
vec("GRID", grid)
case <- function(tag, d, ic) {
  pfun <- switch(ic, AIC = pAIC, AICc = pAICc)
  m <- list(lm(y ~ 1, d), lm(y ~ x1, d), lm(y ~ x2, d), lm(y ~ x1 + x2, d))
  pics <- sapply(m, pfun)
  w <- exp(-0.5 * (pics - apply(pics, 1, min))); w <- w / rowSums(w)
  wl <- log(w / (1 - w)); icm <- colMeans(pics)
  fFinder <- function(f) {
    s <- wl; s[] <- sapply(apply(wl, 2, lowess, f = f), "[[", "y")
    s[] <- exp(s) / (1 + exp(s)); s[] <- s / apply(s, 1, sum)
    sum(apply(s, 2, mean) * icm)
  }
  fit <- lmDynamic(d, ic = ic, bruteforce = TRUE)
  vec(paste0(tag, "_X1"), d$x1); vec(paste0(tag, "_X2"), d$x2); vec(paste0(tag, "_Y"), d$y)
  vec(paste0(tag, "_CRITERION"), sapply(grid, fFinder))
  vec(paste0(tag, "_GREYBOX_F"), c(fit$f, fFinder(fit$f)))
}
set.seed(2115); n <- 80
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1 + ifelse(1:n > 40, 1.2, 0.3) * x1 + 0.2 * x2 + rnorm(n, sd = 1.2)
case("STEP_AICC", data.frame(y, x1, x2), "AICc")
y <- 0.5 + 0.6 * sin(1:n / 10) * x1 + 0.3 * x2 + rnorm(n, sd = 1.5)
case("WAVE_AIC", data.frame(y, x1, x2), "AIC")
