options(digits=15)
i <- 0:59; x1 <- i/10; x2 <- sin(i); y <- 2 + 1.5*x1 - 0.8*x2 + 0.3*cos(i*7)
w <- 1 + (i %% 5)/2
x3 <- cos(i/3)
show <- function(tag, f) { s <- summary(f); cat(tag, "\n coef:", paste(format(coef(f), digits=15), collapse=", "),
  "\n se:", paste(format(s$coefficients[,2], digits=15), collapse=", "),
  "\n sigma:", format(s$sigma, digits=15), " r2:", format(s$r.squared, digits=15), " adj:", format(s$adj.r.squared,digits=15), " rank:", f$rank, "\n") }
show("dup", lm(y ~ x1 + I(x1)))
x1b <- x1; show("dup2", lm(y ~ x1 + x1b))
x1b <- 2*x1; show("scaled", lm(y ~ x1 + x1b + x2))
x3c <- x1 + x2; show("lincomb", lm(y ~ x2 + x1 + x3c))
x4 <- 0.5*x1 - 2*x2 + 3*x3; show("lincomb3", lm(y ~ x1 + x4 + x2 + x3))
e <- sin(13*i); e <- e - fitted(lm(e ~ x1 + x2)); e <- e / sqrt(sum(e^2))
xn <- x1 + x2  # base
nb <- sqrt(sum(xn^2))
xbelow <- xn + 0.5e-7*nb*e; show("below", lm(y ~ x1 + x2 + xbelow))
xabove <- xn + 2e-7*nb*e; show("above", lm(y ~ x1 + x2 + xabove))
x1b <- 2*x1; show("wls_scaled", lm(y ~ x1 + x1b + x2, weights=w))
show("wls_dup_noint", lm(y ~ 0 + x1 + x2 + I(3*x2), weights=w))
show("noint_lincomb", lm(y ~ 0 + x2 + x1 + x3c))
s <- summary(lm(y ~ x1 + x2)); cat("lars", s$r.squared, s$adj.r.squared, s$sigma, "\n")
