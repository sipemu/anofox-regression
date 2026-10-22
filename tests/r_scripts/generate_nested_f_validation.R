# Reference values for tests/r_validation_nested_f.rs (issue #90):
# anova(lm(reduced), lm(full)) and summary(...)$adj.r.squared.
options(digits = 17)

x1 <- c(1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9)
x2 <- c(0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5)
x3 <- c(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0)
y  <- c(2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4)
x4 <- 2 * x1 - x2  # exactly collinear with x1 and x2

show <- function(label, reduced, full) {
  a <- anova(reduced, full)
  cat(sprintf("== %s\n", label))
  cat(sprintf("F        = %.17e\n", a$F[2]))
  cat(sprintf("df1      = %d\n", a$Df[2]))
  cat(sprintf("df2      = %d\n", a$Res.Df[2]))
  cat(sprintf("p        = %.17e\n", a$`Pr(>F)`[2]))
  cat(sprintf("rss_r    = %.17e\n", a$RSS[1]))
  cat(sprintf("rss_f    = %.17e\n", a$RSS[2]))
  cat(sprintf("adj_r    = %.17e\n", summary(reduced)$adj.r.squared))
  cat(sprintf("adj_f    = %.17e\n", summary(full)$adj.r.squared))
}

show("x1 vs x1+x2+x3", lm(y ~ x1), lm(y ~ x1 + x2 + x3))
show("df1 = 1: x1+x2 vs x1+x2+x3", lm(y ~ x1 + x2), lm(y ~ x1 + x2 + x3))
cat(sprintf("t^2 of x3 = %.17e\n", summary(lm(y ~ x1 + x2 + x3))$coefficients["x3", "t value"]^2))
show("intercept-only vs x1+x2+x3", lm(y ~ 1), lm(y ~ x1 + x2 + x3))
cat(sprintf("overall F = %.17e\n", summary(lm(y ~ x1 + x2 + x3))$fstatistic[1]))
show("no intercept: x1 vs x1+x2+x3", lm(y ~ 0 + x1), lm(y ~ 0 + x1 + x2 + x3))
show("rank-deficient full: x1 vs x1+x2+x4+x3", lm(y ~ x1), lm(y ~ x1 + x2 + x4 + x3))
