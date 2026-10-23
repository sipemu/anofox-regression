# Reference values for tests/r_validation_anova_table.rs:
# anova(lm(...)) (type I, sequential) and car::Anova(fit, type = 2)
# (type II, partial). Run with R >= 4.0 and car >= 3.1.
options(digits = 17)
suppressPackageStartupMessages(library(car))

x1 <- c(1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9)
x2 <- c(0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5)
x3 <- c(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0)
y  <- c(2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4)
x4 <- 2 * x1 - x2  # exactly collinear with x1 and x2

show <- function(label, tab) {
  cat(sprintf("== %s\n", label))
  ss <- if ("Sum Sq" %in% names(tab)) tab$`Sum Sq` else tab$`Sum of Sq`
  for (i in seq_len(nrow(tab))) {
    cat(sprintf("%-12s df=%d ss=%.17e F=%.17e p=%.17e\n", rownames(tab)[i],
                as.integer(tab$Df[i]), ss[i], tab$`F value`[i], tab$`Pr(>F)`[i]))
  }
}

f <- lm(y ~ x1 + x2 + x3)
show("type I: y ~ x1 + x2 + x3", anova(f))
show("type II: y ~ x1 + x2 + x3", Anova(f, type = 2))

f <- lm(y ~ x3 + x1 + x2)
show("type I: y ~ x3 + x1 + x2", anova(f))

f <- lm(y ~ 0 + x1 + x2 + x3)
show("type I, no intercept: y ~ 0 + x1 + x2 + x3", anova(f))
show("type II, no intercept: y ~ 0 + x1 + x2 + x3", Anova(f, type = 2))

f <- lm(y ~ x1 + x2 + x4 + x3)
show("type I, aliased x4: y ~ x1 + x2 + x4 + x3", anova(f))

# A two-column term: factor(cyl) is coded as the dummies cyl == 6, cyl == 8.
f <- lm(mpg ~ wt + factor(cyl) + hp, data = mtcars)
show("mtcars type I: mpg ~ wt + factor(cyl) + hp", anova(f))
show("mtcars type II: mpg ~ wt + factor(cyl) + hp", Anova(f, type = 2))
