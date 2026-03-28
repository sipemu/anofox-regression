#!/usr/bin/env Rscript
# Generate validation data for LogisticRegression implementation
# Uses R's glm() with family=binomial(link="logit") as the oracle

set.seed(42)

cat("// =============================================================================\n")
cat("// Logistic Regression Validation Data Generated from R glm()\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// R version:", R.version.string, "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Simple binary classification
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Simple binary classification\n")
cat("// R: glm(y ~ x, family=binomial(link='logit'))\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 40
x1 <- seq(-3, 3, length.out = n)
prob1 <- 1 / (1 + exp(-(0.5 + 1.5 * x1)))
y1 <- rbinom(n, 1, prob1)

fit1 <- glm(y1 ~ x1, family = binomial(link = "logit"))
s1 <- summary(fit1)

cat(sprintf("const N_SIMPLE: usize = %d;\n", n))
cat(sprintf("const X_SIMPLE: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x1), collapse = ", ")))
cat(sprintf("const Y_SIMPLE: [f64; %d] = [%s];\n", n, paste(sprintf("%.1f", y1), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_SIMPLE: f64 = %.10f;\n", coef(fit1)[1]))
cat(sprintf("const EXPECTED_COEF_SIMPLE: f64 = %.10f;\n", coef(fit1)[2]))

# Predicted probabilities
probs1 <- predict(fit1, type = "response")
cat(sprintf("const EXPECTED_PROBS_SIMPLE: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", probs1), collapse = ", ")))

# Predicted classes (threshold 0.5)
classes1 <- as.numeric(probs1 >= 0.5)
cat(sprintf("const EXPECTED_CLASSES_SIMPLE: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.1f", classes1), collapse = ", ")))

# Accuracy
accuracy1 <- mean(classes1 == y1)
cat(sprintf("const EXPECTED_ACCURACY_SIMPLE: f64 = %.10f;\n", accuracy1))

# Log-odds (linear predictor)
logodds1 <- predict(fit1, type = "link")
cat(sprintf("const EXPECTED_LOGODDS_SIMPLE: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", logodds1), collapse = ", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Multiple predictors
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Multiple predictors\n")
cat("// R: glm(y ~ x1 + x2, family=binomial(link='logit'))\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
x2a <- rnorm(n, mean = 0, sd = 1)
x2b <- rnorm(n, mean = 0, sd = 1)
logit2 <- -0.3 + 1.0 * x2a + 0.8 * x2b
prob2 <- 1 / (1 + exp(-logit2))
y2 <- rbinom(n, 1, prob2)

fit2 <- glm(y2 ~ x2a + x2b, family = binomial(link = "logit"))

cat(sprintf("const N_MULTI: usize = %d;\n", n))
cat(sprintf("const X_MULTI_1: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x2a), collapse = ", ")))
cat(sprintf("const X_MULTI_2: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x2b), collapse = ", ")))
cat(sprintf("const Y_MULTI: [f64; %d] = [%s];\n", n, paste(sprintf("%.1f", y2), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_MULTI: f64 = %.10f;\n", coef(fit2)[1]))
cat(sprintf("const EXPECTED_COEF_MULTI_1: f64 = %.10f;\n", coef(fit2)[2]))
cat(sprintf("const EXPECTED_COEF_MULTI_2: f64 = %.10f;\n", coef(fit2)[3]))

probs2 <- predict(fit2, type = "response")
classes2 <- as.numeric(probs2 >= 0.5)
accuracy2 <- mean(classes2 == y2)
cat(sprintf("const EXPECTED_ACCURACY_MULTI: f64 = %.10f;\n", accuracy2))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 3: Well-separated data (high accuracy)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: Well-separated data\n")
cat("// R: glm(y ~ x, family=binomial(link='logit'))\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x3 <- c(rnorm(15, mean = -2, sd = 0.5), rnorm(15, mean = 2, sd = 0.5))
y3 <- c(rep(0, 15), rep(1, 15))

fit3 <- glm(y3 ~ x3, family = binomial(link = "logit"))

cat(sprintf("const N_SEP: usize = %d;\n", n))
cat(sprintf("const X_SEP: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x3), collapse = ", ")))
cat(sprintf("const Y_SEP: [f64; %d] = [%s];\n", n, paste(sprintf("%.1f", y3), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_SEP: f64 = %.10f;\n", coef(fit3)[1]))
cat(sprintf("const EXPECTED_COEF_SEP: f64 = %.10f;\n", coef(fit3)[2]))

probs3 <- predict(fit3, type = "response")
classes3 <- as.numeric(probs3 >= 0.5)
accuracy3 <- mean(classes3 == y3)
cat(sprintf("const EXPECTED_ACCURACY_SEP: f64 = %.10f;\n", accuracy3))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 4: L2 regularized logistic regression (using penalized ML)
# Note: R's glm doesn't support regularization natively, but we can compare
# the unregularized case and check that L2 shrinks coefficients
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Coefficient comparison for regularization check\n")
cat("// (unregularized R baseline; Rust test verifies L2 shrinks toward zero)\n")
cat("// -----------------------------------------------------------------------------\n")

# Reuse fit1 (simple case) coefficients as unregularized baseline
cat(sprintf("const UNREG_INTERCEPT: f64 = %.10f;\n", coef(fit1)[1]))
cat(sprintf("const UNREG_COEF: f64 = %.10f;\n", coef(fit1)[2]))
cat("\n")
