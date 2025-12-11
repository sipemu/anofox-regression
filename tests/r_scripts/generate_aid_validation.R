#!/usr/bin/env Rscript
# AID (Automatic Identification of Demand) Validation Data Generation
# This script generates test data for validating the Rust AID implementation
# against R's greybox::aid() function

library(greybox)

# Set seed for reproducibility
set.seed(42)

# Helper function to print AID results in Rust-compatible format
print_aid_results <- function(name, result, y) {
  name_upper <- toupper(gsub(" ", "_", gsub("-", "_", name)))

  cat(sprintf("\n// Test Case: %s\n", name))
  cat(sprintf("// Distribution: %s\n", result$distribution))
  cat(sprintf("// Zero proportion: %.6f\n", sum(y == 0) / length(y)))

  # Print y data
  cat(sprintf("const Y_%s: [f64; %d] = [%s];\n", name_upper, length(y),
              paste(sprintf("%.6f", y), collapse=", ")))

  # Demand classification
  cat(sprintf("const EXPECTED_DISTRIBUTION_%s: &str = \"%s\";\n", name_upper, result$distribution))

  # New product flag
  cat(sprintf("const EXPECTED_NEW_PRODUCT_%s: bool = %s;\n", name_upper,
              tolower(as.character(result$new))))

  # Obsolete product flag
  cat(sprintf("const EXPECTED_OBSOLETE_%s: bool = %s;\n", name_upper,
              tolower(as.character(result$obsolete))))

  # IC values for each distribution tested
  cat(sprintf("// IC values:\n"))
  if(!is.null(result$ICs) && length(result$ICs) > 0) {
    for(i in 1:length(result$ICs)) {
      ic_val <- result$ICs[i]
      if(is.finite(ic_val)) {
        cat(sprintf("//   %s: %.6f\n", names(result$ICs)[i], ic_val))
      } else {
        cat(sprintf("//   %s: %s\n", names(result$ICs)[i], as.character(ic_val)))
      }
    }
    # Best IC value (filter out non-finite values)
    finite_ics <- result$ICs[is.finite(result$ICs)]
    if(length(finite_ics) > 0) {
      cat(sprintf("const EXPECTED_BEST_IC_%s: f64 = %.10f;\n", name_upper, min(finite_ics)))
    } else {
      cat(sprintf("// EXPECTED_BEST_IC_%s: No finite IC values\n", name_upper))
    }
  } else {
    cat(sprintf("// No IC values available\n"))
  }

  # Stockouts (if any)
  stockouts <- result$stockouts
  if(!is.null(stockouts) && is.numeric(stockouts) && length(stockouts) > 0) {
    # Convert to 0-indexed for Rust
    cat(sprintf("const EXPECTED_STOCKOUTS_%s: [usize; %d] = [%s];\n",
                name_upper, length(stockouts),
                paste(stockouts - 1, collapse=", ")))
  } else {
    cat(sprintf("const EXPECTED_STOCKOUTS_%s: [usize; 0] = [];\n", name_upper))
  }

  cat("\n")
}

cat("// =============================================================================\n")
cat("// AID Validation Data Generated from R greybox package\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// greybox version:", as.character(packageVersion("greybox")), "\n")
cat("// =============================================================================\n\n")

# =============================================================================
# TEST CASE 1: Regular Count Demand (Poisson-like)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 1: Regular Count Demand (Poisson-like)\n")
cat("// =============================================================================\n")

y_regular_count <- rpois(100, lambda = 10)

aid_regular_count <- aid(y_regular_count, ic = "AICc")
print_aid_results("Regular Count", aid_regular_count, y_regular_count)

# =============================================================================
# TEST CASE 2: Regular Fractional Demand (Normal-like)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 2: Regular Fractional Demand (Normal-like)\n")
cat("// =============================================================================\n")

y_regular_frac <- 10 + rnorm(100, sd = 2)
# Ensure no negative values
y_regular_frac <- pmax(y_regular_frac, 0.1)

aid_regular_frac <- aid(y_regular_frac, ic = "AICc")
print_aid_results("Regular Fractional", aid_regular_frac, y_regular_frac)

# =============================================================================
# TEST CASE 3: Intermittent Count Demand (many zeros)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 3: Intermittent Count Demand (many zeros)\n")
cat("// =============================================================================\n")

y_intermittent_count <- rpois(100, lambda = 0.5)

aid_intermittent_count <- aid(y_intermittent_count, ic = "AICc")
print_aid_results("Intermittent Count", aid_intermittent_count, y_intermittent_count)

# =============================================================================
# TEST CASE 4: Intermittent Fractional Demand
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 4: Intermittent Fractional Demand\n")
cat("// =============================================================================\n")

y_intermittent_frac <- ifelse(runif(100) < 0.4, 0, rgamma(100, shape = 2, rate = 0.5))

aid_intermittent_frac <- aid(y_intermittent_frac, ic = "AICc")
print_aid_results("Intermittent Fractional", aid_intermittent_frac, y_intermittent_frac)

# =============================================================================
# TEST CASE 5: New Product (leading zeros)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 5: New Product (leading zeros)\n")
cat("// =============================================================================\n")

y_new_product <- c(rep(0, 30), rpois(70, lambda = 5))

aid_new_product <- aid(y_new_product, ic = "AICc")
print_aid_results("New Product", aid_new_product, y_new_product)

# =============================================================================
# TEST CASE 6: Obsolete Product (trailing zeros)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 6: Obsolete Product (trailing zeros)\n")
cat("// =============================================================================\n")

y_obsolete <- c(rpois(70, lambda = 8), rep(0, 30))

aid_obsolete <- aid(y_obsolete, ic = "AICc")
print_aid_results("Obsolete Product", aid_obsolete, y_obsolete)

# =============================================================================
# TEST CASE 7: Stockouts (unexpected zeros in middle)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 7: Stockouts (unexpected zeros in middle)\n")
cat("// =============================================================================\n")

y_stockouts <- rpois(100, lambda = 15)
# Introduce stockouts at specific positions
y_stockouts[c(25, 50, 75)] <- 0

aid_stockouts <- aid(y_stockouts, ic = "AICc", level = 0.95)
print_aid_results("Stockouts", aid_stockouts, y_stockouts)

# =============================================================================
# TEST CASE 8: Overdispersed Count (Negative Binomial territory)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 8: Overdispersed Count (Negative Binomial territory)\n")
cat("// =============================================================================\n")

y_overdispersed <- rnbinom(100, size = 2, mu = 10)

aid_overdispersed <- aid(y_overdispersed, ic = "AICc")
print_aid_results("Overdispersed", aid_overdispersed, y_overdispersed)

# =============================================================================
# TEST CASE 9: Skewed Positive (Gamma/LogNormal territory)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 9: Skewed Positive (Gamma/LogNormal territory)\n")
cat("// =============================================================================\n")

y_skewed <- rgamma(100, shape = 2, rate = 0.3)

aid_skewed <- aid(y_skewed, ic = "AICc")
print_aid_results("Skewed Positive", aid_skewed, y_skewed)

# =============================================================================
# TEST CASE 10: IC Comparison (AIC vs BIC vs AICc)
# =============================================================================

cat("// =============================================================================\n")
cat("// TEST CASE 10: IC Comparison (AIC vs BIC vs AICc)\n")
cat("// =============================================================================\n")

y_ic_test <- rpois(100, lambda = 7)

aid_aic <- aid(y_ic_test, ic = "AIC")
aid_bic <- aid(y_ic_test, ic = "BIC")
aid_aicc <- aid(y_ic_test, ic = "AICc")

cat("// AIC Selection:\n")
print_aid_results("IC Test AIC", aid_aic, y_ic_test)

cat("// BIC Selection:\n")
print_aid_results("IC Test BIC", aid_bic, y_ic_test)

cat("// AICc Selection:\n")
print_aid_results("IC Test AICc", aid_aicc, y_ic_test)

# =============================================================================
# SUMMARY
# =============================================================================

cat("// =============================================================================\n")
cat("// Summary of AID Classifications\n")
cat("// =============================================================================\n")
cat(sprintf("// Regular Count:        distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_regular_count$distribution, aid_regular_count$new, aid_regular_count$obsolete,
            sum(y_regular_count == 0) / length(y_regular_count)))
cat(sprintf("// Regular Fractional:   distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_regular_frac$distribution, aid_regular_frac$new, aid_regular_frac$obsolete,
            sum(y_regular_frac == 0) / length(y_regular_frac)))
cat(sprintf("// Intermittent Count:   distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_intermittent_count$distribution, aid_intermittent_count$new, aid_intermittent_count$obsolete,
            sum(y_intermittent_count == 0) / length(y_intermittent_count)))
cat(sprintf("// Intermittent Frac:    distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_intermittent_frac$distribution, aid_intermittent_frac$new, aid_intermittent_frac$obsolete,
            sum(y_intermittent_frac == 0) / length(y_intermittent_frac)))
cat(sprintf("// New Product:          distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_new_product$distribution, aid_new_product$new, aid_new_product$obsolete,
            sum(y_new_product == 0) / length(y_new_product)))
cat(sprintf("// Obsolete:             distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_obsolete$distribution, aid_obsolete$new, aid_obsolete$obsolete,
            sum(y_obsolete == 0) / length(y_obsolete)))
cat(sprintf("// Stockouts:            distribution=%s, n_stockouts=%d\n",
            aid_stockouts$distribution, length(aid_stockouts$stockouts)))
cat(sprintf("// Overdispersed:        distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_overdispersed$distribution, aid_overdispersed$new, aid_overdispersed$obsolete,
            sum(y_overdispersed == 0) / length(y_overdispersed)))
cat(sprintf("// Skewed Positive:      distribution=%s, new=%s, obsolete=%s, zero_prop=%.3f\n",
            aid_skewed$distribution, aid_skewed$new, aid_skewed$obsolete,
            sum(y_skewed == 0) / length(y_skewed)))
cat(sprintf("// IC Test (AIC):        distribution=%s\n", aid_aic$distribution))
cat(sprintf("// IC Test (BIC):        distribution=%s\n", aid_bic$distribution))
cat(sprintf("// IC Test (AICc):       distribution=%s\n", aid_aicc$distribution))
