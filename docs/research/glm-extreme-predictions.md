# Research Plan: GLM Extreme Predictions with Sparse Data

**GitHub Issue:** #3
**Branch:** `fix/glm-extreme-predictions-3`
**Epic:** `regression-1xn`

## Problem Statement

GLM models (Poisson, Gamma, Negative Binomial) produce extreme or infinite predictions when:
1. High changepoint count (many binary segment indicators)
2. Sparse/intermittent data (high zero ratio)
3. Log link function exponentiates estimation errors

**Impact:** ~1.7% of time series (337/20,000) produce unusable predictions.

## Research Tasks

### Phase 1: Root Cause Analysis (P1)

#### 1. IRLS Convergence Behavior (`regression-zvb`) ✅ COMPLETE
- [x] Trace IRLS iterations for problematic vs stable series
- [x] Check convergence criteria and iteration counts
- [x] Identify if algorithm diverges or converges to bad solution
- [x] Examine working weights during iteration

#### 2. Coefficient Explosion Investigation (`regression-fc6`) ✅ COMPLETE
- [x] Analyze coefficient magnitudes for extreme prediction cases
- [x] Check for near-singular design matrices (X'WX)
- [x] Examine relationship between segment sparsity and coefficient variance
- [x] Root cause identified: quasi-separation + unbounded IRLS

### Phase 2: Solution Implementation (P1/P2)

#### 3. Validity Checks (`regression-uo6`) - P1
- [ ] Implement `validmu()` checks for each GLM family
- [ ] Implement `valideta()` checks for each link function
- [ ] Add appropriate bounds (e.g., Poisson: μ > 0, Binomial: 0 < μ < 1)
- [ ] Return graceful errors when validity fails

#### 4. Deviance Monitoring (`regression-k3r`) - P1
- [ ] Compute deviance at each IRLS iteration
- [ ] Reject coefficient updates that increase deviance
- [ ] Track deviance history for convergence diagnostics

#### 5. Step Halving Mechanism (`regression-bcm`) - P2 (blocked by k3r)
- [ ] Implement step-halving when deviance increases (like R's glm.fit)
- [ ] Configure maximum halvings per iteration
- [ ] Fall back to previous coefficients when step-halving exhausted

#### 6. Quasi-Separation Detection (`regression-mxm`) - P2
- [ ] Detect quasi-separation patterns in design matrix
- [ ] Warn users when quasi-separation detected
- [ ] Suggest remediation (regularization, feature removal)

#### 7. Penalized IRLS (`regression-euj`) - P2
- [ ] Add optional L2 penalty to IRLS (Ridge GLM)
- [ ] Consider Firth's bias-reduced approach for severe cases
- [ ] Assess impact on prediction accuracy for stable series

#### 8. Condition Number Diagnostics (`regression-84y`) - P2
- [ ] Implement condition number calculation for X'WX
- [ ] Define thresholds for "dangerous" multicollinearity
- [ ] Design early warning system for unstable fits

### Phase 3: Documentation (`regression-d34`) - blocked by implementation
- [ ] Summarize findings with evidence
- [ ] Recommend implementation approach
- [ ] Document trade-offs and limitations
- [ ] Propose API changes if needed

## Key Files to Investigate

```
src/glm/
├── poisson.rs      # Poisson GLM implementation
├── binomial.rs     # Binomial GLM (similar IRLS)
├── negative_binomial.rs
├── tweedie.rs
└── irls.rs         # Core IRLS algorithm (if separate)
```

## Hypothesis

The primary cause is **quasi-separation** or **near-perfect prediction** in sparse segments:
- When a binary changepoint indicator has very few (or zero) non-zero y values
- The coefficient for that segment can grow unboundedly
- Log link exponentiates: exp(large_coef) → extreme prediction

## Potential Solutions (to evaluate)

1. **Regularization**: Add L2 penalty to IRLS (Ridge GLM)
2. **Coefficient bounds**: Clamp coefficients during iteration
3. **Condition number check**: Warn/fail when X'WX is ill-conditioned
4. **Prediction bounds**: Cap predictions at reasonable multiples of training data
5. **Feature selection**: Recommend dropping sparse changepoint indicators
6. **Alternative models**: Suggest using ElasticNet for high-changepoint series

## Success Criteria

- Understand exact mechanism causing extreme predictions
- Identify at least one implementable solution
- Solution should not degrade accuracy for stable series
- Provide diagnostic capability for users to detect risky inputs

---

## Research Findings

### IRLS Convergence Analysis (regression-zvb) ✅ COMPLETE

**Date:** 2026-01-16

#### Key Files Analyzed

| File | Purpose |
|------|---------|
| `src/core/family.rs` | GlmFamily trait, IRLS weight/working response |
| `src/core/poisson.rs` | Poisson family (variance, link functions) |
| `src/solvers/poisson.rs` | Poisson IRLS solver |
| `src/solvers/tweedie.rs` | Tweedie IRLS solver |
| `src/solvers/negative_binomial.rs` | NegBin IRLS solver |

#### IRLS Algorithm Summary

The IRLS iteration follows standard GLM theory:

```
For each iteration:
  1. Compute weights: w = 1 / (V(μ) × (dη/dμ)²)
  2. Compute working response: z = η + (y - μ) × dη/dμ
  3. Solve weighted LS: β = (X'WX)⁻¹ X'Wz  (via QR)
  4. Update: η = Xβ, μ = g⁻¹(η)
  5. Check: max|β_new - β_old| < tolerance
```

**Convergence criteria:** `max_change < 1e-6` (default tolerance)
**Max iterations:** 1000 (default)

#### Critical Issue: No μ Bounds During Iteration

**Finding:** The Poisson and Tweedie solvers do NOT clamp μ during IRLS iteration.

```rust
// In fit_irls() - poisson.rs line 167
mu[i] = self.family.link_inverse(eta_i);
// NO clamping! μ can become arbitrarily small or large
```

**Contrast with ALM solver** which DOES clamp:
```rust
// In alm.rs line 782
mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
```

#### Mathematical Analysis: Poisson with Log Link

For Poisson GLM with log link (canonical):
- **Variance:** V(μ) = μ
- **Link:** η = log(μ), so dη/dμ = 1/μ
- **IRLS Weight:** w = 1/(μ × (1/μ)²) = **μ**
- **Working Response:** z = log(μ) + (y - μ)/μ

**Problem Scenario (sparse segment):**

| Condition | Weight (w) | Working Response (z) |
|-----------|------------|---------------------|
| μ → 0, y = 0 | → 0 | log(μ) - 1 → **-∞** |
| μ → 0, y > 0 | → 0 | log(μ) + y/μ → **+∞** |
| μ → large | → large | stable |

When μ is small (sparse data with few positive observations):
1. **Small weights** reduce influence but don't prevent issues
2. **Extreme working response** drives coefficient updates
3. **No bounds on β** allow unbounded growth
4. **exp(large β)** produces extreme predictions

#### Safeguards Present (Insufficient)

| Safeguard | Location | Effect |
|-----------|----------|--------|
| Weight floor | `family.rs:40-42` | Returns 1e-10 when V or dη/dμ near zero |
| μ initialization | `poisson.rs:129` | Clamps initial μ ≥ 1e-3 |
| Rank tolerance | `poisson.rs:242-246` | Sets β=0 for near-zero R diagonal |

**Missing safeguards:**
- ❌ No μ clamping during iteration
- ❌ No coefficient magnitude limits
- ❌ No step-halving when diverging
- ❌ No condition number monitoring

#### Root Cause Confirmed

The issue is **quasi-separation** combined with **unbounded IRLS**:

1. Binary changepoint indicator has segment with mostly zeros
2. Initial μ for that segment is small (driven toward mean of sparse data)
3. IRLS computes extreme working response for those observations
4. Coefficient for that indicator grows large (positive or negative)
5. No bounds prevent growth → coefficient explodes
6. exp(large_coef) → extreme μ → even more extreme next iteration
7. May "converge" to extreme solution or hit max iterations

#### Recommendations (Updated after R/Python research)

**Primary approach (following R's glm.fit):**
1. **Validity checks** - Add `validmu()` and `valideta()` checks to each family
2. **Deviance monitoring** - Compute deviance at each iteration, reject diverging steps
3. **Step-halving** - When deviance increases, halve the step size (up to N times)

**Secondary safeguards:**
4. **Quasi-separation detection** - Warn when design matrix suggests separation
5. **Penalized IRLS option** - Add Ridge GLM for high-collinearity cases

**Diagnostics:**
6. Return condition number of X'WX
7. Flag coefficients with large standard errors

**Note:** Simple μ clamping is NOT the standard approach. R's glm.fit uses step-halving
with deviance monitoring, which is more mathematically principled.

---

### Industry Research: How Other Packages Handle GLM Stability

**Date:** 2026-01-16

#### R's glm.fit (stats package)

R's `glm.fit()` uses a sophisticated approach:

1. **Validity Functions:**
   - `validmu()` - Family-specific checks (e.g., Poisson: `all(mu > 0)`)
   - `valideta()` - Link-specific checks (e.g., log: `all(is.finite(eta))`)

2. **Step Halving (key mechanism):**
   ```r
   # From glm.fit source
   for (i in seq_len(maxit)) {
       # ... compute new coefficients ...
       dev <- sum(dev.resids(y, mu, weights))
       if (control$trace) cat("Deviance =", dev, "\n")

       # Step halving when deviance increases
       if (is.finite(dev)) {
           for (j in seq_len(control$nshalf)) {
               if (dev > dev.old + 1e-7 * dev) {
                   # Halve the step
                   coefnew <- (coef + coefold)/2
                   # ... recompute mu and dev ...
               } else break
           }
       }
   }
   ```

3. **Convergence Based on Deviance:**
   - Primary criterion: `abs(dev - devold)/(0.1 + abs(dev)) < epsilon`
   - Not just coefficient change

#### Python statsmodels

- Uses similar IRLS with deviance-based convergence
- Less sophisticated step-halving than R
- Relies more on regularization options

#### scikit-learn

- Uses regularization by default (even with small λ)
- L-BFGS optimizer rather than pure IRLS
- Different design philosophy

#### Specialized Packages

- **brglm** (R): Firth's bias-reduced method for separation
- **detectseparation** (R): Formal quasi-separation detection
- **safeBinaryRegression** (R): Warns on separation

#### Key Insight

**R's approach (step-halving + validity checks) is the gold standard** for GLM stability.
It maintains the statistical properties of MLE while preventing numerical divergence.
Simple μ clamping can introduce bias and is not recommended for general use.
