# @sipemu/anofox-regression

WebAssembly bindings for [anofox-regression](https://github.com/sipemu/anofox-regression), a comprehensive statistical regression library.

## Features

### Linear Models
- **OLS Regression** - Ordinary Least Squares with full inference (standard errors, p-values, confidence intervals)
- **WLS Regression** - Weighted Least Squares for heteroscedastic data
- **Ridge Regression** - L2 regularization for handling multicollinearity
- **Elastic Net** - Combined L1/L2 regularization (Lasso + Ridge)
- **BLS Regression** - Bounded/Non-Negative Least Squares (Lawson-Hanson algorithm)
- **PLS Regression** - Partial Least Squares (SIMPLS) for collinear data
- **RLS Regression** - Recursive Least Squares for online learning

### Quantile & Monotonic
- **Quantile Regression** - Estimate conditional quantiles (median, quartiles, etc.)
- **Isotonic Regression** - Monotonic regression using Pool Adjacent Violators Algorithm

### Generalized Linear Models (GLM)
- **Poisson Regression** - For count data (log/identity/sqrt link)
- **Binomial Regression** - Logistic/Probit for binary outcomes
- **Negative Binomial** - For overdispersed count data
- **Tweedie Regression** - Flexible variance (Gamma, Compound Poisson-Gamma, etc.)

### Ordinal Response & Penalized GLM
- **Ordinal** - Cumulative link models (logistic, probit, loglog, cloglog, cauchit; `rms::orm`, `MASS::polr`) with means, quantiles and the proportional-odds score test
- **Continuation ratio** - The `rms::cr.setup` + `lrm` model, with cohort-specific slopes
- **Penalized GLM** - Poisson, binomial, logistic, negative binomial, Gamma and Tweedie GLMs with a ridge penalty and Normal / Laplace priors (posterior mode, as `arm::bayesglm`)

### Longitudinal, Survival & Smoothing
- **GLMM** - Linear and generalized linear mixed models (Gaussian, Poisson, binomial, negative binomial, Gamma, Tweedie) with random intercepts and slopes, crossed and nested factors, Satterthwaite tests (`lme4`, `lmerTest`)
- **GLS** - Generalized least squares with within-group correlation (AR(1), CAR(1), compound symmetry, spatial, unstructured) and variance functions (`nlme::gls`)
- **AFT** - Accelerated failure time regression with right censoring (Weibull, lognormal, log-logistic, exponential; `survival::survreg`)
- **P-spline** - Penalized B-spline smoother with GCV-selected smoothing (`mgcv::gam(bs = "ps")`)

### Augmented Linear Model (ALM)
- **ALM Regression** - Maximum likelihood with various distributions (Normal, Laplace, Student-t, Gamma, etc.)

### Time-Varying Models
- **LmDynamic** - Dynamic Linear Model with time-varying parameters using information criteria weighting

## Installation

```bash
npm install @sipemu/anofox-regression
```

## Usage

### Browser (ES Modules)

```javascript
import init, {
  OlsRegressor, RidgeRegressor, QuantileRegressor,
  BlsRegressor, PlsRegressor, RlsRegressor, AlmRegressor, LmDynamicRegressor
} from '@sipemu/anofox-regression';

async function main() {
  // Initialize the WASM module
  await init();

  // Create sample data (row-major flat array)
  const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);  // 5 rows, 2 cols
  const y = new Float64Array([2.1, 3.9, 6.2, 7.8, 10.1]);

  // Fit OLS regression
  const ols = new OlsRegressor();
  ols.setWithIntercept(true);
  ols.setComputeInference(true);

  const fitted = ols.fit(x, 5, 2, y);

  // Get results
  const result = fitted.getResult();
  console.log('R-squared:', result.rSquared);
  console.log('Coefficients:', result.coefficients);
  console.log('P-values:', result.pValues);

  // Make predictions
  const xNew = new Float64Array([11, 12]);
  const predictions = fitted.predict(xNew, 1);
  console.log('Prediction:', predictions);
}

main();
```

### Node.js

```javascript
import { readFile } from 'fs/promises';
import { initSync, OlsRegressor } from '@sipemu/anofox-regression';

// Load and initialize WASM synchronously
const wasmBuffer = await readFile('./node_modules/@sipemu/anofox-regression/anofox_regression_js_bg.wasm');
initSync(wasmBuffer);

// Use the library
const ols = new OlsRegressor();
// ...
```

## API Reference

### OlsRegressor

Ordinary Least Squares regression with full statistical inference.

```typescript
class OlsRegressor {
  constructor();
  setWithIntercept(include: boolean): void;
  setComputeInference(compute: boolean): void;
  setConfidenceLevel(level: number): void;  // default: 0.95
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedOls;
}

class FittedOls {
  getResult(): OlsResult;
  getCoefficients(): Float64Array;
  getIntercept(): number | undefined;
  getRSquared(): number;
  predict(x: Float64Array, nRows: number): Float64Array;
}
```

### RidgeRegressor

Ridge regression with L2 regularization.

```typescript
class RidgeRegressor {
  constructor();
  setLambda(lambda: number): void;  // regularization strength
  setWithIntercept(include: boolean): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedRidge;
}
```

### QuantileRegressor

Quantile regression for estimating conditional quantiles.

```typescript
class QuantileRegressor {
  constructor();
  setTau(tau: number): void;  // quantile to estimate (0 < tau < 1)
  setWithIntercept(include: boolean): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedQuantile;
}
```

### IsotonicRegressor

Isotonic (monotonic) regression.

```typescript
class IsotonicRegressor {
  constructor();
  setIncreasing(increasing: boolean): void;
  setOutOfBounds(mode: 'clip' | 'nan' | 'extrapolate'): void;
  fit(x: Float64Array, y: Float64Array): FittedIsotonic;  // 1D data only
}
```

### WlsRegressor

Weighted Least Squares regression.

```typescript
class WlsRegressor {
  constructor();
  setWeights(weights: Float64Array): void;
  setWithIntercept(include: boolean): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedWls;
}
```

### ElasticNetRegressor

Elastic Net with L1+L2 regularization.

```typescript
class ElasticNetRegressor {
  constructor();
  setLambda(lambda: number): void;  // regularization strength
  setAlpha(alpha: number): void;    // L1/L2 mix (0=Ridge, 1=Lasso)
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedElasticNet;
}
```

### PoissonRegressor

Poisson GLM for count data.

```typescript
class PoissonRegressor {
  constructor();
  setLink(link: 'log' | 'identity' | 'sqrt'): void;
  setWithIntercept(include: boolean): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedPoisson;
}
```

### BinomialRegressor

Logistic/Probit regression for binary outcomes.

```typescript
class BinomialRegressor {
  constructor();
  setLink(link: 'logit' | 'probit' | 'cloglog'): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedBinomial;
}
```

### NegativeBinomialRegressor

For overdispersed count data.

```typescript
class NegativeBinomialRegressor {
  constructor();
  setTheta(theta: number): void;      // fixed dispersion
  setEstimateTheta(estimate: boolean): void;  // estimate from data
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedNegativeBinomial;
}
```

### TweedieRegressor

Flexible GLM with Tweedie variance function.

```typescript
class TweedieRegressor {
  constructor();
  static gamma(): TweedieRegressor;  // Gamma regression
  setVarPower(p: number): void;  // 0=Gaussian, 1=Poisson, 2=Gamma, 3=InvGauss
  setLinkPower(p: number): void; // 0=log, 1=identity
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedTweedie;
}
```

### OrdinalRegressor

Cumulative link model for an ordered response, as `rms::orm`.

```typescript
class OrdinalRegressor {
  constructor();
  setLink(link: 'logistic' | 'probit' | 'loglog' | 'cloglog' | 'cauchit'): void;
  setWeights(w: Float64Array): void;
  setOffset(offset: Float64Array): void;
  setPenalty(lambda: number): void;                 // or setPenaltyMatrix(rowMajorP)
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedOrdinal;
}
// FittedOrdinal: getResult(), vcov(), vcovSlopes(), vcovSandwich(), categoryProbabilities,
// exceedanceProbabilities, predictMean, predictMeanWithSe, predictQuantile(x, nRows, q,
// 'interpolated' | 'discrete'), predictQuantileCi, proportionalOddsScoreTest, scoreContributions
// xmeanOrdinaly(x, y, cr) — rms::plot.xmean.ordinaly numbers
```

### ContinuationRatioRegressor

```typescript
class ContinuationRatioRegressor {
  constructor();
  setCohortSpecific(columns: Uint32Array): void;  // slopes that differ by cohort
  setWeights(w: Float64Array): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedContinuationRatio;
}
// FittedContinuationRatio: getResult(), vcov(), conditionalProbabilities, categoryProbabilities
// crSetup(y) — the rms::cr.setup expansion
```

### PenalizedGlmRegressor

```typescript
class PenalizedGlmRegressor {
  constructor();
  // link: Poisson 'log' | 'identity' | 'sqrt', binomial 'logit' | 'probit' | 'cloglog';
  // param: Tweedie power, fixed negative-binomial theta, or logistic threshold
  setFamily(family: 'poisson' | 'binomial' | 'logistic' | 'negative_binomial' | 'gamma' | 'tweedie',
            link?: string, param?: number): void;
  setLambda(lambda: number): void;
  setPriors(priors: { kind: 'flat' | 'normal' | 'laplace'; loc?: number; scale?: number }[]): void;
  setVcovType(vcov: 'laplace' | 'sandwich' | 'naive'): void;
  setOffset(offset: Float64Array): void;
  setComputeInference(compute: boolean): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedPenalizedGlm;
}
// FittedPenalizedGlm: getResult(), predict, predictLinear, predictWithOffset
```

### GlmmRegressor

Mixed model with random intercepts (and slopes) per group, as `lme4::lmer` / `glmer(nAGQ = 0)`.

```typescript
class GlmmRegressor {
  constructor(family?: 'gaussian' | 'poisson' | 'binomial' | 'negative_binomial' | 'gamma' | 'tweedie',
              power?: number);
  setRandomSlopes(columns: Uint32Array): void;
  setRandomSlopesPerFactor(columns: number[][]): void;  // for fitCrossed
  setReml(reml: boolean): void;
  setSatterthwaite(compute: boolean): void;
  setNAgq(nAgq: number): void;
  setNbTheta(theta: number): void;
  setOffset(offset: Float64Array): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array, group: Uint32Array): FittedGlmm;
  // crossed (1|a) + (1|b): [a, b]; nested (1|a/b): [a, ab] with a unique id per (a, b)
  fitCrossed(x: Float64Array, nRows: number, nCols: number, y: Float64Array,
             groups: (number[] | Uint32Array)[]): FittedGlmm;
}
// FittedGlmm: getResult() (fixed effects, SEs, vcov, BLUPs and their SEs, variance components,
// logLik / AIC / BIC, fitted values, residuals), confInt(level), inference('wald' | 'satterthwaite',
// level), randomEffectsConfInt(level), factorRandomCov / factorRandomSd / factorRandomEffects /
// factorRandomEffectsSe / factorRandomEffectsConfInt(f), predictFixed (linear predictor)
```

### GlsRegressor

Generalized least squares for grouped (longitudinal) data, as `nlme::gls`.

```typescript
class GlsRegressor {
  constructor();
  setCorrelation(name: 'independent' | 'compound_symmetry' | 'ar1' | 'car1' | 'exponential'
    | 'gaussian' | 'linear' | 'spherical' | 'unstructured', nugget?: boolean): void;
  setVarIdent(strata: Uint32Array): void;  // or setVarPower(v) / setVarExp(v) / setVarConstant()
  setMethod(method: 'reml' | 'ml'): void;
  // group: subject id per row; time: optional observation times
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array,
      group: Uint32Array, time?: Float64Array): FittedGls;
}
// FittedGls: getResult(), residuals('response' | 'pearson' | 'normalized'),
// predict(x, nRows), predictWithSe(x, nRows), variogram(time, kind, nint)
```

### AftRegressor

Accelerated failure time model with right censoring, as `survival::survreg`.

```typescript
class AftRegressor {
  constructor();
  setDistribution(dist: 'weibull' | 'lognormal' | 'loglogistic' | 'exponential'): void;
  setComputeInference(compute: boolean): void;
  setPriors(priors: { kind: 'flat' | 'normal' | 'laplace'; loc?: number; scale?: number }[]): void;
  setVcovType(vcov: 'laplace' | 'sandwich' | 'naive'): void;
  // event[i] = 1 for an observed event, 0 for a right-censored time
  fit(x: Float64Array, nRows: number, nCols: number, time: Float64Array, event: Uint8Array): FittedAft;
}
// FittedAft: getResult(), predictMedian / predictQuantile(x, nRows, p) / predictSurvival(x, nRows, t)
// / predictCdf / predictLinear, standardizedResiduals, coxSnellResiduals, scoreContributions
```

### PSplineRegressor

Penalized B-spline smoother of one predictor, as `mgcv::gam(y ~ s(x, bs = "ps"), method = "GCV.Cp")`.

```typescript
class PSplineRegressor {
  constructor();
  setNBasis(k: number): void;        // 0 = automatic
  setPenaltyOrder(m: number): void;  // default 2
  setLambda(lambda: number): void;   // fixed; GCV-selected by default
  fit(x: Float64Array, y: Float64Array): FittedPSpline;
}
// FittedPSpline: predict(x), predictInterval(x, 'confidence' | 'prediction', level),
// getEdf(), getSigma2(), getLambda(), getResult()
```

### Convergence

Every iteratively fitted model (`FittedPoisson`, `FittedBinomial`, `FittedNegativeBinomial`,
`FittedTweedie`, `FittedGamma`, `FittedLogistic`, `FittedHuber`, `FittedQuantile`,
`FittedBayesianRidge`, `FittedArd`, `FittedPassiveAggressive`, `FittedAft`, `FittedOrdinal`,
`FittedContinuationRatio`, `FittedPenalizedGlm`, `FittedGlmm`) has
`getConverged(): boolean` and `getIterations(): number`. The IRLS GLMs throw when they don't
converge; `setErrorOnNonConvergence(false)` returns the last iterate instead.

### Diagnostics

Model-based diagnostics are methods on the fitted classes; the rest are free functions on arrays
(matrices are row-major `Float64Array`s with their dimensions).

```typescript
// FittedOls, FittedWls, FittedPoisson, FittedBinomial, FittedLogistic, FittedNegativeBinomial,
// FittedTweedie, FittedGamma (and FittedPenalizedGlm.augment):
fitted.augment(x, nRows, y, residType?: 'response' | 'pearson' | 'deviance', weights?)
//   -> { fitted, residual, stdResidual, studResidual, leverage, cooksD, dffits, looResidual, ... }
fitted.press(x, nRows, y, weights?)    // { press, pressRmse, looRSquared, n }
fitted.partialResiduals(x, nRows, y)   // most fitted classes; [{ term, x, component, partialResidual }]
fittedRidge.loo(x, nRows, y)           // exact ridge leave-one-out

// free functions
concordance(x, y, w?); somers2(p, y, w?); rcorrCens(x, time, event); brierScore(y, p, w?);
lrmStats(y, lp, w?, df); valProb(p, y); giniMd(x); r2Measures(lr, p, n, ess?); nagelkerkeR2(...);
effectiveSampleSize(n, freqs);
varianceInflationFactor(x, nRows, nCols); generalizedVif(x, nRows, nCols, groupSizes);
conditionNumber / conditionDiagnostic / varianceDecompositionProportions(x, nRows, nCols, withIntercept);
computeLeverage(x, nRows, nCols, withIntercept); cooksDistance(res, lev, mse, p); dffits(...);
standardizedResiduals / studentizedResiduals / externallyStudentizedResiduals(...);
pearsonResiduals / devianceResiduals / workingResiduals(y, mu, family, link?, param?);
breuschPagan(res, z, nCols, studentize); whiteTest(res, x, nCols);
addedVariable(x, nRows, nCols, y, w?, withIntercept); checkBinarySeparation / detectSeparation(...)
```

### Inference

```typescript
// methods
fitted.predictSe(x, nRows, { scale?: 'response' | 'link', interval?: 'confidence' | 'prediction', level? })
//   OLS, WLS, Ridge, Poisson, Binomial, NegativeBinomial, Tweedie, Gamma, PenalizedGlm
fitted.likelihood(); fitted.modelLrTest()   // GLM fits, AFT, FittedGlm; lrTest(reduced.likelihood(), full.likelihood())
fittedOls.clusterVcov(x, nRows, [codes, codes2?], 'cr1' | 'hc0' | 'robcov' | { type: 'hc0'..'hc3', cadjust })
//   also WLS, Ordinal (x, nRows, y, ...), AFT (x, nRows, time, event, ...)
fittedOls.hcStandardErrors(x, nRows, 'hc3'); fittedOls.hcInference(x, nRows, 'hc3')
fittedOls.vcov(); fittedOls.varianceFactor(); fittedOls.leverageNew(x, nRows)  // also WLS, Ridge

// free functions
const terms = new TermMap(); terms.addMain('x1', [0, 1, 1]); terms.addInteraction(a, b, false);
anova(beta, vcov, terms, dfResidual?, sigma2?)          // rms::anova
fastbw(beta, vcov, nIntercepts, terms, 'aic' | 'p', threshold?, 'residual' | 'individual', k?)
waldTest(beta, vcov, c, r?, dfResidual?); waldChunk(beta, vcov, columns, dfResidual?)
anovaTable(x, nRows, nCols, y, terms?: number[][], 'I' | 'II')
nestedFTest(xReduced, pReduced, xFull, pFull, y); nestedFTestIndices(xFull, p, y, reduced)
fitGlm(family, x, nRows, nCols, y, { dispersion?: number | 'estimated', weights?, offset?, link?, param? })
glmScoreTest(...); profileConfint(...); lrTest(reduced, full)
lmClusterVcov(...); glmClusterVcov(...); clusterSandwich(bread, scores, nRows, clusters, options)
poolRubin(estimates, vcovs, dfComplete?).waldD1(indices)  // Rubin's rules
approximateModel(...); approximationPath(...); heuristicShrinkage(lr, df); shrinkSlopes(...)
computeHcStandardErrors / computeHcInference / computeXtxInverse / computeVarianceFactor /
computeRidgeVarianceFactor / computePredictionIntervals / intervalsFromVarianceFactor / leverageNew
```

### Missing values, offsets and prior weights

```typescript
// OLS, WLS, Ridge, ElasticNet, Poisson, NegativeBinomial, Tweedie, Gamma, Binomial
setNaAction(action: 'omit' | 'exclude' | 'fail' | 'pass'): void;  // R's na.action, default 'omit'
// Poisson, NegativeBinomial, Tweedie, Gamma, Binomial
setOffset(offset: Float64Array): void;  // per-row offset, e.g. log(exposure)
// on the fitted model: predictions with a new offset
predictWithOffset(x: Float64Array, nRows: number, offset: Float64Array): Float64Array;
// Binomial: prior weights (numbers of trials for a proportion response)
setWeights(weights: Float64Array): void;
```

With `'exclude'`, `residuals` and `fittedValues` of `getResult()` are padded with
`NaN` at the dropped rows. Rows with `NaN` cannot be dropped when an offset or
weights are set: remove them from all inputs first.

### BlsRegressor

Bounded/Non-Negative Least Squares using Lawson-Hanson algorithm.

```typescript
class BlsRegressor {
  constructor();
  static nnls(): BlsRegressor;  // Non-negative least squares
  setWithIntercept(include: boolean): void;
  setLowerBoundAll(bound: number): void;  // Same lower bound for all coefficients
  setUpperBoundAll(bound: number): void;  // Same upper bound for all coefficients
  setLowerBounds(bounds: Float64Array): void;  // Per-variable lower bounds
  setUpperBounds(bounds: Float64Array): void;  // Per-variable upper bounds
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedBls;
}
```

### PlsRegressor

Partial Least Squares using SIMPLS algorithm.

```typescript
class PlsRegressor {
  constructor();
  setNComponents(n: number): void;  // Number of latent components
  setWithIntercept(include: boolean): void;
  setScale(scale: boolean): void;  // Scale X to unit variance
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedPls;
}

class FittedPls {
  getResult(): PlsResult;
  getNComponents(): number;
  transform(x: Float64Array, nRows: number): Float64Array;  // Project to latent space
  predict(x: Float64Array, nRows: number): Float64Array;
}
```

### RlsRegressor

Recursive Least Squares for online learning.

```typescript
class RlsRegressor {
  constructor();
  setWithIntercept(include: boolean): void;
  setForgettingFactor(lambda: number): void;  // 1.0 = standard RLS, <1 = weight recent data
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedRls;
}

class FittedRls {
  getResult(): RlsResult;
  getForgettingFactor(): number;
  predict(x: Float64Array, nRows: number): Float64Array;
}
```

### AlmRegressor

Augmented Linear Model with various error distributions.

```typescript
class AlmRegressor {
  constructor();
  setDistribution(dist: string): void;  // 'normal', 'laplace', 'student_t', 'gamma', etc.
  setWithIntercept(include: boolean): void;
  setComputeInference(compute: boolean): void;
  setMaxIterations(maxIter: number): void;
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedAlm;
}
```

Supported distributions: `normal`, `laplace`, `student_t`, `logistic`, `asymmetric_laplace`,
`generalised_normal`, `log_normal`, `log_laplace`, `gamma`, `inverse_gaussian`, `exponential`,
`poisson`, `negative_binomial`, `beta`, `folded_normal`, `s`.

### LmDynamicRegressor

Dynamic Linear Model with time-varying parameters.

```typescript
class LmDynamicRegressor {
  constructor();
  setIc(ic: 'aic' | 'aicc' | 'bic'): void;  // Information criterion
  setWithIntercept(include: boolean): void;
  setLowessSpan(span: number | null): void;  // LOWESS smoothing (null to disable)
  setMaxModels(max: number): void;  // Limit candidate models
  fit(x: Float64Array, nRows: number, nCols: number, y: Float64Array): FittedLmDynamic;
}

class FittedLmDynamic {
  getResult(): LmDynamicResult;
  getDynamicCoefficients(): Float64Array;  // Time-varying coefficients (row-major)
  getDynamicCoefficientsRows(): number;
  getDynamicCoefficientsCols(): number;
  predict(x: Float64Array, nRows: number): Float64Array;
}
```

## Data Format

All matrix data is passed as flat `Float64Array` in **row-major order**:

```javascript
// For a 3x2 matrix:
// [[1, 2],
//  [3, 4],
//  [5, 6]]
const x = new Float64Array([1, 2, 3, 4, 5, 6]);
const nRows = 3;
const nCols = 2;
```

## Development

### Running Tests

The package includes a comprehensive test suite using [Vitest](https://vitest.dev/).

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Building from Source

To build the WASM package from the Rust source:

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the package
wasm-pack build crates/anofox-regression-js --target web --out-dir pkg
```

## License

MIT
