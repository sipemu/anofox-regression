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
generalizedVifTerms(x, nRows, nCols, groupSizes);  // [{ gvif, df, adjustedGvif }]
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

### Survival

```typescript
// event: Uint8Array (1 = event, 0 = censored); codes: non-negative integers
const km = kaplanMeier(time, event, { confLevel?, confType?: 'log' | 'log-log' | 'plain' })
const curves = survfit(time, event, { entry?, weights?, cluster?, strata? },
  { confLevel?, confType?, hazard?: 'nelson-aalen' | 'fleming-harrington', timefix?, robust? })
km.getResult()      // { time, nRisk, nEvent, nCensor, surv, stdErr, lower, upper, cumhaz, stdChaz, ... }
km.median(); km.restrictedMean(tau); km.summary(tau); km.survivalAt(t)
survdiff(time, event, group, strata?, rho?)   // rho = 0 log-rank, 1 Peto–Peto

const cox = coxph(x, nRows, nCols, time, event, { entry?, weights?, offset?, strata? },
  { ties?: 'efron' | 'breslow', maxIter?, eps?, ridge?, confLevel?, timefix? })
cox.getResult(); cox.getCoefficients(); cox.getVcov(); cox.concordance(); cox.dxy(); cox.r2()
cox.martingaleResiduals(); cox.devianceResiduals(); cox.scoreResiduals(); cox.dfbeta(); cox.dfbetas()
cox.schoenfeldResiduals(scaled?); cox.robustVcov(clusters)
cox.coxZph('km' | 'rank' | 'identity' | 'log', terms?)   // proportional-hazards test
cox.predictSurvival(x0, stratum?)                        // survfit(fit, newdata)
cox.likelihood(); cox.modelLrTest()
```

### Splines, designs and effects

```typescript
// spline bases (matrices are flat row-major)
const rcs = Rcs.fromData(age, 4);            // or new Rcs(knots); rcs.setNorm('none' | 'last-interval' | 'range')
const b = rcs.basis(age, 'age');             // Basis: columns(), names(), nonlinear(), nonlinearColumns()
rcs.restate(intercept, coef).eval(x)         // Hmisc::rcspline.restate
new Lsp([3, 6]).basis(x, 'x'); restrictedInteraction(b1, b2); rcsDefaultKnots(x, nk, fractied?)
const d = new Design(); d.add('age', b); d.addRestrictedInteraction(0, 1); d.matrix(); d.terms()

// rms-style effects; options { df? (t reference), level? }
const spec = new ModelSpec();
spec.rcs('age', rcs, age); spec.categorical('sex', sex); spec.linear('chol', chol); spec.interaction(0, 2);
spec.design([age, sex, chol]).matrix()        // fit any regressor on it
spec.summary(beta, vcov, true)                // summary.rms (exp() → odds / hazard ratios)
spec.contrast(beta, vcov, true, { age: 60 }, { age: 45 }); spec.doubleContrast(...)
spec.predictGrid(beta, vcov, true, 0, values?); spec.predictGridSimultaneous(...)
const nom = spec.nomogram(beta, true, linearPredictors, ticks?, maxscale?, nint?);
nom.getResult(); nom.functionAxis((lp) => 1 / (1 + Math.exp(-lp))); nom.pointsToLp(total)
numericLimits(x); simultaneousCritical(rows, vcov, options)
```

### Validation and calibration

```typescript
const r = Resamples.bootstrapSeeded(n, 200, 42);  // or bootstrap(n, samples), crossvalidation(perm, B),
                                                  // crossvalidationSeeded, point632, point632Seeded
const t = validateLrm(x, nRows, nCols, y, r);     // validateOls; t.get('Dxy') → { orig, optimism, corrected, ... }
calibrateLrm(x, nRows, nCols, y, r, predy?); calibrateOls(...)
calibrateCph(x, nRows, nCols, time, event, u, r, m, cuts?, { eps: 1e-4, maxIter: 10 })

// any modelling process: fit(rows) → model (null = failed), measure(model, rows, apparent) → number[]
predabResample(r, ['R2', 'MSE'], fit, measure)
const b = bootcov(bootstrapRows(n, 500, 1), (rows) => coefficients);  // or clusterBootstrapRows(ids, B, seed)
b.getVcov(); b.percentileCi(0.95); b.linearCombinationCi(c, 0.95)
rankIntervals(original, replicates, 0.9)
```

### Data reduction, transformations and imputation

```typescript
spearman2(x, y, 2); spearman2Categorical(codes, y); hoeffdingD(x, nRows, nCols)
const vc = varclus(x, nRows, nCols, { similarity: 'spearman', transform: 'square', linkage: 'complete' });
vc.sim(); vc.hclust().getResult(); hclust(diss, n, 'average').cutHeight(h)
redun(x, nRows, nCols, ['numeric', 'linear', 'categorical', ...], { r2: 0.8 })
principalComponents(x, nRows, nCols, true).transform(newX, nNew, k); clusterScores(x, nRows, nCols, [[0, 1], [2]], 1)
transcan(x, nRows, nCols, categoricalFlags)                 // NaN = missing

boxCox(x, nRows, nCols, y, { lambdas? }); boxCoxTransform(y, λ); boxCoxInverse(z, λ)
smearingEstimate(lp, residuals, Math.exp, 'mean' | 'median' | 'fitted' | q)
const a = areg(x, nRows, nCols, y, { nk: 4, xTypes: ['spline', 'linear', 'categorical'], yType: 'spline' });
a.inverseY(lp); a.bootstrap(indices)

const imp = aregImpute(x, nRows, nCols, kinds, { nImpute: 5, seed: 1 });   // NaN = missing
imp.complete(x, nRows, m)                                    // m-th completed data set, row-major
```

### Regularisation paths, cross-validation and pentrace

```typescript
// path options { alpha? (1 lasso … 0 ridge), standardize?, withIntercept?, tol?, maxIter? }
const lambdas = lambdaGrid(x, nRows, nCols, y, { alpha: 0.5 }, 100);
fitPath(x, nRows, nCols, y, lambdas, { alpha: 0.5 }).predict(newX, nNew)
const cv = cvPath(x, nRows, nCols, y, lambdas, foldIds, { alpha: 0.5 }).getResult();  // lambdaMin, lambda1se
cvPathSplits(x, nRows, nCols, y, lambdas, splits, options, 'mae')

// folds and splits (keys: numbers or strings)
kfoldIds(n, k, seed); groupKfoldIds(groups, k); logoIds(groups); stratifiedKfoldIds(strata, k, seed?)
stratifiedGroupKfoldIds(groups, strata, k); quantileStrata(y, bins); splitsFromFoldIds(foldIds)
timeSeriesSplits(n, { nSplits, testSize?, gap?, maxTrainSize? }); groupTimeSeriesSplits(timeKeys, opts)
nestedCvPath(x, nRows, nCols, y, null, outerSplits, { alpha: 1 }, { inner: { type: 'kfold', k: 5, seed: 1 }, rule: '1se' })

// rms penalties: a number or { simple, nonlinear?, interaction?, nonlinearInteraction? }
const pm = penaltyMatrix(x, nRows, nCols, termMap, categoricalTerms, { simple: 1, nonlinear: 4 });
new OrdinalRegressor().setPenaltyMatrix(pm); penalizedOls(x, nRows, nCols, y, pm)
pentrace(x, nRows, nCols, y, termMap, categoricalTerms, penaltyGrid([0, 2], [0, 4, 16], [], [])).bestFit()
```

### Mediation, shrinkage, quantile inference and streaming fits

```typescript
mediate(y, m, x, { method: 'bootstrap' | 'sobel' | 'quasi-bayesian', covariates: [c1, c2], nSims: 1000, seed: 42 })
ebShrink(estimates, standardErrors, { method: 'dl' | 'none', tauSquared? })

const inf = new QuantileRegressor().fit(x, nRows, nCols, y).inference(x, nRows, y, 'nid', 0.95);
inf.getResult(); inf.predictInterval(newX, nNew, 0.9)
quantileInference(x, nRows, nCols, y, 0.5, true, { type: 'boot', seed: 1, r: 500 }, 0.95)
quantileProcess(x, nRows, nCols, y, [0.1, 0.5, 0.9], true, 'nid', 0.9)

// streaming / distributed fits: accumulate chunks, merge, serialise, fit
const s = SufficientStats.fromData(x1, n1, p, y1).merged(SufficientStats.fromData(x2, n2, p, y2));
new OlsRegressor().fitFromState(SufficientStats.fromBytes(s.toBytes()))
const acc = new MomentAccumulator(p); acc.pushRows(x, nRows, y); new RidgeRegressor().fitFromAccumulator(acc)
new RidgeRegressor().path(x, nRows, nCols, y, lambdas); new ElasticNetRegressor().cvPath(..., foldIds)
new LarsRegressor().fit(x, nRows, nCols, y).path()
```

### More on the fitted models

```typescript
// IRLS GLMs (Poisson, Binomial, NegativeBinomial, Tweedie, Gamma)
fitted.devianceResiduals(); fitted.pearsonResiduals(); fitted.workingResiduals()
fitted.predictLinear(x, n); fitted.predictCount(x, n)          // predictProbability / predictEta / predictMu
fitted.predictWithSe(x, n, 'link', 'confidence', 0.95); fitted.vcov()
// Bayesian ridge / ARD
fitted.predictStd(x, n); fitted.credibleIntervals(0.9); fitted.sigma()
// online updates
fittedRls.update(xRow, y); const s = new PaState(p); new PassiveAggressiveRegressor().partialFit(s, xRow, y)
// helpers
lowess(x, y, { frac: 2 / 3 }); pretty(lo, hi, 5); estimateThetaMl(y, mu); gramConditionNumber(x, n, p, true)
```

### Parity with the Rust crate

The npm package has the same functionality as the Rust crate. CI lists the
public Rust API and the WASM exports and fails when a public item has no
binding, unless [`js/parity-allowlist.txt`](https://github.com/sipemu/anofox-regression/blob/main/js/parity-allowlist.txt) lists it with
a reason. Builders and option structs are `set*` methods and plain option
objects, result structs are plain objects, and enums are strings.

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
