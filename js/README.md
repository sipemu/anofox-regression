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

### Augmented Linear Model (ALM)
- **ALM Regression** - Maximum likelihood with various distributions (Normal, Laplace, Student-t, Gamma, etc.)

## Installation

```bash
npm install @sipemu/anofox-regression
```

## Usage

### Browser (ES Modules)

```javascript
import init, {
  OlsRegressor, RidgeRegressor, QuantileRegressor,
  BlsRegressor, PlsRegressor, RlsRegressor, AlmRegressor
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

## License

MIT
