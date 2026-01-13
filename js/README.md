# @sipemu/anofox-regression

WebAssembly bindings for [anofox-regression](https://github.com/sipemu/anofox-regression), a comprehensive statistical regression library.

## Features

- **OLS Regression** - Ordinary Least Squares with full inference (standard errors, p-values, confidence intervals)
- **Ridge Regression** - L2 regularization for handling multicollinearity
- **Quantile Regression** - Estimate conditional quantiles (median, quartiles, etc.)
- **Isotonic Regression** - Monotonic regression using Pool Adjacent Violators Algorithm
- **Poisson Regression** - GLM for count data with log/identity/sqrt link functions

## Installation

```bash
npm install @sipemu/anofox-regression
```

## Usage

### Browser (ES Modules)

```javascript
import init, { OlsRegressor, RidgeRegressor, QuantileRegressor } from '@sipemu/anofox-regression';

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
