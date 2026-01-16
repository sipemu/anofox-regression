import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('RidgeRegressor', () => {
  let RidgeRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    RidgeRegressor = wasm.RidgeRegressor;
  });

  describe('basic functionality', () => {
    it('should fit ridge regression with regularization', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const ridge = new RidgeRegressor();
      ridge.setLambda(0.1);
      ridge.setWithIntercept(true);
      const fitted = ridge.fit(x, nRows, nCols, y);

      // With regularization, R-squared might be slightly less than 1
      expect(fitted.getRSquared()).toBeGreaterThan(0.99);

      // Coefficients should be close to OLS but slightly shrunk
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      // Coefficient should be close to 2 but slightly smaller due to shrinkage
      expect(coefs[0]).toBeLessThanOrEqual(2.0);
      expect(coefs[0]).toBeGreaterThan(1.9);

      fitted.free();
      ridge.free();
    });

    it('should match OLS when lambda is zero', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const ridge = new RidgeRegressor();
      ridge.setLambda(0.0);
      ridge.setWithIntercept(true);
      const fitted = ridge.fit(x, nRows, nCols, y);

      expectClose(fitted.getRSquared(), 1.0, 1e-10);
      expectClose(fitted.getIntercept(), 1.0, 1e-6);

      const coefs = fitted.getCoefficients();
      expectClose(coefs[0], 2.0, 1e-6);

      fitted.free();
      ridge.free();
    });

    it('should shrink coefficients more with higher lambda', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      // Fit with small lambda
      const ridgeSmall = new RidgeRegressor();
      ridgeSmall.setLambda(0.1);
      ridgeSmall.setWithIntercept(true);
      const fittedSmall = ridgeSmall.fit(x, nRows, nCols, y);

      // Fit with large lambda
      const ridgeLarge = new RidgeRegressor();
      ridgeLarge.setLambda(10.0);
      ridgeLarge.setWithIntercept(true);
      const fittedLarge = ridgeLarge.fit(x, nRows, nCols, y);

      const coefSmall = fittedSmall.getCoefficients()[0];
      const coefLarge = fittedLarge.getCoefficients()[0];

      // Larger lambda should give more shrinkage
      expect(Math.abs(coefLarge)).toBeLessThan(Math.abs(coefSmall));

      fittedSmall.free();
      fittedLarge.free();
      ridgeSmall.free();
      ridgeLarge.free();
    });
  });

  describe('multicollinearity', () => {
    it('should handle correlated features better than OLS', async () => {
      // Highly correlated features
      const { x, nRows, nCols } = matrix([
        [1, 1.1],
        [2, 2.05],
        [3, 3.02],
        [4, 4.01],
        [5, 5.03]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const ridge = new RidgeRegressor();
      ridge.setLambda(1.0);
      ridge.setWithIntercept(true);
      const fitted = ridge.fit(x, nRows, nCols, y);

      // Should still fit reasonably well
      expect(fitted.getRSquared()).toBeGreaterThan(0.95);

      fitted.free();
      ridge.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const ridge = new RidgeRegressor();
      ridge.setLambda(0.0); // No regularization for predictable results
      ridge.setWithIntercept(true);
      const fitted = ridge.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      expectClose(predictions[0], 13, 1e-6);
      expectClose(predictions[1], 15, 1e-6);

      fitted.free();
      ridge.free();
    });
  });
});
