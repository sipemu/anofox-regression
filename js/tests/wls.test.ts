import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('WlsRegressor', () => {
  let WlsRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    WlsRegressor = wasm.WlsRegressor;
  });

  describe('basic functionality', () => {
    it('should fit weighted least squares regression', async () => {
      // y = 1 + 2*x with different weights
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);
      const weights = vector([1, 1, 1, 1, 1]); // Equal weights = OLS

      const wls = new WlsRegressor();
      wls.setWithIntercept(true);
      wls.setWeights(weights);
      const fitted = wls.fit(x, nRows, nCols, y);

      // Check R-squared (should be 1.0 for perfect fit)
      expectClose(fitted.getRSquared(), 1.0, 1e-10);

      // Check intercept (should be 1)
      expectClose(fitted.getIntercept(), 1.0, 1e-10);

      // Check coefficient (should be 2)
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 2.0, 1e-10);

      fitted.free();
      wls.free();
    });

    it('should handle heteroscedastic data with weights', async () => {
      // Data with increasing variance - use higher weights for more reliable points
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3.1, 4.8, 7.2, 8.7, 11.5]);
      // Higher weights for points closer to expected line
      const weights = vector([10, 10, 5, 5, 1]);

      const wls = new WlsRegressor();
      wls.setWithIntercept(true);
      wls.setWeights(weights);
      const fitted = wls.fit(x, nRows, nCols, y);

      // R-squared should be high but not perfect due to noise
      expect(fitted.getRSquared()).toBeGreaterThan(0.9);

      fitted.free();
      wls.free();
    });

    it('should fit without intercept', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 4, 6, 8, 10]);
      const weights = vector([1, 1, 1, 1, 1]);

      const wls = new WlsRegressor();
      wls.setWithIntercept(false);
      wls.setWeights(weights);
      const fitted = wls.fit(x, nRows, nCols, y);

      // Coefficient should be 2
      const coefs = fitted.getCoefficients();
      expectClose(coefs[0], 2.0, 1e-10);

      // Intercept should be undefined
      expect(fitted.getIntercept()).toBeUndefined();

      fitted.free();
      wls.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);
      const weights = vector([1, 1, 1, 1, 1]);

      const wls = new WlsRegressor();
      wls.setWithIntercept(true);
      wls.setWeights(weights);
      const fitted = wls.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      expectClose(predictions[0], 13, 1e-10); // 1 + 2*6
      expectClose(predictions[1], 15, 1e-10); // 1 + 2*7

      fitted.free();
      wls.free();
    });
  });

  describe('multiple predictors', () => {
    it('should handle multiple features', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 3]
      ]);
      const y = vector([6, 8, 9, 11, 16]); // y = 1 + 2*x1 + 3*x2
      const weights = vector([1, 1, 1, 1, 1]);

      const wls = new WlsRegressor();
      wls.setWithIntercept(true);
      wls.setWeights(weights);
      const fitted = wls.fit(x, nRows, nCols, y);

      expectClose(fitted.getIntercept(), 1.0, 1e-10);
      const coefs = fitted.getCoefficients();
      expectClose(coefs[0], 2.0, 1e-10);
      expectClose(coefs[1], 3.0, 1e-10);

      fitted.free();
      wls.free();
    });
  });
});
