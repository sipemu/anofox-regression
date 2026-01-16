import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose, expectArrayClose } from './setup';

describe('OlsRegressor', () => {
  let OlsRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    OlsRegressor = wasm.OlsRegressor;
  });

  describe('basic functionality', () => {
    it('should fit a simple linear regression', async () => {
      // y = 2 + 3*x
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([5, 8, 11, 14, 17]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      // Check R-squared (should be 1.0 for perfect fit)
      expectClose(fitted.getRSquared(), 1.0, 1e-10);

      // Check intercept (should be 2)
      expectClose(fitted.getIntercept(), 2.0, 1e-10);

      // Check coefficient (should be 3)
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 3.0, 1e-10);

      // Clean up
      fitted.free();
      ols.free();
    });

    it('should fit regression without intercept', async () => {
      // y = 2*x (no intercept)
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 4, 6, 8, 10]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(false);
      const fitted = ols.fit(x, nRows, nCols, y);

      // Check coefficient (should be 2)
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 2.0, 1e-10);

      // Intercept should be undefined when not included
      expect(fitted.getIntercept()).toBeUndefined();

      fitted.free();
      ols.free();
    });

    it('should handle multiple predictors', async () => {
      // y = 1 + 2*x1 + 3*x2
      const { x, nRows, nCols } = matrix([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 3]
      ]);
      const y = vector([6, 8, 9, 11, 16]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      expectClose(fitted.getRSquared(), 1.0, 1e-10);
      expectClose(fitted.getIntercept(), 1.0, 1e-10);

      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(2);
      expectClose(coefs[0], 2.0, 1e-10);
      expectClose(coefs[1], 3.0, 1e-10);

      fitted.free();
      ols.free();
    });
  });

  describe('predictions', () => {
    it('should make accurate predictions', async () => {
      // y = 1 + 2*x
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      // Predict on new data
      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7], [8]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(3);
      expectClose(predictions[0], 13, 1e-10); // 1 + 2*6
      expectClose(predictions[1], 15, 1e-10); // 1 + 2*7
      expectClose(predictions[2], 17, 1e-10); // 1 + 2*8

      fitted.free();
      ols.free();
    });
  });

  describe('statistical inference', () => {
    it('should compute inference statistics when enabled', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      // y with some noise
      const y = vector([2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 17.9, 20.2]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      ols.setComputeInference(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      const result = fitted.getResult();

      // Check that inference results are present
      expect(result.coefficients).toBeDefined();
      expect(result.stdErrors).toBeDefined();
      expect(result.tStatistics).toBeDefined();
      expect(result.pValues).toBeDefined();
      expect(result.rSquared).toBeDefined();

      // R-squared should be very high for this near-linear data
      expect(result.rSquared).toBeGreaterThan(0.99);

      fitted.free();
      ols.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result object', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2.1, 3.9, 6.1, 7.9, 10.1]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      ols.setComputeInference(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      const result = fitted.getResult();

      // Verify structure
      expect(typeof result.rSquared).toBe('number');
      expect(typeof result.adjRSquared).toBe('number');
      expect(Array.isArray(result.coefficients) || result.coefficients instanceof Float64Array).toBe(true);

      fitted.free();
      ols.free();
    });

    it('should provide adjusted R-squared', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

      const ols = new OlsRegressor();
      ols.setWithIntercept(true);
      const fitted = ols.fit(x, nRows, nCols, y);

      const rSquared = fitted.getRSquared();
      const adjRSquared = fitted.getAdjRSquared();

      // For perfect fit, both should be 1.0
      expectClose(rSquared, 1.0, 1e-10);
      expectClose(adjRSquared, 1.0, 1e-10);

      fitted.free();
      ols.free();
    });
  });
});
