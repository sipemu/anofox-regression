import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('RlsRegressor', () => {
  let RlsRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    RlsRegressor = wasm.RlsRegressor;
  });

  describe('basic functionality', () => {
    it('should fit recursive least squares', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const rls = new RlsRegressor();
      rls.setWithIntercept(true);
      const fitted = rls.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.rSquared).toBeGreaterThan(0.99);

      // Coefficients should be close to [2] with intercept ~1
      const coefs = result.coefficients;
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 2.0, 0.1);

      fitted.free();
      rls.free();
    });

    it('should handle forgetting factor for non-stationary data', async () => {
      // Simulating concept drift - relationship changes over time
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      // First half: y = 2*x, Second half: y = 3*x
      const y = vector([2, 4, 6, 8, 10, 18, 21, 24, 27, 30]);

      const rls = new RlsRegressor();
      rls.setWithIntercept(false);
      rls.setForgettingFactor(0.9); // Discount older observations
      const fitted = rls.fit(x, nRows, nCols, y);

      // With forgetting, should adapt to recent data (coefficient closer to 3)
      const coefs = fitted.getResult().coefficients;
      expect(coefs[0]).toBeGreaterThan(2.0);

      expect(fitted.getForgettingFactor()).toBe(0.9);

      fitted.free();
      rls.free();
    });

    it('should behave like OLS with forgetting factor 1.0', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const rls = new RlsRegressor();
      rls.setWithIntercept(true);
      rls.setForgettingFactor(1.0); // Standard RLS = OLS
      const fitted = rls.fit(x, nRows, nCols, y);

      expectClose(fitted.getResult().rSquared, 1.0, 1e-6);

      fitted.free();
      rls.free();
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

      const rls = new RlsRegressor();
      rls.setWithIntercept(true);
      const fitted = rls.fit(x, nRows, nCols, y);

      expect(fitted.getResult().rSquared).toBeGreaterThan(0.99);

      fitted.free();
      rls.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const rls = new RlsRegressor();
      rls.setWithIntercept(true);
      const fitted = rls.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Should be close to 13 and 15
      expectClose(predictions[0], 13, 0.5);
      expectClose(predictions[1], 15, 0.5);

      fitted.free();
      rls.free();
    });
  });

  describe('online learning simulation', () => {
    it('should work with sequentially arriving data', async () => {
      // First batch
      const { x: x1, nRows: nRows1, nCols: nCols1 } = matrix([
        [1], [2], [3]
      ]);
      const y1 = vector([3, 5, 7]); // y = 1 + 2*x

      const rls = new RlsRegressor();
      rls.setWithIntercept(true);
      const fitted1 = rls.fit(x1, nRows1, nCols1, y1);

      expect(fitted1.getResult().rSquared).toBeGreaterThan(0.99);

      // In a real online scenario, you'd update the model incrementally
      // Here we just verify it works with small batches

      fitted1.free();
      rls.free();
    });
  });
});
