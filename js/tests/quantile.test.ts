import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('QuantileRegressor', () => {
  let QuantileRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    QuantileRegressor = wasm.QuantileRegressor;
  });

  describe('median regression (tau=0.5)', () => {
    it('should fit median regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const qr = new QuantileRegressor();
      qr.setTau(0.5); // Median
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      // For perfect linear data, median regression should match OLS
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 2.0, 0.1);

      fitted.free();
      qr.free();
    });

    it('should be robust to outliers', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      // Most points follow y = 2*x, but point at x=10 is an outlier
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 100]);

      const qr = new QuantileRegressor();
      qr.setTau(0.5);
      qr.setWithIntercept(false);
      const fitted = qr.fit(x, nRows, nCols, y);

      // Median regression should be less affected by outlier
      const coefs = fitted.getCoefficients();
      // Should be close to 2, not pulled toward outlier as much as OLS would be
      expect(coefs[0]).toBeLessThan(5); // OLS would give much higher
      expect(coefs[0]).toBeGreaterThan(1.5);

      fitted.free();
      qr.free();
    });
  });

  describe('quantile levels', () => {
    it('should fit 25th percentile regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const qr = new QuantileRegressor();
      qr.setTau(0.25); // 25th percentile
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      // Lower quantile should have lower predictions
      const result = fitted.getResult();
      expect(result.tau).toBe(0.25);

      fitted.free();
      qr.free();
    });

    it('should fit 75th percentile regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const qr = new QuantileRegressor();
      qr.setTau(0.75); // 75th percentile
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.tau).toBe(0.75);

      fitted.free();
      qr.free();
    });

    it('should fit 90th percentile regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const qr = new QuantileRegressor();
      qr.setTau(0.9);
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.tau).toBe(0.9);

      fitted.free();
      qr.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const qr = new QuantileRegressor();
      qr.setTau(0.5);
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Should be approximately 13 and 15
      expect(predictions[0]).toBeGreaterThan(11);
      expect(predictions[0]).toBeLessThan(15);
      expect(predictions[1]).toBeGreaterThan(13);
      expect(predictions[1]).toBeLessThan(17);

      fitted.free();
      qr.free();
    });
  });

  describe('multiple features', () => {
    it('should handle multiple predictors', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 3]
      ]);
      const y = vector([6, 8, 9, 11, 16]); // y ≈ 1 + 2*x1 + 3*x2

      const qr = new QuantileRegressor();
      qr.setTau(0.5);
      qr.setWithIntercept(true);
      const fitted = qr.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(2);

      fitted.free();
      qr.free();
    });
  });
});
