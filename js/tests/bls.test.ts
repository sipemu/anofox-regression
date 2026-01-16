import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('BlsRegressor', () => {
  let BlsRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    BlsRegressor = wasm.BlsRegressor;
  });

  describe('non-negative least squares', () => {
    it('should create NNLS regressor via static factory', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [1, 2]
      ]);
      const y = vector([1, 2, 3, 4, 5]);

      const bls = BlsRegressor.nnls();
      const fitted = bls.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(2);

      // All coefficients should be non-negative
      expect(coefs[0]).toBeGreaterThanOrEqual(0);
      expect(coefs[1]).toBeGreaterThanOrEqual(0);

      fitted.free();
      bls.free();
    });

    it('should enforce non-negativity even when OLS would give negative', async () => {
      // Design data where OLS would give negative coefficient
      const { x, nRows, nCols } = matrix([
        [1, 5],
        [2, 4],
        [3, 3],
        [4, 2],
        [5, 1]
      ]);
      // y depends positively on x1, negatively on x2 (for OLS)
      const y = vector([1, 3, 5, 7, 9]);

      const bls = BlsRegressor.nnls();
      const fitted = bls.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();

      // All coefficients should be non-negative
      for (let i = 0; i < coefs.length; i++) {
        expect(coefs[i]).toBeGreaterThanOrEqual(-1e-10); // Allow tiny numerical errors
      }

      fitted.free();
      bls.free();
    });
  });

  describe('bounded least squares', () => {
    it('should respect lower bounds', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const bls = new BlsRegressor();
      bls.setWithIntercept(true);
      bls.setLowerBoundAll(1.5); // Lower bound of 1.5 for coefficient
      const fitted = bls.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs[0]).toBeGreaterThanOrEqual(1.5 - 1e-10);

      fitted.free();
      bls.free();
    });

    it('should respect upper bounds', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const bls = new BlsRegressor();
      bls.setWithIntercept(true);
      bls.setUpperBoundAll(1.5); // Upper bound of 1.5 for coefficient (true is 2)
      const fitted = bls.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs[0]).toBeLessThanOrEqual(1.5 + 1e-10);

      fitted.free();
      bls.free();
    });

    it('should handle per-variable bounds', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 3]
      ]);
      const y = vector([6, 8, 9, 11, 16]); // y = 1 + 2*x1 + 3*x2

      const bls = new BlsRegressor();
      bls.setWithIntercept(true);
      bls.setLowerBounds(new Float64Array([0, 2.5])); // x1 >= 0, x2 >= 2.5
      bls.setUpperBounds(new Float64Array([3, 4]));   // x1 <= 3, x2 <= 4
      const fitted = bls.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(2);

      // Check bounds are respected
      expect(coefs[0]).toBeGreaterThanOrEqual(-1e-10);
      expect(coefs[0]).toBeLessThanOrEqual(3 + 1e-10);
      expect(coefs[1]).toBeGreaterThanOrEqual(2.5 - 1e-10);
      expect(coefs[1]).toBeLessThanOrEqual(4 + 1e-10);

      fitted.free();
      bls.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 4, 6, 8, 10]); // y = 2*x

      const bls = BlsRegressor.nnls();
      const fitted = bls.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Should be approximately 12 and 14
      expect(predictions[0]).toBeGreaterThan(10);
      expect(predictions[1]).toBeGreaterThan(12);

      fitted.free();
      bls.free();
    });
  });

  describe('result object', () => {
    it('should return result with R-squared', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 4, 6, 8, 10]);

      const bls = BlsRegressor.nnls();
      const fitted = bls.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.rSquared).toBeDefined();
      expect(result.rSquared).toBeGreaterThan(0.99);

      fitted.free();
      bls.free();
    });
  });
});
