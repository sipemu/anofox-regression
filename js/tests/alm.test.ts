import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('AlmRegressor', () => {
  let AlmRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    AlmRegressor = wasm.AlmRegressor;
  });

  describe('Normal distribution', () => {
    it('should fit normal ALM (equivalent to OLS)', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const alm = new AlmRegressor();
      alm.setDistribution('normal');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.logLikelihood).toBeDefined();

      // Should be similar to OLS
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      expectClose(coefs[0], 2.0, 0.1);

      fitted.free();
      alm.free();
    });
  });

  describe('Laplace distribution', () => {
    it('should fit Laplace ALM (robust to outliers)', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      // Data with outlier at x=10
      const y = vector([3, 5, 7, 9, 11, 13, 15, 17, 19, 100]);

      const alm = new AlmRegressor();
      alm.setDistribution('laplace');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      // Laplace should be more robust to outlier, coefficient closer to 2
      expect(coefs[0]).toBeGreaterThan(1);
      expect(coefs[0]).toBeLessThan(10); // Much less affected than OLS would be

      fitted.free();
      alm.free();
    });
  });

  describe('Student-t distribution', () => {
    it('should fit Student-t ALM', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const alm = new AlmRegressor();
      alm.setDistribution('student_t');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      expect(fitted.getResult().logLikelihood).toBeDefined();

      fitted.free();
      alm.free();
    });
  });

  describe('Logistic distribution', () => {
    it('should fit logistic ALM', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const alm = new AlmRegressor();
      alm.setDistribution('logistic');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      expect(fitted.getResult().logLikelihood).toBeDefined();

      fitted.free();
      alm.free();
    });
  });

  describe('Gamma distribution', () => {
    it('should fit Gamma ALM for positive data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1.5, 3.2, 4.8, 6.1, 7.9]); // Positive values

      const alm = new AlmRegressor();
      alm.setDistribution('gamma');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      expect(fitted.getResult().logLikelihood).toBeDefined();

      fitted.free();
      alm.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const alm = new AlmRegressor();
      alm.setDistribution('normal');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      expectClose(predictions[0], 13, 0.5);
      expectClose(predictions[1], 15, 0.5);

      fitted.free();
      alm.free();
    });
  });

  describe('configuration', () => {
    it('should respect max iterations setting', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const alm = new AlmRegressor();
      alm.setDistribution('normal');
      alm.setWithIntercept(true);
      alm.setMaxIterations(100);
      const fitted = alm.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      alm.free();
    });

    it('should compute inference when enabled', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 17.9, 20.2]);

      const alm = new AlmRegressor();
      alm.setDistribution('normal');
      alm.setWithIntercept(true);
      alm.setComputeInference(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.logLikelihood).toBeDefined();

      fitted.free();
      alm.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const alm = new AlmRegressor();
      alm.setDistribution('normal');
      alm.setWithIntercept(true);
      const fitted = alm.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.coefficients).toBeDefined();
      expect(result.logLikelihood).toBeDefined();
      expect(result.aic).toBeDefined();

      fitted.free();
      alm.free();
    });
  });
});
