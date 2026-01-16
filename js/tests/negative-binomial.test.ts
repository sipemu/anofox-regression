import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('NegativeBinomialRegressor', () => {
  let NegativeBinomialRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    NegativeBinomialRegressor = wasm.NegativeBinomialRegressor;
  });

  describe('basic functionality', () => {
    it('should fit negative binomial regression', async () => {
      // Overdispersed count data
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      // Counts with extra variance (overdispersion)
      const y = vector([2, 5, 8, 15, 12, 25, 30, 28, 45, 50]);

      const nb = new NegativeBinomialRegressor();
      nb.setWithIntercept(true);
      nb.setEstimateTheta(true);
      const fitted = nb.fit(x, nRows, nCols, y);

      // Check deviance
      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      // Theta (dispersion) should be estimated
      const theta = fitted.getTheta();
      expect(theta).toBeGreaterThan(0);

      fitted.free();
      nb.free();
    });

    it('should work with fixed theta', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 5, 10, 20, 40]);

      const nb = new NegativeBinomialRegressor();
      nb.setWithIntercept(true);
      nb.setTheta(1.0); // Fixed dispersion
      const fitted = nb.fit(x, nRows, nCols, y);

      expect(fitted.getTheta()).toBeCloseTo(1.0, 1);

      fitted.free();
      nb.free();
    });
  });

  describe('predictions', () => {
    it('should predict positive counts', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 5, 10, 20, 40]);

      const nb = new NegativeBinomialRegressor();
      nb.setWithIntercept(true);
      nb.setEstimateTheta(true);
      const fitted = nb.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [3], [6]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Predictions should be positive
      expect(predictions[0]).toBeGreaterThan(0);
      expect(predictions[1]).toBeGreaterThan(0);

      fitted.free();
      nb.free();
    });
  });

  describe('vs Poisson', () => {
    it('should handle overdispersion better than Poisson', async () => {
      // Highly variable count data
      const { x, nRows, nCols } = matrix([
        [1], [1], [2], [2], [3], [3], [4], [4], [5], [5]
      ]);
      // High variance at each x level
      const y = vector([1, 10, 5, 20, 8, 35, 15, 50, 20, 80]);

      const nb = new NegativeBinomialRegressor();
      nb.setWithIntercept(true);
      nb.setEstimateTheta(true);
      const fitted = nb.fit(x, nRows, nCols, y);

      // NB should estimate a finite theta for overdispersed data
      const theta = fitted.getTheta();
      expect(theta).toBeGreaterThan(0);
      expect(theta).toBeLessThan(1000); // Not infinite (would indicate Poisson is adequate)

      fitted.free();
      nb.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([2, 5, 10, 20, 40]);

      const nb = new NegativeBinomialRegressor();
      nb.setWithIntercept(true);
      nb.setEstimateTheta(true);
      const fitted = nb.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.deviance).toBeDefined();
      expect(result.coefficients).toBeDefined();
      expect(result.theta).toBeDefined();

      fitted.free();
      nb.free();
    });
  });
});
