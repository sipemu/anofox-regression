import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('PoissonRegressor', () => {
  let PoissonRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    PoissonRegressor = wasm.PoissonRegressor;
  });

  describe('log link (default)', () => {
    it('should fit Poisson regression with log link', async () => {
      // Count data that follows exponential relationship
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 7, 20, 55, 148]); // Approximately exp(0.5 + x)

      const poisson = new PoissonRegressor();
      poisson.setLink('log');
      poisson.setWithIntercept(true);
      const fitted = poisson.fit(x, nRows, nCols, y);

      // Check deviance is computed
      const deviance = fitted.getDeviance();
      expect(deviance).toBeDefined();
      expect(deviance).toBeGreaterThanOrEqual(0);

      // Check coefficients
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);

      fitted.free();
      poisson.free();
    });

    it('should predict counts', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 7, 20, 55, 148]);

      const poisson = new PoissonRegressor();
      poisson.setLink('log');
      poisson.setWithIntercept(true);
      const fitted = poisson.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [3], [4]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // All predictions should be positive (counts)
      expect(predictions[0]).toBeGreaterThan(0);
      expect(predictions[1]).toBeGreaterThan(0);

      fitted.free();
      poisson.free();
    });
  });

  describe('identity link', () => {
    it('should fit Poisson with identity link', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // Linear relationship

      const poisson = new PoissonRegressor();
      poisson.setLink('identity');
      poisson.setWithIntercept(true);
      const fitted = poisson.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeDefined();

      fitted.free();
      poisson.free();
    });
  });

  describe('sqrt link', () => {
    it('should fit Poisson with sqrt link', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1, 4, 9, 16, 25]); // Quadratic relationship

      const poisson = new PoissonRegressor();
      poisson.setLink('sqrt');
      poisson.setWithIntercept(true);
      const fitted = poisson.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeDefined();

      fitted.free();
      poisson.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 7, 20, 55, 148]);

      const poisson = new PoissonRegressor();
      poisson.setLink('log');
      poisson.setWithIntercept(true);
      const fitted = poisson.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.deviance).toBeDefined();
      expect(result.coefficients).toBeDefined();

      fitted.free();
      poisson.free();
    });
  });
});
