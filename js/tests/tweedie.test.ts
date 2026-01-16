import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('TweedieRegressor', () => {
  let TweedieRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    TweedieRegressor = wasm.TweedieRegressor;
  });

  describe('gamma regression (var_power=2)', () => {
    it('should fit Gamma regression via static factory', async () => {
      // Positive continuous data
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1.5, 3.2, 4.8, 6.1, 7.9]);

      const tweedie = TweedieRegressor.gamma();
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      tweedie.free();
    });

    it('should fit Gamma regression via var_power', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1.5, 3.2, 4.8, 6.1, 7.9]);

      const tweedie = new TweedieRegressor();
      tweedie.setVarPower(2.0); // Gamma
      tweedie.setLinkPower(0.0); // Log link
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      tweedie.free();
    });
  });

  describe('Poisson-like (var_power=1)', () => {
    it('should behave like Poisson with var_power=1', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 7, 20, 55, 148]);

      const tweedie = new TweedieRegressor();
      tweedie.setVarPower(1.0); // Poisson variance
      tweedie.setLinkPower(0.0); // Log link
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      tweedie.free();
    });
  });

  describe('Gaussian (var_power=0)', () => {
    it('should behave like Gaussian with var_power=0', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const tweedie = new TweedieRegressor();
      tweedie.setVarPower(0.0); // Gaussian variance
      tweedie.setLinkPower(1.0); // Identity link
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      tweedie.free();
    });
  });

  describe('Compound Poisson-Gamma (1 < var_power < 2)', () => {
    it('should handle compound Poisson-Gamma distribution', async () => {
      // Insurance claims-like data (zeros and positive values)
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 5, 0, 12, 0, 8, 0, 25, 0, 30]);

      const tweedie = new TweedieRegressor();
      tweedie.setVarPower(1.5); // Compound Poisson-Gamma
      tweedie.setLinkPower(0.0); // Log link
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeDefined();

      fitted.free();
      tweedie.free();
    });
  });

  describe('Inverse Gaussian (var_power=3)', () => {
    it('should fit Inverse Gaussian regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([0.5, 1.2, 2.1, 3.5, 5.2]);

      const tweedie = new TweedieRegressor();
      tweedie.setVarPower(3.0); // Inverse Gaussian
      tweedie.setLinkPower(-1.0); // Inverse link (common for IG)
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      tweedie.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1.5, 3.2, 4.8, 6.1, 7.9]);

      const tweedie = TweedieRegressor.gamma();
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [3], [6]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      expect(predictions[0]).toBeGreaterThan(0);
      expect(predictions[1]).toBeGreaterThan(0);

      fitted.free();
      tweedie.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([1.5, 3.2, 4.8, 6.1, 7.9]);

      const tweedie = TweedieRegressor.gamma();
      tweedie.setWithIntercept(true);
      const fitted = tweedie.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.deviance).toBeDefined();
      expect(result.coefficients).toBeDefined();

      fitted.free();
      tweedie.free();
    });
  });
});
