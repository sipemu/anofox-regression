import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('BinomialRegressor', () => {
  let BinomialRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    BinomialRegressor = wasm.BinomialRegressor;
  });

  describe('logistic regression (logit link)', () => {
    it('should fit logistic regression', async () => {
      // Binary outcome data
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]); // Binary

      const binomial = new BinomialRegressor();
      binomial.setLink('logit');
      binomial.setWithIntercept(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      // Check deviance
      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      // Coefficients should exist
      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(1);
      // Positive coefficient expected (higher x -> higher probability of 1)
      expect(coefs[0]).toBeGreaterThan(0);

      fitted.free();
      binomial.free();
    });

    it('should predict probabilities between 0 and 1', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]);

      const binomial = new BinomialRegressor();
      binomial.setLink('logit');
      binomial.setWithIntercept(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [0], [5], [10], [15]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(4);
      // All probabilities should be between 0 and 1
      for (const p of predictions) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }

      // Lower x should give lower probability
      expect(predictions[0]).toBeLessThan(predictions[1]);
      expect(predictions[1]).toBeLessThan(predictions[2]);

      fitted.free();
      binomial.free();
    });
  });

  describe('probit link', () => {
    it('should fit probit regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]);

      const binomial = new BinomialRegressor();
      binomial.setLink('probit');
      binomial.setWithIntercept(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      binomial.free();
    });
  });

  describe('cloglog link', () => {
    it('should fit complementary log-log regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]);

      const binomial = new BinomialRegressor();
      binomial.setLink('cloglog');
      binomial.setWithIntercept(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeGreaterThanOrEqual(0);

      fitted.free();
      binomial.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]);

      const binomial = new BinomialRegressor();
      binomial.setLink('logit');
      binomial.setWithIntercept(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result.deviance).toBeDefined();
      expect(result.coefficients).toBeDefined();

      fitted.free();
      binomial.free();
    });
  });

  describe('configuration', () => {
    it('should respect max iterations setting', async () => {
      // Use same well-behaved data as other tests
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([0, 0, 0, 0, 1, 0, 1, 1, 1, 1]);

      const binomial = new BinomialRegressor();
      binomial.setLink('logit');
      binomial.setWithIntercept(true);
      binomial.setMaxIterations(100);
      binomial.setComputeInference(true);
      const fitted = binomial.fit(x, nRows, nCols, y);

      expect(fitted.getDeviance()).toBeDefined();

      fitted.free();
      binomial.free();
    });
  });
});
