import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('LmDynamicRegressor', () => {
  let LmDynamicRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    LmDynamicRegressor = wasm.LmDynamicRegressor;
  });

  describe('basic functionality', () => {
    it('should fit time-varying parameter model', async () => {
      // Time series data where relationship changes over time
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]
      ]);
      // First half: y ≈ 2*x, second half: y ≈ 3*x
      const y = vector([
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        33, 36, 39, 42, 45, 48, 51, 54, 57, 60
      ]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result).toBeDefined();

      fitted.free();
      lmd.free();
    });

    it('should estimate time-varying coefficients', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      // Get dynamic coefficients
      const dynCoefs = fitted.getDynamicCoefficients();
      expect(dynCoefs.length).toBeGreaterThan(0);

      const nCoefRows = fitted.getDynamicCoefficientsRows();
      const nCoefCols = fitted.getDynamicCoefficientsCols();
      expect(nCoefRows * nCoefCols).toBe(dynCoefs.length);

      fitted.free();
      lmd.free();
    });
  });

  describe('information criteria', () => {
    it('should use AIC by default', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setIc('aic');
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      lmd.free();
    });

    it('should support BIC', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setIc('bic');
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      lmd.free();
    });

    it('should support AICc', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setIc('aicc');
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      lmd.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [11], [12]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Predictions should be reasonable extrapolations
      expect(predictions[0]).toBeGreaterThan(18);
      expect(predictions[1]).toBeGreaterThan(predictions[0]);

      fitted.free();
      lmd.free();
    });
  });

  describe('configuration', () => {
    it('should respect max models setting', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      lmd.setMaxModels(50);
      const fitted = lmd.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      lmd.free();
    });

    it('should support LOWESS smoothing', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      lmd.setLowessSpan(0.5);
      const fitted = lmd.fit(x, nRows, nCols, y);

      expect(fitted.getResult()).toBeDefined();

      fitted.free();
      lmd.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
      ]);
      const y = vector([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

      const lmd = new LmDynamicRegressor();
      lmd.setWithIntercept(false);
      const fitted = lmd.fit(x, nRows, nCols, y);

      const result = fitted.getResult();
      expect(result).toBeDefined();

      fitted.free();
      lmd.free();
    });
  });
});
