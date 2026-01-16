import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('PlsRegressor', () => {
  let PlsRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    PlsRegressor = wasm.PlsRegressor;
  });

  describe('basic functionality', () => {
    it('should fit PLS regression with specified components', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10]
      ]);
      const y = vector([10, 13, 16, 19, 22, 25, 28, 31]);

      const pls = new PlsRegressor();
      pls.setNComponents(2);
      pls.setWithIntercept(true);
      const fitted = pls.fit(x, nRows, nCols, y);

      // Should fit well
      const result = fitted.getResult();
      expect(result.rSquared).toBeGreaterThan(0.9);

      // Should use specified number of components
      expect(fitted.getNComponents()).toBe(2);

      fitted.free();
      pls.free();
    });

    it('should handle highly collinear features', async () => {
      // Highly correlated features that would cause problems for OLS
      const { x, nRows, nCols } = matrix([
        [1, 1.01, 0.99],
        [2, 2.02, 1.98],
        [3, 3.01, 2.99],
        [4, 4.03, 3.97],
        [5, 5.02, 4.98],
        [6, 6.01, 5.99],
        [7, 7.02, 6.98],
        [8, 8.01, 7.99]
      ]);
      const y = vector([3, 6, 9, 12, 15, 18, 21, 24]); // y = 3 * x1

      const pls = new PlsRegressor();
      pls.setNComponents(1); // Only need 1 component for this relationship
      pls.setWithIntercept(true);
      const fitted = pls.fit(x, nRows, nCols, y);

      expect(fitted.getResult().rSquared).toBeGreaterThan(0.99);

      fitted.free();
      pls.free();
    });

    it('should scale features when requested', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 100],
        [2, 200],
        [3, 300],
        [4, 400],
        [5, 500]
      ]);
      const y = vector([101, 202, 303, 404, 505]);

      const pls = new PlsRegressor();
      pls.setNComponents(2);
      pls.setWithIntercept(true);
      pls.setScale(true); // Scale X to unit variance
      const fitted = pls.fit(x, nRows, nCols, y);

      expect(fitted.getResult().rSquared).toBeGreaterThan(0.99);

      fitted.free();
      pls.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9]
      ]);
      const y = vector([5, 8, 11, 14, 17, 20, 23, 26]); // y = 2 + x1 + x2

      const pls = new PlsRegressor();
      pls.setNComponents(2);
      pls.setWithIntercept(true);
      const fitted = pls.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [9, 10],
        [10, 11]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Should be close to 29 and 32
      expect(predictions[0]).toBeGreaterThan(27);
      expect(predictions[0]).toBeLessThan(31);
      expect(predictions[1]).toBeGreaterThan(30);
      expect(predictions[1]).toBeLessThan(34);

      fitted.free();
      pls.free();
    });
  });

  describe('transform', () => {
    it('should project data to latent space', async () => {
      const { x, nRows, nCols } = matrix([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7]
      ]);
      const y = vector([6, 9, 12, 15, 18]);

      const pls = new PlsRegressor();
      pls.setNComponents(2);
      pls.setWithIntercept(true);
      const fitted = pls.fit(x, nRows, nCols, y);

      // Transform the data to latent space
      const transformed = fitted.transform(x, nRows);

      // Should have nRows * nComponents elements
      expect(transformed.length).toBe(nRows * 2);

      fitted.free();
      pls.free();
    });
  });
});
