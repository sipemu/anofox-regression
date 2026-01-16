import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

describe('ElasticNetRegressor', () => {
  let ElasticNetRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    ElasticNetRegressor = wasm.ElasticNetRegressor;
  });

  describe('basic functionality', () => {
    it('should fit elastic net regression', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]); // y = 1 + 2*x

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.1);
      elasticNet.setAlpha(0.5); // 50% L1, 50% L2
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      // Should fit reasonably well
      expect(fitted.getRSquared()).toBeGreaterThan(0.95);

      fitted.free();
      elasticNet.free();
    });

    it('should behave like ridge when alpha=0', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.1);
      elasticNet.setAlpha(0.0); // Pure L2 = Ridge
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      expect(fitted.getRSquared()).toBeGreaterThan(0.99);

      fitted.free();
      elasticNet.free();
    });

    it('should behave like lasso when alpha=1', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.1);
      elasticNet.setAlpha(1.0); // Pure L1 = Lasso
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      expect(fitted.getRSquared()).toBeGreaterThan(0.95);

      fitted.free();
      elasticNet.free();
    });
  });

  describe('feature selection', () => {
    it('should potentially zero out irrelevant features with high alpha', async () => {
      // x1 is relevant, x2 is noise
      const { x, nRows, nCols } = matrix([
        [1, 0.5],
        [2, 0.3],
        [3, 0.8],
        [4, 0.2],
        [5, 0.6],
        [6, 0.4],
        [7, 0.7],
        [8, 0.1],
        [9, 0.9],
        [10, 0.5]
      ]);
      // y depends only on x1
      const y = vector([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.5);
      elasticNet.setAlpha(0.9); // High L1 for sparsity
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      const coefs = fitted.getCoefficients();
      expect(coefs.length).toBe(2);

      // The coefficient for x1 should be significant
      expect(Math.abs(coefs[0])).toBeGreaterThan(0.5);

      // The coefficient for x2 (noise) should be smaller
      expect(Math.abs(coefs[1])).toBeLessThan(Math.abs(coefs[0]));

      fitted.free();
      elasticNet.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.01); // Small regularization
      elasticNet.setAlpha(0.5);
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      const { x: xNew, nRows: nRowsNew } = matrix([
        [6], [7]
      ]);
      const predictions = fitted.predict(xNew, nRowsNew);

      expect(predictions.length).toBe(2);
      // Should be close to 13 and 15 with small regularization
      expect(predictions[0]).toBeGreaterThan(12);
      expect(predictions[0]).toBeLessThan(14);
      expect(predictions[1]).toBeGreaterThan(14);
      expect(predictions[1]).toBeLessThan(16);

      fitted.free();
      elasticNet.free();
    });
  });

  describe('configuration', () => {
    it('should respect tolerance setting', async () => {
      const { x, nRows, nCols } = matrix([
        [1], [2], [3], [4], [5]
      ]);
      const y = vector([3, 5, 7, 9, 11]);

      const elasticNet = new ElasticNetRegressor();
      elasticNet.setLambda(0.1);
      elasticNet.setAlpha(0.5);
      elasticNet.setTolerance(1e-8);
      elasticNet.setMaxIterations(1000);
      const fitted = elasticNet.fit(x, nRows, nCols, y);

      expect(fitted.getRSquared()).toBeGreaterThan(0.95);

      fitted.free();
      elasticNet.free();
    });
  });
});
