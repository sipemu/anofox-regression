import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';

describe('IsotonicRegressor', () => {
  let IsotonicRegressor: any;

  beforeAll(async () => {
    const wasm = await getWasmModule();
    IsotonicRegressor = wasm.IsotonicRegressor;
  });

  describe('increasing regression', () => {
    it('should fit monotonically increasing function', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 3, 2, 5, 4]); // Not monotonic

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      const fitted = iso.fit(x, y);

      const fittedValues = fitted.getFittedValues();
      expect(fittedValues.length).toBe(5);

      // Check monotonicity: each value should be >= previous
      for (let i = 1; i < fittedValues.length; i++) {
        expect(fittedValues[i]).toBeGreaterThanOrEqual(fittedValues[i - 1] - 1e-10);
      }

      fitted.free();
      iso.free();
    });

    it('should preserve already monotonic data', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 2, 3, 4, 5]); // Already monotonic

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      const fitted = iso.fit(x, y);

      const fittedValues = fitted.getFittedValues();

      // Should be very close to original
      for (let i = 0; i < fittedValues.length; i++) {
        expectClose(fittedValues[i], y[i], 1e-10);
      }

      expect(fitted.getRSquared()).toBeGreaterThan(0.99);

      fitted.free();
      iso.free();
    });

    it('should return high R-squared for good fit', async () => {
      const x = vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const y = vector([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.0, 8.1, 8.9, 10.2]);

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      const fitted = iso.fit(x, y);

      expect(fitted.getRSquared()).toBeGreaterThan(0.95);

      fitted.free();
      iso.free();
    });
  });

  describe('decreasing regression', () => {
    it('should fit monotonically decreasing function', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([5, 3, 4, 2, 1]); // Not monotonic decreasing

      const iso = new IsotonicRegressor();
      iso.setIncreasing(false);
      const fitted = iso.fit(x, y);

      const fittedValues = fitted.getFittedValues();
      expect(fittedValues.length).toBe(5);

      // Check monotonicity: each value should be <= previous
      for (let i = 1; i < fittedValues.length; i++) {
        expect(fittedValues[i]).toBeLessThanOrEqual(fittedValues[i - 1] + 1e-10);
      }

      expect(fitted.isIncreasing()).toBe(false);

      fitted.free();
      iso.free();
    });
  });

  describe('predictions', () => {
    it('should make predictions on new data', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 2, 3, 4, 5]);

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      const fitted = iso.fit(x, y);

      const xNew = vector([1.5, 2.5, 3.5, 4.5]);
      const predictions = fitted.predict(xNew);

      expect(predictions.length).toBe(4);

      // Predictions should be monotonic
      for (let i = 1; i < predictions.length; i++) {
        expect(predictions[i]).toBeGreaterThanOrEqual(predictions[i - 1] - 1e-10);
      }

      fitted.free();
      iso.free();
    });

    it('should handle out-of-bounds prediction with clip mode', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 2, 3, 4, 5]);

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      iso.setOutOfBounds('clip');
      const fitted = iso.fit(x, y);

      const xNew = vector([0, 6]); // Out of training range
      const predictions = fitted.predict(xNew);

      expect(predictions.length).toBe(2);
      // With clip mode, predictions should be clipped to training range
      expect(predictions[0]).toBeGreaterThanOrEqual(0.9); // Close to min fitted value
      expect(predictions[1]).toBeLessThanOrEqual(5.1); // Close to max fitted value

      fitted.free();
      iso.free();
    });

    it('should handle out-of-bounds prediction with extrapolate mode', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 4, 6, 8, 10]); // y = 2*x

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      iso.setOutOfBounds('extrapolate');
      const fitted = iso.fit(x, y);

      const xNew = vector([0, 6]);
      const predictions = fitted.predict(xNew);

      expect(predictions.length).toBe(2);
      // With extrapolation, values beyond range are allowed
      // x=0 should give something less than 2
      // x=6 should give something more than 10

      fitted.free();
      iso.free();
    });
  });

  describe('result object', () => {
    it('should return comprehensive result', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 3, 2, 5, 4]);

      const iso = new IsotonicRegressor();
      iso.setIncreasing(true);
      const fitted = iso.fit(x, y);

      const result = fitted.getResult();
      expect(result.rSquared).toBeDefined();
      expect(result.increasing).toBe(true);

      fitted.free();
      iso.free();
    });
  });
});
