// #210: the estimators that had no JS test (aid, bayesian, gamma, huber,
// lars, logistic, lowess, passive-aggressive, ransac, theil-sen; gamma,
// huber and logistic are in r-validation.test.ts).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector, expectClose } from './setup';

// y = 1 + 2 x1 − x2 exactly
const X = [
  [0.1, 1.0], [0.7, -0.3], [1.3, 0.4], [2.0, 2.2], [2.4, -1.1],
  [3.1, 0.9], [3.6, 0.0], [4.2, -0.6], [4.9, 1.7], [5.5, 0.3],
];
const Y = X.map(([a, b]) => 1 + 2 * a - b);

describe('estimators without JS tests', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('Theil–Sen (one feature) is the median of the pairwise slopes', () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8];
    const ys = [2.1, 3.9, 6.2, 7.8, 30.0, 12.1, 13.8, 16.2];
    const slopes: number[] = [];
    for (let i = 0; i < xs.length; i++)
      for (let j = i + 1; j < xs.length; j++) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
    slopes.sort((a, b) => a - b);
    const mid = slopes.length / 2;
    const slope = slopes.length % 2 ? slopes[Math.floor(mid)] : (slopes[mid - 1] + slopes[mid]) / 2;
    const res = ys.map((v, i) => v - slope * xs[i]).sort((a, b) => a - b);
    const intercept = (res[3] + res[4]) / 2;
    const { x, nRows, nCols } = matrix(xs.map((v) => [v]));
    const f = new wasm.TheilSenRegressor().fit(x, nRows, nCols, vector(ys));
    expectClose(f.getCoefficients()[0], slope, 1e-12);
    expectClose(f.getIntercept(), intercept, 1e-12);
  });

  it('LARS / lasso, Bayesian ridge, ARD, RANSAC and passive-aggressive recover an exact line', () => {
    const { x, nRows, nCols } = matrix(X);
    const y = vector(Y);
    const lars = new wasm.LarsRegressor().fit(x, nRows, nCols, y);
    expectClose(lars.getCoefficients()[0], 2, 1e-8);
    expectClose(lars.getCoefficients()[1], -1, 1e-8);
    expectClose(lars.getIntercept(), 1, 1e-8);
    expect(lars.getAlphas().length).toBeGreaterThan(0);
    for (const cls of ['BayesianRidge', 'ArdRegression']) {
      const f = new wasm[cls]().fit(x, nRows, nCols, y);
      expectClose(f.getCoefficients()[0], 2, 1e-3);
      expectClose(f.getCoefficients()[1], -1, 1e-3);
      const p = f.predict(x, nRows);
      expect(p.length).toBe(nRows);
    }
    const ransac = new wasm.RansacRegressor();
    ransac.setRandomState(BigInt(1));
    const r = ransac.fit(x, nRows, nCols, y);
    expectClose(r.getCoefficients()[0], 2, 1e-8);
    expect(r.getNInliers()).toBe(nRows);
    const pa = new wasm.PassiveAggressiveRegressor();
    pa.setEpsilon(0);
    pa.setMaxIter(5000);
    pa.setTolerance(1e-12);
    const p = pa.fit(x, nRows, nCols, y).predict(x, nRows);
    for (let i = 0; i < nRows; i++) expectClose(p[i], Y[i], 0.5);
  });

  it('LOWESS of a line is the line', () => {
    const y = Array.from({ length: 15 }, (_, i) => 3 - 0.5 * i);
    const s = wasm.lowessSmooth(vector(y), 0.5);
    for (let i = 0; i < y.length; i++) expectClose(s[i], y[i], 1e-10);
  });

  it('AID classifies regular and intermittent demand', () => {
    const regular = vector(Array.from({ length: 40 }, (_, i) => 20 + (i % 5)));
    const sparse = vector(Array.from({ length: 40 }, (_, i) => (i % 6 === 0 ? 3 : 0)));
    const aid = new wasm.AidClassifier();
    const a = aid.classify(regular);
    const b = aid.classify(sparse);
    expect(a.getDemandType()).not.toBe(b.getDemandType());
    expect(b.getZeroProportion()).toBeGreaterThan(0.5);
    expectClose(a.getZeroProportion(), 0, 1e-12);
  });
});
