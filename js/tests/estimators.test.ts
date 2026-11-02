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

  it('LassoLars drops a variable whose coefficient crosses zero (#305)', () => {
    // sklearn lars_path(method="lasso"): feature 2 leaves at knot 3 and
    // re-enters with the opposite sign at the last step.
    const xs = [
      0.22, -0.34, -0.15, -0.4, -0.6, -0.24, -0.27, 0.58, 0.25, -1.46, -0.84, 0.27, -0.0, -1.02,
      -0.49, -1.19, 1.56, -0.93, 0.39, -0.15, 0.52, 1.34, 1.17, 1.26, 0.91, -0.84, 0.18, -1.02,
      0.18, -0.42, 0.18, 1.92, -1.03, -0.48, -0.85, 0.06, 0.55, 0.85, 0.52, 0.37, 0.94, -0.35,
      0.36, -0.05, 0.42, -0.7, -0.12, 0.47, 0.04, 0.72, 0.31, -0.05, -0.27, 0.31, -0.25, 1.15,
      -0.22, -1.22, -0.77, 1.55, -0.21, -0.24, -0.24, 0.56,
    ];
    const ys = [
      0.41, -0.19, -0.73, -1.47, 0.58, 1.9, -0.61, -0.29, -1.87, 2.17, 0.7, -0.11, 0.4, 0.32,
      -1.39, -1.02,
    ];
    const reg = new wasm.LarsRegressor();
    reg.setMethod('lasso');
    const f = reg.fit(new Float64Array(xs), 16, 4, new Float64Array(ys));
    expect(Array.from(f.active())).toEqual([1, 0, 3, 2]);
    expect(f.getAlphas().length).toBe(7);
    expect(f.coefsPath()[4][2]).toBe(0);
    const b = f.getCoefficients();
    expectClose(b[2], -1.1654478794527305, 1e-9);
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
