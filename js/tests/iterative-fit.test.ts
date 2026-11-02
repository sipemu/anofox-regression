// #287: IterativeFit (converged, iterations) on every iterative estimator.
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, matrix, vector } from './setup';

const X = [[0.1], [0.5], [0.9], [1.4], [1.8], [2.3], [2.9], [3.2], [3.8], [4.4], [4.9], [5.3]];
const COUNTS = [1, 2, 1, 3, 4, 3, 6, 7, 9, 12, 15, 20];
const BINARY = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1];
const POSITIVE = COUNTS.map((c) => c + 0.5);

describe('getConverged / getIterations', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('are exposed on every iterative wrapper', () => {
    const { x, nRows, nCols } = matrix(X);
    const fits: [string, any][] = [
      ['PoissonRegressor', new wasm.PoissonRegressor().fit(x, nRows, nCols, vector(COUNTS))],
      ['NegativeBinomialRegressor', new wasm.NegativeBinomialRegressor().fit(x, nRows, nCols, vector(COUNTS))],
      ['GammaRegressor', new wasm.GammaRegressor().fit(x, nRows, nCols, vector(POSITIVE))],
      ['TweedieRegressor', new wasm.TweedieRegressor().fit(x, nRows, nCols, vector(POSITIVE))],
      ['BinomialRegressor', new wasm.BinomialRegressor().fit(x, nRows, nCols, vector(BINARY))],
      ['LogisticRegression', new wasm.LogisticRegression().fit(x, nRows, nCols, vector(BINARY))],
      ['HuberRegressor', new wasm.HuberRegressor().fit(x, nRows, nCols, vector(POSITIVE))],
      ['QuantileRegressor', new wasm.QuantileRegressor().fit(x, nRows, nCols, vector(POSITIVE))],
      ['BayesianRidge', new wasm.BayesianRidge().fit(x, nRows, nCols, vector(POSITIVE))],
      ['ArdRegression', new wasm.ArdRegression().fit(x, nRows, nCols, vector(POSITIVE))],
      ['PassiveAggressiveRegressor', new wasm.PassiveAggressiveRegressor().fit(x, nRows, nCols, vector(POSITIVE))],
    ];
    for (const [name, f] of fits) {
      expect(typeof f.getConverged(), name).toBe('boolean');
      expect(f.getIterations(), name).toBeGreaterThan(0);
      f.free();
    }
  });

  it('report non-convergence when the iteration cap is hit', () => {
    const { x, nRows, nCols } = matrix(X);
    const cases: [string, number[]][] = [
      ['PoissonRegressor', COUNTS],
      ['NegativeBinomialRegressor', COUNTS],
      ['GammaRegressor', POSITIVE],
      ['TweedieRegressor', POSITIVE],
      ['BinomialRegressor', BINARY],
    ];
    for (const [name, y] of cases) {
      const m = new wasm[name]();
      m.setMaxIterations(1);
      // with θ estimated the outer alternation can still converge
      if (name === 'NegativeBinomialRegressor') m.setTheta(2.0);
      // throws by default, as the Rust builders do
      expect(() => m.fit(x, nRows, nCols, vector(y)), name).toThrow(/convergence/);
      m.setErrorOnNonConvergence(false);
      const f = m.fit(x, nRows, nCols, vector(y));
      expect(f.getConverged(), name).toBe(false);
      expect(f.getIterations(), name).toBeGreaterThanOrEqual(1);
    }
  });

  it('ALM, GLS and Theil–Sen report convergence (#281)', () => {
    const n = 40;
    const x2 = new Float64Array(n * 2);
    const y2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x2[2 * i] = Math.sin(i * 0.37) * 2;
      x2[2 * i + 1] = ((i * 7) % 11) / 5 - 1;
      y2[i] = Math.max(Math.exp(0.4 + 0.6 * x2[2 * i] - 0.3 * x2[2 * i + 1]) + ((i * 3) % 5) * 0.3, 0.05);
    }
    const alm = new wasm.AlmRegressor();
    alm.setDistribution('gamma');
    expect(alm.fit(x2, n, 2, y2).getConverged()).toBe(true);
    alm.setMaxIterations(1);
    expect(alm.fit(x2, n, 2, y2).getConverged()).toBe(false);
    const gls = new wasm.GlsRegressor();
    gls.setCorrelation('ar1');
    const group = new Uint32Array(Array.from({ length: n }, (_, i) => Math.floor(i / 8)));
    const g = gls.fit(x2, n, 2, y2, group);
    expect(g.getConverged()).toBe(true);
    expect(g.getIterations()).toBeGreaterThanOrEqual(1);
    const ts = new wasm.TheilSenRegressor();
    expect(ts.fit(x2, n, 2, y2).getConverged()).toBe(true);
    ts.setMaxIter(1);
    const short = ts.fit(x2, n, 2, y2);
    expect(short.getConverged()).toBe(false);
    expect(short.getIterations()).toBe(1);
  });
});
