// #210: numbers through the WASM bindings against R. The reference values
// are those of the Rust R validation (see references.ts).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';
import * as R from './references';

const col = (v: number[]) => ({ x: vector(v), nRows: v.length, nCols: 1 });

describe('R validation through WASM', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('logistic regression matches glm(binomial)', () => {
    const { x, nRows, nCols } = col(R.X_BINOMIAL_LOGIT);
    const m = new wasm.LogisticRegression();
    m.setComputeInference(true);
    m.setTolerance(1e-14);
    m.setMaxIterations(200);
    const f = m.fit(x, nRows, nCols, vector(R.Y_BINOMIAL_LOGIT));
    const r = f.getResult();
    expectClose(r.intercept, R.BINOMIAL_LOGIT[0], 1e-7);
    expectClose(r.coefficients[0], R.BINOMIAL_LOGIT[1], 1e-7);
    expectClose(r.stdErrors[0], R.BINOMIAL_LOGIT[3], 1e-7);
    expectClose(r.deviance, R.BINOMIAL_LOGIT[4], 1e-7);
    f.free();
    m.free();
  });

  it('Gamma regression matches glm(Gamma("log"))', () => {
    const { x, nRows, nCols } = col(R.X_GAMMA_LOG);
    const m = new wasm.GammaRegressor();
    m.setComputeInference(true);
    m.setTolerance(1e-13);
    m.setMaxIterations(200);
    const f = m.fit(x, nRows, nCols, vector(R.Y_GAMMA_LOG));
    const r = f.getResult();
    expectClose(r.intercept, R.GAMMA_LOG[0], 1e-7);
    expectClose(r.coefficients[0], R.GAMMA_LOG[1], 1e-7);
    expectClose(r.stdErrors[0], R.GAMMA_LOG[3], 1e-7);
    expectClose(f.getDeviance(), R.GAMMA_LOG[4], 1e-6);
    f.free();
    m.free();
  });

  it('Huber regression matches MASS::rlm(k = 1.35)', () => {
    // tests/issue36_huber_inference.rs
    const n = 20;
    const xs = Array.from({ length: n }, (_, i) => i + 1);
    const ys = xs.map((v) => 1 + 2 * v + (((v * 7) % 5) - 2) * 0.4);
    ys[5] = 40;
    ys[14] = 5;
    const { x, nRows, nCols } = col(xs);
    const m = new wasm.HuberRegressor();
    m.setEpsilon(1.35);
    m.setAlpha(0);
    m.setMaxIterations(500);
    m.setTolerance(1e-12);
    const f = m.fit(x, nRows, nCols, vector(ys));
    expectClose(f.getCoefficients()[0], 1.9769296831089778, 1e-6);
    expectClose(f.getIntercept(), 1.2866827718001757, 1e-5);
    expect(f.getNOutliers()).toBeGreaterThanOrEqual(2);
    f.free();
    m.free();
  });
});
