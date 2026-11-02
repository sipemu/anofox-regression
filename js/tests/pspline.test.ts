// #287: PSplineRegressor against mgcv::gam(bs = "ps", method = "GCV.Cp")
// (tests/r_validation_pspline.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear, near } from './rel';
import * as R from './references';

describe('PSplineRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  function check(f: any, stats: number[], fitted: number[], pred: number[], tol: number) {
    const r = f.getResult();
    const n = R.PS_X.length;
    const rss = r.residuals.reduce((s: number, e: number) => s + e * e, 0);
    const gcv = (n * rss) / (n - f.getEdf()) ** 2;
    allNear([f.getEdf(), f.getSigma2(), gcv], stats.slice(0, 3), tol, 'edf / sig2 / gcv');
    near(f.getLambda(), stats[3], 1e3 * tol, 'lambda');
    allNear(r.fittedValues, fitted, tol, 'fitted');
    const p = f.predictInterval(vector(R.PS_X_NEW), 'confidence', 0.95);
    allNear([...p.fit, ...p.se], pred, tol, 'predict / se');
    for (let i = 0; i < p.fit.length; i++) {
      expect(p.lower[i]).toBeLessThan(p.fit[i]);
      expect(p.upper[i]).toBeGreaterThan(p.fit[i]);
    }
    allNear(f.predict(vector(R.PS_X_NEW)), p.fit, 1e-14, 'predict');
  }

  it('GCV fits match mgcv', () => {
    const m = new wasm.PSplineRegressor();
    m.setNBasis(12);
    check(m.fit(vector(R.PS_X), vector(R.PS_Y)), R.PS_GCV_K12_STATS, R.PS_GCV_K12_FITTED, R.PS_GCV_K12_PRED, 1e-6);
    m.setNBasis(20);
    m.setPenaltyOrder(3);
    check(
      m.fit(vector(R.PS_X), vector(R.PS_Y)),
      R.PS_GCV_K20_M3_STATS,
      R.PS_GCV_K20_M3_FITTED,
      R.PS_GCV_K20_M3_PRED,
      1e-6,
    );
  });

  it('a fixed lambda matches mgcv sp', () => {
    const m = new wasm.PSplineRegressor();
    m.setNBasis(12);
    m.setLambda(2.0);
    check(m.fit(vector(R.PS_X), vector(R.PS_Y)), R.PS_FIXED_K12_STATS, R.PS_FIXED_K12_FITTED, R.PS_FIXED_K12_PRED, 1e-10);
  });

  it('prediction intervals are wider than confidence intervals', () => {
    const f = new wasm.PSplineRegressor().fit(vector(R.PS_X), vector(R.PS_Y));
    const c = f.predictInterval(vector([1, 2]), 'confidence', 0.95);
    const p = f.predictInterval(vector([1, 2]), 'prediction', 0.95);
    expect(p.se[0]).toBeGreaterThan(c.se[0]);
    expect(() => f.predictInterval(vector([1]), 'bogus', 0.95)).toThrow();
  });

  it('rejects mismatched lengths', () => {
    expect(() => new wasm.PSplineRegressor().fit(vector([1, 2, 3]), vector([1, 2]))).toThrow();
  });
});
