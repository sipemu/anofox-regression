// 0.12: rebuilding fitted models from their parts (fromParts /
// fromInformation) predicts as the fit; the HC covariance matrix
// (tests/model_from_parts.rs, tests/covariance_012.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear } from './rel';

const N = 120;

function uniform(n: number, seed: number): number[] {
  let s = seed;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) % 2147483648;
    out.push((s + 0.5) / 2147483648);
  }
  return out;
}

const u = uniform(3 * N, 5);
const x: number[] = [];
for (let i = 0; i < N; i++) x.push(4 * u[i] - 2, u[N + i]);
const yOrd = Array.from({ length: N }, (_, i) => {
  const e = u[2 * N + i];
  const latent = 0.9 * x[2 * i] - 1.2 * x[2 * i + 1] + Math.log(e / (1 - e));
  return Math.min(3, Math.max(0, Math.floor(((latent + 3) / 6) * 4)));
});
const xNew = vector([-2, 0.1, -0.5, 0.4, 0.3, 0.9, 1.8, 0.5]);

describe('fromParts', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('multinomial logistic', () => {
    const m = new wasm.MultinomialLogisticRegressor();
    const f = m.fit(vector(x), N, 2, vector(yOrd));
    const coef = f.coefficients().flat();
    const g = wasm.FittedMultinomialLogistic.fromParts(vector(f.classes()), vector(coef), vector(f.intercepts()), 2);
    expect(g.predictProba(xNew, 4)).toEqual(f.predictProba(xNew, 4));
    expect(Array.from(g.predict(xNew, 4))).toEqual(Array.from(f.predict(xNew, 4)));
  });

  it('ordinal: covariance and compact information', () => {
    const f = new wasm.OrdinalRegressor().fit(vector(x), N, 2, vector(yOrd));
    const r = f.getResult();
    expect(r.zValues.length).toBe(2);
    expect(r.pValues[0]).toBeGreaterThanOrEqual(0);
    const g = wasm.FittedOrdinal.fromParts('logistic', vector(r.levels), vector(r.intercepts),
      vector(r.coefficients), vector(f.vcov()));
    expect(Array.from(g.categoryProbabilities(xNew, 4))).toEqual(Array.from(f.categoryProbabilities(xNew, 4)));
    allNear(g.predictMeanWithSe(xNew, 4).se, f.predictMeanWithSe(xNew, 4).se, 1e-9, 'mean se');
    const h = wasm.FittedOrdinal.fromInformation('logistic', vector(r.levels), vector(r.intercepts),
      vector(r.coefficients), f.information());
    expect(h.predictQuantileCi(xNew, 4, 0.5, 'interpolated', 0.95)).toEqual(
      f.predictQuantileCi(xNew, 4, 0.5, 'interpolated', 0.95));
  });

  it('continuation ratio', () => {
    const f = new wasm.ContinuationRatioRegressor().fit(vector(x), N, 2, vector(yOrd));
    const g = wasm.FittedContinuationRatio.fromParts(vector(f.getLevels()), 2, new Uint32Array(f.cohortSpecific()),
      vector(f.getCoefficients()), vector(f.vcov()));
    expect(Array.from(g.categoryProbabilities(xNew, 4))).toEqual(Array.from(f.categoryProbabilities(xNew, 4)));
  });

  it('P-spline', () => {
    const xs = Array.from({ length: 60 }, (_, i) => (6 * i) / 59);
    const ys = xs.map((v, i) => Math.sin(v) + 0.2 * (u[i] - 0.5));
    const m = new wasm.PSplineRegressor();
    m.setNBasis(12);
    const f = m.fit(vector(xs), vector(ys));
    const r = f.getResult();
    expect(Number.isFinite(r.aic)).toBe(true);
    const g = wasm.FittedPSpline.fromParts(vector(f.knots()), vector(f.coefficients()), vector(f.covariance()),
      f.getSigma2(), f.getEdf(), f.nObservations());
    const xn = vector([-0.5, 1, 3.3, 6.5]);
    expect(Array.from(g.predict(xn))).toEqual(Array.from(f.predict(xn)));
    allNear(g.predictInterval(xn, 'confidence', 0.95).se, f.predictInterval(xn, 'confidence', 0.95).se, 1e-9, 'se');
  });

  it('HC covariance matrix of OLS and WLS', () => {
    const y = Array.from({ length: N }, (_, i) => 1 + x[2 * i] - x[2 * i + 1] + (u[i] - 0.5) * (1 + Math.abs(x[2 * i])));
    const ols = new wasm.OlsRegressor().fit(vector(x), N, 2, vector(y));
    const v = ols.hcVcov(vector(x), N, 'hc3');
    const se = ols.hcStandardErrors(vector(x), N, 'hc3');
    expect(Math.abs(Math.sqrt(v[1][1]) - se.stdErrors[0])).toBeLessThan(1e-12);
    const w = new wasm.WlsRegressor();
    w.setWeights(vector(u.slice(0, N).map((t) => 0.5 + t)));
    const fw = w.fit(vector(x), N, 2, vector(y));
    expect(fw.hcVcov(vector(x), N, 'hc1').length).toBe(3);
  });

  it('areg', () => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < N; i++) {
      xs.push(x[2 * i], i % 3);
      ys.push(Math.exp(0.5 * x[2 * i] + 0.3 * (i % 3) + 0.2 * (u[i] - 0.5)));
    }
    const f = wasm.areg(vector(xs), N, 2, vector(ys), { xTypes: ['s', 'c'] });
    const r = f.getResult();
    const [lo, hi] = f.yRange();
    const g = wasm.Areg.fromParts(f.xBases(), vector(r.xCoefficients), f.yBasis(), vector(r.yCoefficients), lo, hi);
    const xn = vector([-1, 0, 0.5, 2, 1.5, 1]);
    expect(Array.from(g.predict(xn, 3))).toEqual(Array.from(f.predict(xn, 3)));
    const lp = f.predict(xn, 3);
    expect(g.inverseY(lp)).toEqual(f.inverseY(lp));
  });

  it('ModelSpec from stored limits and levels', () => {
    const s = new wasm.ModelSpec();
    s.linear('x', vector(x.filter((_, i) => i % 2 === 0)));
    s.categorical('g', vector(yOrd));
    const r = new wasm.ModelSpec();
    for (const p of s.predictors()) {
      r.addPredictor(p.name, p.kind, p.levels ? vector(p.levels) : undefined, p.limits, p.distinct ? vector(p.distinct) : undefined);
    }
    const beta = vector([0.1, 0.5, 0.2, -0.3, 0.4]);
    const vcov = vector(Array.from({ length: 25 }, (_, k) => (k % 6 === 0 ? 0.1 : 0.01)));
    expect(r.summary(beta, vcov, true)).toEqual(s.summary(beta, vcov, true));
    expect(r.contrast(beta, vcov, true, { g: 3 }, { g: 0 })).toEqual(s.contrast(beta, vcov, true, { g: 3 }, { g: 0 }));
    expect(() => s.contrast(beta, vcov, true, { g: 7 }, { g: 0 })).toThrow();
    const t = new wasm.ModelSpec();
    t.categoricalWithLevels('g', vector([2, 0, 1]));
    expect(Array.from(t.row({ g: 2 }, false))).toEqual([0, 1]);
  });
});
