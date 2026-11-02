// #287 phase 6: the methods and functions bound to close the last parity
// gaps (GLM residuals / predictions / vcov, Bayesian posteriors, PLS
// loadings, online RLS / passive-aggressive updates, LARS / LmDynamic
// internals, WLS from a state, small numerical helpers). Checked through
// identities with the existing API.
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

// a small deterministic design: n = 60, p = 2
const n = 60;
const x = new Float64Array(n * 2);
const yc = new Float64Array(n);
const counts = new Float64Array(n);
const binary = new Float64Array(n);
for (let i = 0; i < n; i++) {
  const a = Math.sin(i) * 1.5;
  const b = ((i * 7) % 11) / 5 - 1;
  x[2 * i] = a;
  x[2 * i + 1] = b;
  const eta = 0.3 + 0.5 * a - 0.4 * b;
  yc[i] = 1 + 2 * a - b + Math.cos(3 * i) * 0.5;
  counts[i] = Math.round(Math.exp(eta) + ((i * 3) % 4) - 1.5) < 0 ? 0 : Math.round(Math.exp(eta) + ((i * 3) % 4) - 1.5);
  binary[i] = eta + Math.cos(5 * i) > 0.3 ? 1 : 0;
}
const sum = (v: ArrayLike<number>) => Array.from(v).reduce((s, a) => s + a, 0);

describe('GLM extras', () => {
  it('Poisson / NB: residual types, predictions, vcov, predictWithSe', () => {
    for (const cls of ['PoissonRegressor', 'NegativeBinomialRegressor']) {
      const f = new wasm[cls]().fit(x, n, 2, counts);
      const dev = f.devianceResiduals();
      near(sum(dev.map((r: number) => r * r)), f.getDeviance(), 1e-8, `${cls} deviance`);
      expect(f.pearsonResiduals().length).toBe(n);
      expect(f.workingResiduals().length).toBe(n);
      allNear(f.predictCount(x, n), f.predict(x, n), 1e-12, `${cls} count`);
      allNear(f.predictLinear(x, n).map(Math.exp), f.predict(x, n), 1e-10, `${cls} link`);
      const v = f.vcov();
      expect(v.length).toBe(9);
      const p = f.predictWithSe(x, n, 'link', 'confidence', 0.9);
      allNear(p.fit, f.predictLinear(x, n), 1e-10, `${cls} withSe fit`);
      expect(p.lower[0]).toBeLessThan(p.fit[0]);
      expect(() => f.predictWithSe(x, n, 'odds')).toThrow(/type/);
    }
    const nb = new wasm.NegativeBinomialRegressor().fit(x, n, 2, counts);
    expect(Number.isFinite(nb.overdispersionRatio())).toBe(true);
  });

  it('binomial, tweedie, gamma, logistic', () => {
    const b = new wasm.BinomialRegressor().fit(x, n, 2, binary);
    allNear(b.predictProbability(x, n), b.predict(x, n), 1e-12, 'binomial prob');
    near(sum(b.devianceResiduals().map((r: number) => r * r)), b.getDeviance(), 1e-8, 'binomial deviance');
    expect(b.vcov().length).toBe(9);
    const yp = yc.map((v) => Math.abs(v) + 0.5);
    const t = new wasm.TweedieRegressor().fit(x, n, 2, yp);
    allNear(t.predictMu(x, n), t.predict(x, n), 1e-12, 'tweedie mu');
    expect(t.predictEta(x, n).length).toBe(n);
    expect(t.predictWithSe(x, n).fit.length).toBe(n);
    const g = new wasm.GammaRegressor().fit(x, n, 2, yp);
    allNear(g.predictMu(x, n), g.predict(x, n), 1e-12, 'gamma mu');
    expect(g.vcov().length).toBe(9);
    const l = new wasm.LogisticRegression();
    l.setComputeInference(true);
    expect(l.fit(x, n, 2, binary).vcov().length).toBe(9);
  });

  it('penalized GLM predictSeWithOffset, quantile and elastic-net extras', () => {
    const pg = new wasm.PenalizedGlmRegressor();
    pg.setFamily('poisson');
    pg.setComputeInference(true);
    const f = pg.fit(x, n, 2, counts);
    const a = f.predictSe(x, n, { scale: 'link' });
    const b = f.predictSeWithOffset(x, n, new Float64Array(n), { scale: 'link' });
    allNear(b.fit, a.fit, 1e-12, 'zero offset');
    const c = f.predictSeWithOffset(x, n, new Float64Array(n).fill(1), { scale: 'link' });
    allNear(c.fit, a.fit.map((v: number) => v + 1), 1e-12, 'unit offset');
    const q = new wasm.QuantileRegressor().fit(x, n, 2, yc);
    expect(q.checkLoss()).toBeGreaterThan(0);
    expect(q.pseudoRSquared()).toBeGreaterThan(0);
    expect(q.hasIntercept()).toBe(true);
    const en = new wasm.ElasticNetRegressor();
    en.setLambda(0.5);
    const ef = en.fit(x, n, 2, yc);
    expect(ef.nNonzero()).toBe(ef.getCoefficients().filter((v: number) => v !== 0).length);
  });
});

describe('model extras', () => {
  it('Bayesian ridge / ARD posteriors', () => {
    for (const cls of ['BayesianRidge', 'ArdRegression']) {
      const f = new wasm[cls]().fit(x, n, 2, yc);
      const sd = f.predictStd(x, n);
      expect(sd.every((v: number) => v > 0)).toBe(true);
      const s = f.sigma();
      allNear(f.sigmaDiag(), [s[0], s[s.length - 1]], 1e-12, `${cls} sigma diag`);
      const ci = f.credibleIntervals(0.9);
      expect(ci.lower[0]).toBeLessThan(ci.upper[0]);
      allNear(ci.sd, Array.from(f.sigmaDiag(), Math.sqrt), 1e-10, `${cls} sd`);
    }
    expect(Number.isFinite(new wasm.BayesianRidge().fit(x, n, 2, yc).logMarginalLikelihood())).toBe(true);
    expect(Array.from(new wasm.ArdRegression().fit(x, n, 2, yc).keptFeatures()).length).toBeGreaterThan(0);
  });

  it('PLS loadings, Huber weights, isotonic, LARS, ordinal counts, RANSAC, CR', () => {
    const pls = new wasm.PlsRegressor();
    pls.setNComponents(2);
    const pf = pls.fit(x, n, 2, yc);
    expect([pf.scores().length, pf.weights().length, pf.xLoadings().length, pf.yLoadings().length]).toEqual([n * 2, 4, 4, 2]);
    expect(pf.explainedVarianceRatio().length).toBe(2);
    const hw = new wasm.HuberRegressor().fit(x, n, 2, yc).weights();
    expect(hw.every((w: number) => w > 0 && w <= 1)).toBe(true);
    const xs = Array.from({ length: 20 }, (_, i) => i);
    const iso = new wasm.IsotonicRegressor().fit(new Float64Array(xs), new Float64Array(xs.map((v) => v + Math.sin(v))));
    near(iso.predictSingle(7.5), iso.predict1d([7.5])[0], 1e-15, 'isotonic');
    expect(iso.xThresholds().length).toBe(iso.yValues().length);
    const lars = new wasm.LarsRegressor().fit(x, n, 2, yc);
    expect(lars.method()).toBe('lar');
    expect(lars.coefsPath().length).toBeGreaterThan(0);
    const ord = new wasm.OrdinalRegressor().fit(x, n, 2, counts.map((v) => Math.min(v, 3)));
    near(sum(ord.counts()), n, 1e-12, 'ordinal counts');
    expect(ord.penaltyMatrix()).toBeUndefined();
    const ra = new wasm.RansacRegressor();
    ra.setRandomState(1n);
    expect(ra.fit(x, n, 2, yc).inlierFit().getCoefficients().length).toBe(2);
    const cr = new wasm.ContinuationRatioRegressor().fit(x, n, 2, counts.map((v) => Math.min(v, 3)));
    expect(cr.logistic().getCoefficients().length).toBeGreaterThan(2);
  });

  it('LmDynamic internals', () => {
    const f = new wasm.LmDynamicRegressor().fit(x, n, 2, yc);
    const w = f.modelWeights();
    const m = f.modelSpecs().length;
    expect(w.length).toBe(n * m);
    near(sum(w.slice(0, m)), 1, 1e-8, 'weights sum');
    expect(f.pointwiseIc().length).toBe(n * m);
    expect(f.coefficientsAt(0).length).toBeGreaterThan(0);
    expect(f.coefficientsAt(n + 5)).toBeUndefined();
  });
});

describe('online updates and states', () => {
  it('RLS update and passive-aggressive partialFit', () => {
    const rls = new wasm.RlsRegressor().fit(x.slice(0, 40), 20, 2, yc.slice(0, 20));
    const before = Array.from(rls.getCoefficients());
    for (let i = 20; i < n; i++) expect(Number.isFinite(rls.update(x.slice(2 * i, 2 * i + 2), yc[i]))).toBe(true);
    expect(Array.from(rls.getCoefficients())).not.toEqual(before);
    expect(rls.pMatrix().length).toBe(9);
    expect(() => rls.update([1], 1)).toThrow();
    const pa = new wasm.PassiveAggressiveRegressor();
    const state = new wasm.PaState(2);
    for (let i = 0; i < n; i++) pa.partialFit(state, x.slice(2 * i, 2 * i + 2), yc[i]);
    expect(state.weights.some((w: number) => w !== 0)).toBe(true);
  });

  it('WLS from a weighted state equals the WLS fit; OLS column checks', () => {
    const w = Array.from({ length: n }, (_, i) => 1 + (i % 3));
    const state = wasm.SufficientStats.fromData(x, n, 2, yc, w);
    const wls = new wasm.WlsRegressor();
    wls.setWeights(new Float64Array(w));
    allNear(wls.fitFromState(state).getCoefficients(), wls.fit(x, n, 2, yc).getCoefficients(), 1e-10, 'wls state');
    const xc = new Float64Array([1, 2, 1, 3, 1, 4]);
    expect(Array.from(wasm.OlsRegressor.detectConstantColumns(xc, 3, 2))).toEqual([1, 0]);
    expect(wasm.OlsRegressor.isFullRank(x, n, 2)).toBe(true);
  });
});

describe('ridge conditioning-aware λ (#300)', () => {
  it('uses the small λ on a well-conditioned design only', () => {
    const r = new wasm.RidgeRegressor();
    r.setLambda(5);
    expect(r.selectLambda(x, n, 2)).toBe(5);
    r.setConditioning();
    expect(wasm.gramConditionNumber(x, n, 2, true)).toBeLessThan(1000);
    expect(r.selectLambda(x, n, 2)).toBe(0.1);
    const f = r.fit(x, n, 2, yc);
    expect(f.getResult().lambda).toBe(0.1);
    // nearly collinear columns: κ above the threshold keeps the configured λ
    const xc = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) xc.set([x[2 * i], x[2 * i] + 1e-4 * Math.cos(i)], 2 * i);
    expect(r.selectLambda(xc, n, 2)).toBe(5);
    expect(r.fit(xc, n, 2, yc).getResult().lambda).toBe(5);
    r.setConditioning(0.5, 1e12);
    expect(r.selectLambda(xc, n, 2)).toBe(0.5);
    r.clearConditioning();
    expect(r.selectLambda(x, n, 2)).toBe(5);
  });
});

describe('helpers', () => {
  it('pretty, lowess, θ estimators, Gram condition number', () => {
    expect(Array.from(wasm.pretty(0, 1, 5))).toEqual([0, 0.2, 0.4, 0.6000000000000001, 0.8, 1]);
    const lw = wasm.lowess([3, 1, 2, 5, 4, 6, 8, 7], [2, 1, 2, 4, 3, 5, 7, 6], { frac: 0.8 });
    expect(lw.x).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(lw.fitted.length).toBe(8);
    const mu = Array.from({ length: n }, () => sum(counts) / n);
    const tm = wasm.estimateThetaMoments(Array.from(counts), mu);
    const tml = wasm.estimateThetaMl(Array.from(counts), mu);
    expect(tm).toBeGreaterThan(0);
    expect(tml).toBeGreaterThan(0);
    expect(() => wasm.estimateThetaMoments([1, 2], [1])).toThrow();
    expect(wasm.gramConditionNumber(x, n, 2, true)).toBeGreaterThanOrEqual(1);
    void R;
  });
});
