// #287: OrdinalRegressor and ContinuationRatioRegressor against rms::orm,
// MASS::polr and cr.setup + lrm (tests/r_validation_ordinal.rs,
// tests/r_validation_continuation_ratio.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear, near } from './rel';
import * as R from './references';

const N = 150;

describe('OrdinalRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  const x = vector(R.ORD_X);
  const xnew = vector(R.ORD_XNEW);

  function fit(y: number[], link: string, weights?: number[], offset?: number[]) {
    const m = new wasm.OrdinalRegressor();
    m.setLink(link);
    if (weights) m.setWeights(vector(weights));
    if (offset) m.setOffset(vector(offset));
    const f = m.fit(x, N, 3, vector(y));
    expect(f.getConverged()).toBe(true);
    return f;
  }

  function checkFit(id: string, f: any, coef: number[], ll: number[], vslopes: number[], vcov?: number[], tol = 1e-5) {
    const r = f.getResult();
    const m = r.intercepts.length;
    expect(m + 3).toBe(coef.length);
    allNear(r.intercepts, coef.slice(0, m), tol, `${id} intercepts`);
    allNear(r.coefficients, coef.slice(m), tol, `${id} slopes`);
    near(r.nullLogLikelihood, ll[ll.length - 2], 1e-9, `${id} null ll`);
    near(r.logLikelihood, ll[ll.length - 1], 1e-9, `${id} ll`);
    allNear(f.vcovSlopes(), vslopes, tol, `${id} vcov slopes`);
    if (vcov) allNear(f.vcov(), vcov, tol, `${id} vcov`);
    return r;
  }

  it('continuous response: logistic, probit and cloglog orm fits with means and quantiles', () => {
    const cases: [string, number[], number[], number[], number[], number[], number[], number[]][] = [
      ['logistic', R.ORD_ORM_LOGISTIC_COEF, R.ORD_ORM_LOGISTIC_LL, R.ORD_ORM_LOGISTIC_VCOV_SLOPES, R.ORD_ORM_LOGISTIC_MEAN,
        R.ORD_ORM_LOGISTIC_MEAN_SE, R.ORD_ORM_LOGISTIC_Q_INTERP, R.ORD_ORM_LOGISTIC_Q_DISCRETE],
      ['probit', R.ORD_ORM_PROBIT_COEF, R.ORD_ORM_PROBIT_LL, R.ORD_ORM_PROBIT_VCOV_SLOPES, R.ORD_ORM_PROBIT_MEAN,
        R.ORD_ORM_PROBIT_MEAN_SE, R.ORD_ORM_PROBIT_Q_INTERP, R.ORD_ORM_PROBIT_Q_DISCRETE],
      ['cloglog', R.ORD_ORM_CLOGLOG_COEF, R.ORD_ORM_CLOGLOG_LL, R.ORD_ORM_CLOGLOG_VCOV_SLOPES, R.ORD_ORM_CLOGLOG_MEAN,
        R.ORD_ORM_CLOGLOG_MEAN_SE, R.ORD_ORM_CLOGLOG_Q_INTERP, R.ORD_ORM_CLOGLOG_Q_DISCRETE],
    ];
    for (const [id, coef, ll, vs, mean, meanSe, qInterp, qDisc] of cases) {
      const f = fit(R.ORD_Y_CONT, id);
      expect(f.getLevels().length).toBe(N);
      const r = checkFit(id, f, coef, ll, vs);
      expect(r.link).toBe(id);
      const ms = f.predictMeanWithSe(xnew, 3);
      allNear(ms.fit, mean, 1e-5, `${id} mean`);
      allNear(ms.se, meanSe, 1e-5, `${id} mean se`);
      allNear(f.predictMean(xnew, 3), ms.fit, 1e-14, `${id} predictMean`);
      [0.25, 0.5, 0.9].forEach((q, k) =>
        allNear(f.predictQuantile(xnew, 3, q, 'interpolated'), qInterp.slice(3 * k, 3 * k + 3), 1e-5, `${id} Q${q}`),
      );
      [0.25, 0.5].forEach((q, k) =>
        allNear(f.predictQuantile(xnew, 3, q, 'discrete'), qDisc.slice(3 * k, 3 * k + 3), 1e-5, `${id} Qd${q}`),
      );
      const ci = f.predictQuantileCi(xnew, 3, 0.5, 'interpolated', 0.95);
      allNear(ci.estimate, qInterp.slice(3, 6), 1e-5, `${id} Q CI estimate`);
      for (let i = 0; i < 3; i++) {
        expect(ci.lower[i]).toBeLessThanOrEqual(ci.estimate[i]);
        expect(ci.upper[i]).toBeGreaterThanOrEqual(ci.estimate[i]);
      }
    }
  });

  it('five categories: logit (= polr), probit, weights and offset', () => {
    let f = fit(R.ORD_Y_CAT, 'logistic');
    checkFit('logit', f, R.ORD_CAT_COEF, R.ORD_CAT_LL, R.ORD_CAT_VCOV_SLOPES, R.ORD_CAT_VCOV);
    allNear(f.getIntercepts().map((a: number) => -a), R.ORD_POLR_ZETA, 1e-5, 'polr zeta');
    allNear(f.getCoefficients(), R.ORD_POLR_BETA, 1e-5, 'polr beta');
    f = fit(R.ORD_Y_CAT, 'probit');
    checkFit('probit', f, R.ORD_CAT_PROBIT_COEF, R.ORD_CAT_PROBIT_LL, R.ORD_CAT_PROBIT_VCOV_SLOPES, R.ORD_CAT_PROBIT_VCOV);
    f = fit(R.ORD_Y_CAT, 'logistic', R.ORD_W);
    checkFit('weights', f, R.ORD_CAT_W_COEF, R.ORD_CAT_W_LL, R.ORD_CAT_W_VCOV_SLOPES, R.ORD_CAT_W_VCOV);
    f = fit(R.ORD_Y_CAT, 'logistic', undefined, R.ORD_OFFSET);
    checkFit('offset', f, R.ORD_CAT_OFF_COEF, R.ORD_CAT_OFF_LL, R.ORD_CAT_OFF_VCOV_SLOPES, R.ORD_CAT_OFF_VCOV);
  });

  it('probabilities, LR test, scores and the proportional-odds score test are consistent', () => {
    const f = fit(R.ORD_Y_CAT, 'logistic');
    const p = f.categoryProbabilities(xnew, 3);
    const e = f.exceedanceProbabilities(xnew, 3);
    for (let i = 0; i < 3; i++) {
      const row = p.slice(5 * i, 5 * i + 5);
      near(row.reduce((s: number, v: number) => s + v, 0), 1, 1e-12, 'sum');
      for (let j = 0; j < 4; j++) {
        const tail = row.slice(j + 1).reduce((s: number, v: number) => s + v, 0);
        near(e[4 * i + j], tail, 1e-12, 'exceedance');
      }
    }
    const r = f.getResult();
    near(r.lrChisq, 2 * (R.ORD_CAT_LL[1] - R.ORD_CAT_LL[0]), 1e-8, 'LR');
    expect(r.lrDf).toBe(3);
    near(r.aic, -2 * R.ORD_CAT_LL[1] + 14, 1e-8, 'AIC');
    allNear(r.stdErrors, [0, 1, 2].map((j) => Math.sqrt(f.vcovSlopes()[4 * j])), 1e-15, 'se');
    allNear(f.effectiveDfDiagonal(), [1, 1, 1], 1e-10, 'edf');
    allNear(f.vcovSandwich(), f.vcov(), 1e-12, 'sandwich = vcov without penalty');
    // scores sum to ~0 at the MLE
    const s = f.scoreContributions(x, N, vector(R.ORD_Y_CAT));
    expect(s.length).toBe(N * 7);
    for (let j = 0; j < 7; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += s[7 * i + j];
      expect(Math.abs(sum)).toBeLessThan(1e-6);
    }
    const t = f.proportionalOddsScoreTest(x, N, vector(R.ORD_Y_CAT), new Uint32Array([]));
    expect(t.df).toBe(3 * 3);
    expect(t.pValue).toBeGreaterThanOrEqual(0);
    expect(t.pValue).toBeLessThanOrEqual(1);
    const lp = f.linearPredictor(xnew, 3);
    near(lp[0], [0, 1, 2].reduce((s, j) => s + R.ORD_XNEW[j] * f.getCoefficients()[j], 0), 1e-12, 'lp');
  });

  it('penalties shrink; xmeanOrdinaly; bad input throws', () => {
    const plain = fit(R.ORD_Y_CAT, 'logistic');
    const m = new wasm.OrdinalRegressor();
    m.setPenalty(50);
    const pen = m.fit(x, N, 3, vector(R.ORD_Y_CAT));
    expect(Math.abs(pen.getCoefficients()[0])).toBeLessThan(Math.abs(plain.getCoefficients()[0]));
    expect(pen.getResult().effectiveDf).toBeLessThan(plain.getResult().effectiveDf);
    m.setPenaltyMatrix(vector([50, 0, 0, 0, 50, 0, 0, 0, 50]));
    expect(m.fit(x, N, 3, vector(R.ORD_Y_CAT)).getCoefficients().length).toBe(3);
    const xm = wasm.xmeanOrdinaly(vector(R.ORD_X.filter((_, i) => i % 3 === 0)), vector(R.ORD_Y_CAT), true);
    expect(xm.levels).toEqual([1, 2, 3, 4, 5]);
    expect(xm.observed.length).toBe(5);
    expect(xm.expectedPo.length).toBe(5);
    expect(xm.expectedCr.length).toBe(5);
    expect(() => new wasm.OrdinalRegressor().setLink('logit')).toThrow();
    expect(() => plain.predictQuantile(xnew, 3, 0.5, 'nearest')).toThrow();
    expect(() => new wasm.OrdinalRegressor().fit(x, N, 3, vector(new Array(N).fill(1)))).toThrow();
    const capped = new wasm.OrdinalRegressor();
    capped.setMaxIterations(1);
    capped.setTolerance(1e-14);
    const c = capped.fit(x, N, 3, vector(R.ORD_Y_CONT));
    expect(c.getConverged()).toBe(false);
    expect(c.getIterations()).toBe(1);
  });
});

describe('ContinuationRatioRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  const x = vector(R.CR_X);
  const y = vector(R.CR_Y_CAT);

  it('crSetup matches rms::cr.setup', () => {
    const s = wasm.crSetup(vector(R.CR_SMALL_Y));
    expect(s.levels).toEqual([1, 2, 3, 4]);
    expect(s.y).toEqual(R.CR_SMALL_CR_Y);
    expect(s.cohort).toEqual(R.CR_SMALL_COHORT);
    expect(s.subs).toEqual(R.CR_SMALL_SUBS);
    expect(s.reps).toEqual(R.CR_SMALL_REPS);
  });

  it('matches lrm on cr.setup data, equal and cohort-specific slopes', () => {
    const cases: [string, number[], number[], number[], number[], number[]][] = [
      ['equal slopes', [], R.CR_EQUAL_COEF, R.CR_EQUAL_VCOV, R.CR_EQUAL_LL, R.CR_EQUAL_COND],
      ['cohort-specific x1', [0], R.CR_X1_COEF, R.CR_X1_VCOV, R.CR_X1_LL, R.CR_X1_COND],
    ];
    for (const [id, specific, coef, vcov, ll, cond] of cases) {
      const m = new wasm.ContinuationRatioRegressor();
      m.setCohortSpecific(new Uint32Array(specific));
      const f = m.fit(x, N, 3, y);
      expect(f.getConverged()).toBe(true);
      allNear(f.getCoefficients(), coef, 1e-7, `${id} coef`);
      allNear(f.vcov(), vcov, 1e-6, `${id} vcov`);
      near(f.getLogLikelihood(), ll[0], 1e-9, `${id} ll`);
      const r = f.getResult();
      allNear(r.stdErrors, coef.map((_, j) => Math.sqrt(vcov[j * coef.length + j])), 1e-6, `${id} se`);
      expect(r.levels).toEqual([1, 2, 3, 4, 5]);
      const h = f.conditionalProbabilities(vector(R.CR_XNEW), 3);
      allNear(h, cond, 1e-6, `${id} conditional`);
      const p = f.categoryProbabilities(vector(R.CR_XNEW), 3);
      for (let i = 0; i < 3; i++) {
        near(p.slice(5 * i, 5 * i + 5).reduce((s: number, v: number) => s + v, 0), 1, 1e-12, 'sum');
        near(p[5 * i], h[4 * i], 1e-15, 'P(Y = y1) = h1');
      }
    }
  });

  it('case weights match weighted lrm; bad input throws', () => {
    const m = new wasm.ContinuationRatioRegressor();
    m.setWeights(vector(R.CR_W));
    const f = m.fit(x, N, 3, y);
    const k = R.CR_WEQUAL_COEF.length;
    allNear(f.getCoefficients(), R.CR_WEQUAL_COEF, 1e-9, 'coef');
    // the R matrix is column-major
    const v = f.vcov();
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) near(v[a * k + b], R.CR_WEQUAL_VCOV[b * k + a], 1e-6, 'vcov');
    near(f.getLogLikelihood(), R.CR_WEQUAL_LL[0], 1e-10, 'll');
    m.setWeights(vector(R.CR_W.slice(0, 10)));
    expect(() => m.fit(x, N, 3, y)).toThrow();
    const bad = new wasm.ContinuationRatioRegressor();
    bad.setCohortSpecific(new Uint32Array([5]));
    expect(() => bad.fit(x, N, 3, y)).toThrow();
    expect(() => new wasm.ContinuationRatioRegressor().fit(x, N, 3, vector(R.CR_Y_CAT.map((v) => (v > 2 ? 1 : 0))))).toThrow();
  });
});
