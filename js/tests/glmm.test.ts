// #287: GlmmRegressor against lme4 (lmer, glmer(nAGQ = 0)) and lmerTest
// (tests/r_validation_glmm.rs, tests/issue89_glmm_inference.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear } from './rel';
import * as R from './references';

const LMM_TOL = 1e-5;
const GLMM_TOL = 1e-6;

/** |a − b| < tol (absolute, as the Rust lme4 tests). */
function abs(a: number, b: number, tol: number, what: string) {
  if (!(Math.abs(a - b) < tol)) throw new Error(`${what}: ${a} vs ${b} (tol ${tol})`);
}

/** |a − b| ≤ tol · max(|b|, 1) elementwise. */
function relAll(a: ArrayLike<number>, b: ArrayLike<number>, tol: number, what: string) {
  expect(a.length, what).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    if (!(Math.abs(a[i] - b[i]) <= tol * Math.max(Math.abs(b[i]), 1))) {
      throw new Error(`${what}[${i}]: ${a[i]} vs ${b[i]}`);
    }
  }
}

/** Reorder fit rows (first-seen group order) to R's sorted level order. */
function byLevel(rows: number[][], ids: number[]): number[] {
  const order: number[] = [];
  for (const g of ids) if (!order.includes(g)) order.push(g);
  const out: number[] = new Array(rows.length * rows[0].length);
  const q = rows[0].length;
  order.forEach((level, c) => rows[c].forEach((v, a) => (out[level * q + a] = v)));
  return out;
}

describe('GlmmRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('random-intercept LMM matches lmer and lmerTest', () => {
    const m = new wasm.GlmmRegressor('gaussian');
    m.setTolerance(1e-10);
    m.setSatterthwaite(true);
    const x = vector(R.GLMM_X_LMM_A);
    const n = R.GLMM_X_LMM_A.length;
    const f = m.fit(x, n, 1, vector(R.GLMM_Y_LMM_A), new Uint32Array(R.GLMM_G_LMM_A));
    expect(f.getConverged()).toBe(true);
    const r = f.getResult();
    abs(r.intercept, R.GLMM_LMM_A_INTERCEPT, LMM_TOL, 'intercept');
    abs(r.fixedEffects[1], R.GLMM_LMM_A_SLOPE, LMM_TOL, 'slope');
    abs(r.sigma, R.GLMM_LMM_A_SIGMA, LMM_TOL, 'sigma');
    abs(r.sdRandom, R.GLMM_LMM_A_SD_RANDOM, LMM_TOL, 'sd random');
    abs(f.getLogLikelihood(), R.GLMM_LMM_A_LOGLIK, 1e-2, 'logLik');
    abs(r.stdErrors[0], R.GLMM_LMM_A_SE_INTERCEPT, LMM_TOL, 'se b0');
    abs(r.stdErrors[1], R.GLMM_LMM_A_SE_SLOPE, LMM_TOL, 'se b1');
    R.GLMM_LMM_A_RANEF.forEach((e, j) => abs(f.getRandomEffects()[j], e, 5e-3, `ranef ${j}`));
    f.predictFixed(x, n).forEach((p: number, i: number) => abs(p, R.GLMM_LMM_A_PREDFIXED[i], LMM_TOL, `pred ${i}`));
    expect(r.isReml).toBe(true);

    relAll(r.vcov.flat(), R.GLMM_LMM_A_VCOV, 1e-6, 'vcov');
    relAll(r.satterthwaiteDf, R.GLMM_LMM_A_SATT_DF, 1e-5, 'satterthwaite df');
    const inf = f.inference('satterthwaite', 0.95);
    relAll(inf.df, R.GLMM_LMM_A_SATT_DF, 1e-5, 'df');
    relAll(inf.pValues, R.GLMM_LMM_A_SATT_P, 1e-6, 'p');
    relAll(inf.confLow, R.GLMM_LMM_A_SATT_LO95, 1e-7, 'lo95');
    relAll(inf.confHigh, R.GLMM_LMM_A_SATT_HI95, 1e-7, 'hi95');
    relAll(byLevel(r.randomEffectsSe, R.GLMM_G_LMM_A), R.GLMM_LMM_A_RE_SE_G, 1e-6, 're se');
    const ci = f.randomEffectsConfInt(0.95);
    relAll(byLevel(ci.lower, R.GLMM_G_LMM_A), R.GLMM_LMM_A_RANVALS_LO95_G, 1e-6, 're lo95');
    relAll(byLevel(ci.upper, R.GLMM_G_LMM_A), R.GLMM_LMM_A_RANVALS_HI95_G, 1e-6, 're hi95');
    relAll(r.fittedValues, R.GLMM_LMM_A_FITTED, 1e-6, 'fitted');
    relAll(r.fittedPopulation, R.GLMM_LMM_A_FITTED_POP, 1e-6, 'fitted pop');
    relAll(r.pearsonResiduals, R.GLMM_LMM_A_PEARSON, 1e-6, 'pearson');
    expect(r.dfResidual).toBe(R.GLMM_LMM_A_DF_RESID);
    const wald = f.confInt(0.95);
    expect(wald.lower[1]).toBeLessThan(r.fixedEffects[1]);
    expect(wald.upper[1]).toBeGreaterThan(r.fixedEffects[1]);
  });

  it('ML and a no-intercept LMM match lmer', () => {
    let m = new wasm.GlmmRegressor();
    m.setReml(false);
    const n = R.GLMM_X_ML_H.length;
    let f = m.fit(vector(R.GLMM_X_ML_H), n, 1, vector(R.GLMM_Y_ML_H), new Uint32Array(R.GLMM_G_ML_H));
    let r = f.getResult();
    expect(r.isReml).toBe(false);
    abs(r.intercept, R.GLMM_ML_H_INTERCEPT, LMM_TOL, 'ML intercept');
    abs(r.fixedEffects[1], R.GLMM_ML_H_SLOPE, LMM_TOL, 'ML slope');
    abs(r.sigma, R.GLMM_ML_H_SIGMA, LMM_TOL, 'ML sigma');
    abs(r.sdRandom, R.GLMM_ML_H_SD_RANDOM, LMM_TOL, 'ML sd');
    m = new wasm.GlmmRegressor('gaussian');
    m.setWithIntercept(false);
    const k = R.GLMM_X_NOINT_K.length;
    f = m.fit(vector(R.GLMM_X_NOINT_K), k, 1, vector(R.GLMM_Y_NOINT_K), new Uint32Array(R.GLMM_G_NOINT_K));
    r = f.getResult();
    expect(r.intercept).toBeUndefined();
    abs(r.fixedEffects[0], R.GLMM_NOINT_K_SLOPE, LMM_TOL, 'no-int slope');
    abs(r.sigma, R.GLMM_NOINT_K_SIGMA, LMM_TOL, 'no-int sigma');
  });

  it('random slopes match lmer', () => {
    const m = new wasm.GlmmRegressor('gaussian');
    m.setRandomSlopes(new Uint32Array([0]));
    m.setTolerance(1e-9);
    const n = R.GLMM_X_SLOPE_E.length;
    const f = m.fit(vector(R.GLMM_X_SLOPE_E), n, 1, vector(R.GLMM_Y_SLOPE_E), new Uint32Array(R.GLMM_G_SLOPE_E));
    const r = f.getResult();
    expect(r.randomCov.length).toBe(2);
    abs(r.intercept, R.GLMM_SLOPE_E_INTERCEPT, LMM_TOL, 'intercept');
    abs(r.fixedEffects[1], R.GLMM_SLOPE_E_SLOPE, LMM_TOL, 'slope');
    abs(r.randomSd[0], R.GLMM_SLOPE_E_SD_INTERCEPT, LMM_TOL, 'sd intercept');
    abs(r.randomSd[1], R.GLMM_SLOPE_E_SD_SLOPE, LMM_TOL, 'sd slope');
    abs(r.randomCorr[0][1], R.GLMM_SLOPE_E_CORR, 5e-3, 'corr');
    R.GLMM_SLOPE_E_RANEF_INT.forEach((e, j) => abs(r.randomEffects[j][0], e, 5e-3, `ranef int ${j}`));
    R.GLMM_SLOPE_E_RANEF_SLOPE.forEach((e, j) => abs(r.randomEffects[j][1], e, 5e-3, `ranef slope ${j}`));
  });

  it('crossed and nested factors match lmer', () => {
    const m = new wasm.GlmmRegressor('gaussian');
    m.setTolerance(1e-10);
    let n = R.GLMM_X_CROSS_F.length;
    // plain arrays and typed arrays are both accepted
    let f = m.fitCrossed(vector(R.GLMM_X_CROSS_F), n, 1, vector(R.GLMM_Y_CROSS_F), [
      R.GLMM_A_CROSS_F,
      new Uint32Array(R.GLMM_B_CROSS_F),
    ]);
    let r = f.getResult();
    expect(r.nFactors).toBe(2);
    abs(r.intercept, R.GLMM_CROSS_F_INTERCEPT, LMM_TOL, 'intercept');
    abs(r.factors[0].sd, R.GLMM_CROSS_F_SD_A, LMM_TOL, 'sd a');
    abs(r.factors[1].sd, R.GLMM_CROSS_F_SD_B, LMM_TOL, 'sd b');
    abs(r.stdErrors[1], R.GLMM_CROSS_F_SE_SLOPE, LMM_TOL, 'se slope');
    R.GLMM_CROSS_F_RANEF_A.forEach((e, l) => abs(r.factors[0].blups[l], e, 5e-3, `ranef a ${l}`));
    R.GLMM_CROSS_F_RANEF_B.forEach((e, l) => abs(r.factors[1].blups[l], e, 5e-3, `ranef b ${l}`));
    relAll(byLevel(f.factorRandomEffectsSe(0), R.GLMM_A_CROSS_F), R.GLMM_CROSS_F_RE_SE_A, 1e-6, 're se a');
    relAll(byLevel(f.factorRandomEffectsSe(1), R.GLMM_B_CROSS_F), R.GLMM_CROSS_F_RE_SE_B, 1e-6, 're se b');
    const ci = f.factorRandomEffectsConfInt(1, 0.95);
    relAll(byLevel(ci.lower, R.GLMM_B_CROSS_F), R.GLMM_CROSS_F_RANVALS_LO95_B, 1e-6, 're lo95 b');
    expect(f.factorRandomSd(0)[0]).toBeCloseTo(r.factors[0].sd, 12);
    expect(f.factorRandomCov(0)[0][0]).toBeCloseTo(r.factors[0].sd ** 2, 12);
    expect(f.factorRandomEffects(0).length).toBe(R.GLMM_CROSS_F_RANEF_A.length);
    expect(() => f.factorRandomSd(5)).toThrow();

    n = R.GLMM_X_NEST_G.length;
    m.setTolerance(1e-9);
    f = m.fitCrossed(vector(R.GLMM_X_NEST_G), n, 1, vector(R.GLMM_Y_NEST_G), [R.GLMM_A_NEST_G, R.GLMM_AB_NEST_G]);
    r = f.getResult();
    abs(r.intercept, R.GLMM_NEST_G_INTERCEPT, LMM_TOL, 'nested intercept');
    abs(r.factors[0].sd, R.GLMM_NEST_G_SD_A, LMM_TOL, 'sd a');
    abs(r.factors[1].sd, R.GLMM_NEST_G_SD_AB, LMM_TOL, 'sd ab');
  });

  it('Poisson and binomial GLMMs match glmer(nAGQ = 0)', () => {
    let m = new wasm.GlmmRegressor('poisson');
    m.setTolerance(1e-10);
    let n = R.GLMM_X_POIS_C.length;
    let f = m.fit(vector(R.GLMM_X_POIS_C), n, 1, vector(R.GLMM_Y_POIS_C), new Uint32Array(R.GLMM_G_POIS_C));
    expect(f.getConverged()).toBe(true);
    let r = f.getResult();
    abs(r.intercept, R.GLMM_POIS_C_INTERCEPT, GLMM_TOL, 'pois intercept');
    abs(r.fixedEffects[1], R.GLMM_POIS_C_SLOPE, GLMM_TOL, 'pois slope');
    abs(r.sdRandom, R.GLMM_POIS_C_SD_RANDOM, GLMM_TOL, 'pois sd');
    abs(r.stdErrors[0], R.GLMM_POIS_C_SE_INTERCEPT, GLMM_TOL, 'pois se b0');
    expect(r.satterthwaiteDf).toBeUndefined();
    expect(() => f.inference('satterthwaite', 0.95)).toThrow();
    const wald = f.inference('wald', 0.95);
    expect(wald.df).toBeUndefined();
    relAll(r.vcov.flat(), R.GLMM_POIS_C_VCOV, 1e-6, 'pois vcov');
    relAll(r.fittedValues, R.GLMM_POIS_C_FITTED, 1e-6, 'pois fitted');
    relAll(r.pearsonResiduals, R.GLMM_POIS_C_PEARSON, 1e-6, 'pois pearson');
    // the linear predictor; the population fit is exp(eta) for the log link
    const eta = f.predictFixedWithOffset(vector(R.GLMM_X_POIS_C), n, vector(new Array(n).fill(0.5)));
    allNear(eta.map((e: number) => Math.exp(e - 0.5)), r.fittedPopulation, 1e-10, 'pop');

    m = new wasm.GlmmRegressor('binomial');
    n = R.GLMM_X_BIN_D.length;
    f = m.fit(vector(R.GLMM_X_BIN_D), n, 1, vector(R.GLMM_Y_BIN_D), new Uint32Array(R.GLMM_G_BIN_D));
    r = f.getResult();
    abs(r.intercept, R.GLMM_BIN_D_INTERCEPT, GLMM_TOL, 'bin intercept');
    abs(r.fixedEffects[1], R.GLMM_BIN_D_SLOPE, GLMM_TOL, 'bin slope');
    abs(r.sdRandom, R.GLMM_BIN_D_SD_RANDOM, GLMM_TOL, 'bin sd');
  });

  it('other families fit; bad options and input throw', () => {
    const n = R.GLMM_X_POIS_C.length;
    const x = vector(R.GLMM_X_POIS_C);
    const g = new Uint32Array(R.GLMM_G_POIS_C);
    const y = vector(R.GLMM_Y_POIS_C);
    const nb = new wasm.GlmmRegressor('negative_binomial');
    expect(nb.fit(x, n, 1, y, g).getResult().nbTheta).toBeGreaterThan(0);
    nb.setNbTheta(3);
    expect(nb.fit(x, n, 1, y, g).getResult().nbTheta).toBe(3);
    const pos = vector(R.GLMM_Y_POIS_C.map((v) => v + 0.5));
    expect(new wasm.GlmmRegressor('gamma').fit(x, n, 1, pos, g).getFixedEffects().length).toBe(2);
    expect(new wasm.GlmmRegressor('tweedie', 1.5).fit(x, n, 1, y, g).getFixedEffects().length).toBe(2);
    const agq = new wasm.GlmmRegressor('poisson');
    agq.setNAgq(5);
    expect(agq.fit(x, n, 1, y, g).getConverged()).toBe(true);
    expect(() => new wasm.GlmmRegressor('weibull')).toThrow();
    expect(() => new wasm.GlmmRegressor('tweedie')).toThrow();
    expect(() => new wasm.GlmmRegressor().fit(x, n, 1, y, new Uint32Array(n - 1))).toThrow();
    expect(() => new wasm.GlmmRegressor().fitCrossed(x, n, 1, y, [R.GLMM_G_POIS_C.slice(1)])).toThrow();
    const f = new wasm.GlmmRegressor().fit(x, n, 1, y, g);
    expect(() => f.inference('kenward-roger', 0.95)).toThrow();
  });
});
