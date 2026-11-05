// #287 phase 2a: diagnostics through WASM against the R references of the
// Rust validation (accuracy, heteroskedasticity, LOO / PRESS, GLM
// residuals, (G)VIF, augment of a GLM).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear, near } from './rel';
import * as R from './references';

/** Row-major flat matrix from columns. */
function design(cols: number[][]): { x: Float64Array; n: number; p: number } {
  const n = cols[0].length;
  const p = cols.length;
  const x = new Float64Array(n * p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) x[i * p + j] = cols[j][i];
  return { x, n, p };
}

/** |a − b| ≤ tol · max(|b|, 1), NaN matching NaN. */
function relAll(a: ArrayLike<number>, b: ArrayLike<number>, tol: number, what: string) {
  expect(a.length, what).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(b[i])) {
      expect(Number.isNaN(a[i]), `${what}[${i}]`).toBe(true);
    } else if (!(Math.abs(a[i] - b[i]) <= tol * Math.max(Math.abs(b[i]), 1))) {
      throw new Error(`${what}[${i}]: ${a[i]} vs ${b[i]}`);
    }
  }
}

describe('diagnostics', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('concordance, somers2, lrmStats, rcorrCens, r2Measures and valProb match Hmisc / rms', () => {
    for (const [w, want] of [[undefined, R.ACC_CONC], [vector(R.ACC_W), R.ACC_CONC_W]] as const) {
      const c = wasm.concordance(vector(R.ACC_XC), vector(R.ACC_YC), w);
      allNear([c.concordant, c.discordant, c.tiedX, c.tiedY, c.tiedXy, c.c], want, 1e-12, 'concordance');
      near(c.dxy, 2 * c.c - 1, 1e-12, 'dxy');
    }
    const p = R.ACC_LP.map((v) => 1 / (1 + Math.exp(-v)));
    const s = wasm.somers2(vector(p), vector(R.ACC_Y));
    allNear([s.c, s.dxy], R.ACC_SOMERS2, 1e-12, 'somers2');
    const sw = wasm.somers2(vector(p), vector(R.ACC_Y), vector(R.ACC_W));
    allNear([sw.c, sw.dxy], R.ACC_SOMERS2_W, 1e-12, 'somers2 weighted');

    const l = wasm.lrmStats(vector(R.ACC_Y), vector(R.ACC_LP), undefined, 2);
    near(l.modelLr, R.ACC_LRM_MODEL_L_R_, 1e-9, 'LR');
    expect(Math.abs(l.pValue - R.ACC_LRM_P)).toBeLessThan(1e-15);
    allNear([l.c, l.dxy, l.gamma, l.tauA, l.brier, l.g, l.gr, l.gp],
      [R.ACC_LRM_C, R.ACC_LRM_DXY, R.ACC_LRM_GAMMA, R.ACC_LRM_TAU_A, R.ACC_LRM_BRIER, R.ACC_LRM_G, R.ACC_LRM_GR, R.ACC_LRM_GP], 1e-12, 'lrm');
    near(l.r2, R.ACC_LRM_R2, 1e-9, 'R2');
    const m = l.r2Measures;
    allNear([m.r2, m.r2Adj, m.r2Ess, m.r2AdjEss], R.ACC_LRM_R2M, 1e-9, 'R2m');
    near(wasm.giniMd(vector(R.ACC_LP)), R.ACC_GINI_LP, 1e-12, 'GiniMd');
    const lw = wasm.lrmStats(vector(R.ACC_Y), vector(R.ACC_LPW), vector(R.ACC_W), 2);
    near(lw.modelLr, R.ACC_LRMW_MODEL_L_R_, 1e-9, 'LR w');
    near(lw.c, R.ACC_LRMW_C, 1e-12, 'C w');
    const pw = R.ACC_LPW.map((v) => 1 / (1 + Math.exp(-v)));
    near(wasm.brierScore(vector(R.ACC_Y), vector(pw), vector(R.ACC_W)), R.ACC_LRMW_BRIER, 1e-12, 'brier');

    const ev = new Uint8Array(R.ACC_EV.map((e) => (e === 1 ? 1 : 0)));
    for (const [event, want] of [[ev, R.ACC_RCORR], [new Uint8Array(R.ACC_TT.length).fill(1), R.ACC_RCORR_UNC]] as const) {
      const r = wasm.rcorrCens(vector(R.ACC_PX), vector(R.ACC_TT), event);
      allNear([r.c, r.dxy, r.sd, r.n, r.missing, r.uncensored, r.relevantPairs, r.concordant, r.uncertain], want, 1e-12, 'rcorr.cens');
    }

    let r2 = wasm.r2Measures(25.3, 4, 200, 60);
    allNear([r2.r2, r2.r2Adj, r2.r2Ess, r2.r2AdjEss], R.ACC_R2M_ESS, 1e-14, 'R2 ess');
    r2 = wasm.r2Measures(25.3, 4, 200, wasm.effectiveSampleSize(200, vector([60, 140])));
    allNear([r2.r2, r2.r2Adj, r2.r2Ess, r2.r2AdjEss], R.ACC_R2M_FREQ, 1e-14, 'R2 freq');
    r2 = wasm.r2Measures(25.3, 4, 200);
    allNear([r2.r2, r2.r2Adj], R.ACC_R2M_NONE, 1e-14, 'R2 none');
    expect(r2.r2Ess).toBeUndefined();
    expect(wasm.nagelkerkeR2(10, 100, 50)).toBeGreaterThan(0);

    const v = wasm.valProb(vector(R.ACC_PV), vector(R.ACC_Y));
    const got = [v.dxy, v.c, v.r2, v.d, v.dChisq, v.dP, v.u, v.uChisq, v.uP, v.q, v.brier, v.intercept, v.slope,
      v.emax, v.e90, v.eavg, v.sZ, v.sP];
    allNear(got, R.ACC_VALPROB, 1e-6, 'val.prob');
    expect(() => wasm.somers2(vector([0.1, 0.2]), vector([1]))).toThrow();
  });

  it('White and Breusch–Pagan match lmtest', () => {
    const { x, n } = design([R.HET_X1, R.HET_X2, R.HET_X3]);
    const fit = new wasm.OlsRegressor().fit(x, n, 3, vector(R.HET_Y));
    const e = vector(fit.getResult().residuals);
    const z = design([R.HET_Z1, R.HET_Z2]).x;
    const cases: [any, number[]][] = [
      [wasm.whiteTest(e, x, 3), R.HET_WHITE3],
      [wasm.breuschPagan(e, z, 2, true), R.HET_BPZ_KOENKER],
      [wasm.breuschPagan(e, z, 2, false), R.HET_BPZ_CLASSIC],
    ];
    for (const [t, want] of cases) {
      near(t.statistic, want[0], 1e-9, 'statistic');
      expect(t.df).toBe(want[1]);
      expect(Math.abs(t.pValue - want[2])).toBeLessThan(1e-10);
      expect(typeof t.method).toBe('string');
    }
  });

  it('augment / press on OLS and WLS match R leave-one-out', () => {
    const { x, n } = design([R.LOO_MT_WT, R.LOO_MT_HP]);
    const y = vector(R.LOO_MT_MPG);
    const ols = new wasm.OlsRegressor().fit(x, n, 2, y);
    const a = ols.augment(x, n, y);
    relAll(a.looResidual, R.LOO_OLS_LOO_RESID, 1e-8, 'OLS loo resid');
    relAll(a.looFitted, R.LOO_OLS_LOO_FITTED, 1e-8, 'OLS loo fitted');
    expect(a.residType).toBe('response');
    expect(a.leverage.length).toBe(n);
    const p = ols.press(x, n, y);
    expect(p.n).toBe(32);
    allNear([p.press, p.pressRmse, p.looRSquared], [R.LOO_OLS_PRESS, R.LOO_OLS_PRESS_RMSE, R.LOO_OLS_LOO_R2], 1e-8, 'press');

    const m0 = new wasm.OlsRegressor();
    m0.setWithIntercept(false);
    const p0 = m0.fit(x, n, 2, y).press(x, n, y);
    allNear([p0.press, p0.pressRmse, p0.looRSquared], [R.LOO_OLS0_PRESS, R.LOO_OLS0_PRESS_RMSE, R.LOO_OLS0_LOO_R2], 1e-8, 'press0');

    const wm = new wasm.WlsRegressor();
    wm.setWeights(vector(R.LOO_MT_W));
    const wf = wm.fit(x, n, 2, y);
    relAll(wf.augment(x, n, y).looResidual, R.LOO_WLS_LOO_RESID, 1e-8, 'WLS loo resid');
    const pw = wf.press(x, n, y);
    allNear([pw.press, pw.pressRmse, pw.looRSquared], [R.LOO_WLS_PRESS, R.LOO_WLS_PRESS_RMSE, R.LOO_WLS_LOO_R2], 1e-8, 'WLS press');

    // one row with leverage 1 has no leave-one-out prediction
    const h = design([R.LOO_H1_X, R.LOO_H1_D]);
    const hf = new wasm.OlsRegressor().fit(h.x, h.n, 2, vector(R.LOO_H1_Y));
    const ha = hf.augment(h.x, h.n, vector(R.LOO_H1_Y));
    relAll(ha.looResidual.slice(0, 7), R.LOO_H1_LOO_RESID.slice(0, 7), 1e-8, 'H1');
    expect(Number.isNaN(ha.looResidual[7])).toBe(true);

    // the free functions reproduce augment's columns
    const r = ols.getResult();
    const lev = wasm.computeLeverage(x, n, 2, true);
    allNear(lev, a.leverage, 1e-10, 'leverage');
    const mse = r.residuals.reduce((s: number, v: number) => s + v * v, 0) / (n - 3);
    allNear(wasm.cooksDistance(vector(r.residuals), lev, mse, 3), a.cooksD, 1e-8, 'cooks');
    allNear(wasm.studentizedResiduals(vector(r.residuals), lev, mse), a.stdResidual, 1e-8, 'rstandard');
    allNear(wasm.externallyStudentizedResiduals(vector(r.residuals), lev, mse, 3), a.studResidual, 1e-8, 'rstudent');
    allNear(wasm.dffits(vector(a.studResidual), lev, mse, 3).length ? [1] : [0], [1], 0, 'dffits runs');
    expect(Array.from(wasm.influentialCooks(vector(a.cooksD))).every((i: number) => a.cooksD[i] > 4 / n)).toBe(true);
    expect(wasm.highLeveragePoints(lev, 3).length).toBeGreaterThanOrEqual(0);
    expect(ols.partialResiduals(x, n, y).length).toBe(2);
    // ridge leave-one-out with lambda 0 equals OLS
    const rg = new wasm.RidgeRegressor();
    rg.setLambda(0);
    const loo = rg.fit(x, n, 2, y).loo(x, n, y);
    relAll(loo.looResidual, R.LOO_OLS_LOO_RESID, 1e-8, 'ridge loo');
    near(loo.press.press, R.LOO_OLS_PRESS, 1e-8, 'ridge press');
  });

  it('GLM residual functions match R', () => {
    const n = R.RES_X.length;
    const x = vector(R.RES_X);
    const pm = new wasm.PoissonRegressor();
    pm.setTolerance(1e-12);
    const mu = vector(pm.fit(x, n, 1, vector(R.RES_YC)).getResult().fittedValues);
    const yc = vector(R.RES_YC);
    relAll(wasm.pearsonResiduals(yc, mu, 'poisson'), R.RES_POIS_PEARSON, 1e-7, 'pois pearson');
    relAll(wasm.devianceResiduals(yc, mu, 'poisson', 'log'), R.RES_POIS_DEVIANCE, 1e-7, 'pois deviance');
    relAll(wasm.workingResiduals(yc, mu, 'poisson'), R.RES_POIS_WORKING, 1e-7, 'pois working');
    const pr = wasm.pearsonResiduals(yc, mu, 'poisson');
    near(wasm.pearsonChiSquared(yc, mu, 'poisson'), pr.reduce((s: number, v: number) => s + v * v, 0), 1e-12, 'chi2');
    near(wasm.estimateDispersionPearson(yc, mu, 2, 'poisson'), wasm.pearsonChiSquared(yc, mu, 'poisson') / (n - 2), 1e-12, 'disp');
    expect(wasm.estimateDispersionDeviance(yc, mu, 2, 'poisson')).toBeGreaterThan(0);
    allNear(wasm.responseResiduals(yc, mu), R.RES_YC.map((v, i) => v - mu[i]), 1e-14, 'response');
    const zeros = vector(new Array(n).fill(0));
    allNear(wasm.standardizedPearsonResiduals(yc, mu, zeros, 1, 'poisson'), pr, 1e-12, 'std pearson, h = 0');
    allNear(wasm.standardizedDevianceResiduals(yc, mu, zeros, 1, 'poisson'), wasm.devianceResiduals(yc, mu, 'poisson'), 1e-12, 'std deviance');

    const gm = wasm.TweedieRegressor.gamma();
    gm.setTolerance(1e-12);
    const gmu = vector(gm.fit(x, n, 1, vector(R.RES_YG)).getResult().fittedValues);
    const yg = vector(R.RES_YG);
    relAll(wasm.pearsonResiduals(yg, gmu, 'gamma'), R.RES_GAMMA_PEARSON, 1e-7, 'gamma pearson');
    relAll(wasm.devianceResiduals(yg, gmu, 'gamma'), R.RES_GAMMA_DEVIANCE, 1e-7, 'gamma deviance');
    relAll(wasm.workingResiduals(yg, gmu, 'tweedie', undefined, 2), R.RES_GAMMA_WORKING, 1e-7, 'tweedie p = 2 working');
    expect(() => wasm.pearsonResiduals(yc, mu, 'gaussian')).toThrow();
    expect(() => wasm.pearsonResiduals(yc, mu, 'negative_binomial')).toThrow();
    expect(() => wasm.pearsonResiduals(yc, vector([1]), 'poisson')).toThrow();
    expect(wasm.pearsonResiduals(yc, mu, 'negative_binomial', undefined, 5).length).toBe(n);
  });

  it('VIF and GVIF match car; condition number diagnostics', () => {
    const { x, n } = design([R.GVIF_WT, R.GVIF_HP, R.GVIF_DISP, R.GVIF_DRAT]);
    const vif = wasm.varianceInflationFactor(x, n, 4);
    allNear(vif, R.GVIF_VIF_PLAIN, 1e-9, 'VIF');
    expect(Array.from(wasm.highVifPredictors(vif, 5))).toEqual(
      R.GVIF_VIF_PLAIN.flatMap((v, j) => (v > 5 ? [j] : [])),
    );
    const d = (v: number[], level: number) => v.map((u) => (u === level ? 1 : 0));
    const g = design([R.GVIF_WT, d(R.GVIF_CYL, 6), d(R.GVIF_CYL, 8), R.GVIF_HP, d(R.GVIF_GEAR, 4), d(R.GVIF_GEAR, 5)]);
    const gvif = wasm.generalizedVif(g.x, n, 6, new Uint32Array([1, 2, 1, 2]));
    allNear(gvif, R.GVIF_GVIF, 1e-9, 'GVIF');
    const terms = wasm.generalizedVifTerms(g.x, n, 6, new Uint32Array([1, 2, 1, 2]));
    allNear(terms.map((t: { gvif: number }) => t.gvif), R.GVIF_GVIF, 1e-9, 'GVIF terms');
    allNear(terms.map((t: { adjustedGvif: number }) => t.adjustedGvif), R.GVIF_GVIF_ADJ, 1e-9, 'GVIF^(1/(2 df))');
    expect(terms.map((t: { df: number }) => t.df)).toEqual([1, 2, 1, 2]);

    const cd = wasm.conditionDiagnostic(x, n, 4, true);
    near(cd.conditionNumber, wasm.conditionNumber(x, n, 4, true), 1e-12, 'kappa');
    expect(['well_conditioned', 'moderate', 'high', 'severe']).toContain(cd.severity);
    expect(wasm.classifyConditionNumber(1)).toBe('well_conditioned');
    const vdp = wasm.varianceDecompositionProportions(x, n, 4, true);
    expect(vdp.length).toBe(5);
    for (const row of vdp) near(row.reduce((s: number, v: number) => s + v, 0), 1, 1e-10, 'proportions sum to 1');
  });

  it('separation checks, added variables and leverage with aliased columns', () => {
    const xs = [[-2], [-1], [-0.5], [0.5], [1], [2]].flat();
    const ys = [0, 0, 0, 1, 1, 1];
    const sep = wasm.checkBinarySeparation(vector(xs), 6, 1, vector(ys));
    expect(sep.hasSeparation).toBe(true);
    expect(sep.separatedPredictors).toEqual([0]);
    expect(typeof sep.separationTypes[0]).toBe('string');
    const lin = wasm.detectSeparation(vector(xs), 6, 1, vector(ys), true);
    expect(lin.separated).toBe(true);
    expect(wasm.checkCountSparsity(vector(xs), 6, 1, vector([0, 0, 0, 3, 4, 5])).hasSeparation).toBeDefined();

    const { x, n } = design([R.LOO_MT_WT, R.LOO_MT_HP]);
    const av = wasm.addedVariable(x, n, 2, vector(R.LOO_MT_MPG), undefined, true);
    const ols = new wasm.OlsRegressor().fit(x, n, 2, vector(R.LOO_MT_MPG));
    near(av[0].slope, ols.getCoefficients()[0], 1e-10, 'AV slope = coefficient');
    const dup = design([R.LOO_MT_WT, R.LOO_MT_HP, R.LOO_MT_WT]);
    allNear(wasm.computeLeverageWithAliased(dup.x, n, 3, new Uint8Array([0, 0, 1]), true),
      wasm.computeLeverage(x, n, 2, true), 1e-10, 'aliased');
  });

  it('augment of a penalized GLM matches R', () => {
    const { x, n } = design([R.AUG_X1, R.AUG_X2]);
    const cases: [string, number[], string, any][] = [
      ['poisson', R.AUG_Y_POIS, 'POIS', null],
      ['binomial', R.AUG_Y_BIN, 'BIN', null],
    ];
    for (const [family, yv, k] of cases) {
      const m = new wasm.PenalizedGlmRegressor();
      m.setFamily(family);
      m.setMaxIterations(200);
      m.setTolerance(1e-15);
      const f = m.fit(x, n, 2, vector(yv));
      const ref = (s: string) => (R as any)[`AUG_${k}_${s}`];
      const a = f.augment(x, n, vector(yv));
      expect(a.residType).toBe('deviance');
      relAll(a.fitted, ref('FITTED'), 1e-6, `${k} fitted`);
      relAll(a.residual, ref('RESID_DEV'), 1e-6, `${k} dev`);
      relAll(a.stdResidual, ref('RSTD_DEV'), 1e-6, `${k} rstandard`);
      relAll(a.studResidual, ref('RSTUD'), 1e-6, `${k} rstudent`);
      relAll(a.leverage, ref('HAT'), 1e-6, `${k} hat`);
      relAll(a.cooksD, ref('COOK'), 1e-6, `${k} cooks`);
      relAll(a.dffits, ref('DFFITS'), 1e-6, `${k} dffits`);
      const p = f.augment(x, n, vector(yv), 'pearson');
      relAll(p.residual, ref('RESID_PEAR'), 1e-6, `${k} pearson`);
      relAll(p.stdResidual, ref('RSTD_PEAR'), 1e-6, `${k} rstandard pearson`);
      relAll(f.augment(x, n, vector(yv), 'response').residual, ref('RESID_RESP'), 1e-6, `${k} response`);
      expect(f.partialResiduals(x, n, vector(yv)).length).toBe(2);
    }
    // the dedicated Poisson regressor gives the same augment
    const pm = new wasm.PoissonRegressor();
    pm.setTolerance(1e-14);
    pm.setMaxIterations(200);
    const pa = pm.fit(x, n, 2, vector(R.AUG_Y_POIS)).augment(x, n, vector(R.AUG_Y_POIS));
    relAll(pa.cooksD, R.AUG_POIS_COOK, 1e-6, 'PoissonRegressor cooks');
    expect(() => pm.fit(x, n, 2, vector(R.AUG_Y_POIS)).augment(x, n, vector(R.AUG_Y_POIS), 'working')).toThrow();
  });
});
