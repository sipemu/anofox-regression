// #287 phase 2b: inference through WASM against the R references of the
// Rust validation (rms::anova / fastbw, sandwich::vcovCL, glm LR / Rao /
// profile CIs, Rubin's rules, predict(se.fit = TRUE), nested F, HC).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear, near } from './rel';
import * as R from './references';

function design(cols: number[][]): { x: Float64Array; n: number; p: number } {
  const n = cols[0].length;
  const p = cols.length;
  const x = new Float64Array(n * p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) x[i * p + j] = cols[j][i];
  return { x, n, p };
}

/** |a − b| ≤ tol · (floor + |b|) elementwise over a nested or flat matrix. */
function matNear(got: number[][] | ArrayLike<number>, want: number[], tol: number, floor: number, what: string) {
  const flat = Array.isArray(got) && Array.isArray(got[0]) ? (got as number[][]).flat() : Array.from(got as ArrayLike<number>);
  expect(flat.length, what).toBe(want.length);
  flat.forEach((a, i) => {
    if (!(Math.abs(a - want[i]) <= tol * (floor + Math.abs(want[i])))) throw new Error(`${what}[${i}]: ${a} vs ${want[i]}`);
  });
}

describe('inference', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  function termsM1() {
    const t = new wasm.TermMap();
    const x1 = t.addMain('x1', new Uint8Array([0, 1, 1]));
    const x3 = t.addMain('x3', new Uint8Array([0]));
    t.addMain('x2', new Uint8Array([0, 1]));
    t.addMain('x4', new Uint8Array([0]));
    t.addInteraction(x1, x3, false);
    return t;
  }

  it('anova (Wald, rms) and general linear hypotheses', () => {
    const t = termsM1();
    expect(t.nColumns()).toBe(10);
    expect(t.find('x2')).toBe(2);
    expect(t.terms()[4].interaction.factors).toEqual([0, 1]);
    const table = wasm.anova(vector(R.AN_BETA_M1), vector(R.AN_VCOV_M1), t);
    const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
    expect(table.rows.map((r: any) => norm(r.label))).toEqual(R.AN_LABELS_M1.map(norm));
    let at = 0;
    table.rows.forEach((r: any, k: number) => {
      expect(r.columns).toEqual(R.AN_WHICH_M1.slice(at, at + R.AN_WHICH_LEN_M1[k]));
      at += R.AN_WHICH_LEN_M1[k];
      expect(r.test.df).toBe(R.AN_DF_M1[k]);
      near(r.test.chiSquare, R.AN_CHISQ_M1[k], 1e-9, r.label);
      expect(Math.abs(r.test.pValue - R.AN_P_M1[k])).toBeLessThanOrEqual(1e-5 * R.AN_P_M1[k] + 1e-15);
      expect(typeof r.kind).toBe('string');
      near(r.test.chiSquareMinusDf, r.test.chiSquare - r.test.df, 1e-12, 'chi2 - df');
    });
    const ft = wasm.anova(vector(R.AN_BETA_M1), vector(R.AN_VCOV_M1), t, R.AN_DFE_M1, R.AN_SIGMA2_M1);
    ft.rows.forEach((r: any, k: number) => {
      near(r.test.f, R.AN_F_M1[k], 1e-9, `${r.label} F`);
      near(r.partialSs, R.AN_SS_M1[k], 1e-9, `${r.label} SS`);
    });
    expect(ft.error[0]).toBe(R.AN_DFE_M1);
    near(ft.error[2], R.AN_SIGMA2_M1, 1e-12, 'mse');

    const p = R.AN_LM_BETA.length;
    const chi = wasm.waldTest(vector(R.AN_LM_BETA), vector(R.AN_LM_VCOV), vector(R.AN_GLH_C), vector(R.AN_GLH_R));
    expect(chi.df).toBe(2);
    near(chi.chiSquare, R.AN_GLH_CHISQ, 1e-10, 'GLH chi2');
    near(chi.pValue, R.AN_GLH_CHISQ_P, 1e-9, 'GLH p');
    const f = wasm.waldTest(vector(R.AN_LM_BETA), vector(R.AN_LM_VCOV), vector(R.AN_GLH_C), vector(R.AN_GLH_R), R.AN_LM_DFE);
    near(f.f, R.AN_GLH_CHISQ / 2, 1e-10, 'GLH F');
    near(f.pValue, R.AN_GLH_F_P, 1e-9, 'GLH F p');
    const chunk = wasm.waldChunk(vector(R.AN_BETA_M1), vector(R.AN_VCOV_M1), new Uint32Array([7]));
    near(chunk.chiSquare, R.AN_BETA_M1[7] ** 2 / R.AN_VCOV_M1[7 * 10 + 7], 1e-10, 'chunk');
    const d = wasm.dropIntercept(vector(R.AN_LM_BETA), vector(R.AN_LM_VCOV));
    expect(d.beta.length).toBe(p - 1);
    expect(d.vcov.length).toBe(p - 1);
    expect(() => wasm.waldTest(vector([1, 2]), vector([1]), vector([1, 0]))).toThrow();
  });

  it('fastbw matches rms', () => {
    const terms = () => {
      const t = new wasm.TermMap();
      t.addMain('x1', new Uint8Array([0, 1, 1]));
      for (const name of ['x2', 'x3', 'x4', 'x5']) t.addMain(name, new Uint8Array([0]));
      t.addMain('g', new Uint8Array([0, 0]));
      return t;
    };
    const cases: [number[], number[], number, string, number, string, number | undefined, number[], number[], number[]][] = [
      [R.FBW_COEF, R.FBW_VCOV, 7, 'aic', 0, 'residual', 2, R.FBW_AIC_RESIDUAL_DELETED, R.FBW_AIC_RESIDUAL_RESULT, R.FBW_AIC_RESIDUAL_COEF],
      [R.FBW_COEF, R.FBW_VCOV, 7, 'aic', 0, 'individual', 2, R.FBW_AIC_INDIVIDUAL_DELETED, R.FBW_AIC_INDIVIDUAL_RESULT, R.FBW_AIC_INDIVIDUAL_COEF],
      [R.FBW_COEF, R.FBW_VCOV, 7, 'p', 0.05, 'residual', undefined, R.FBW_P_RESIDUAL_DELETED, R.FBW_P_RESIDUAL_RESULT, R.FBW_P_RESIDUAL_COEF],
      [R.FBW_COEF, R.FBW_VCOV, 7, 'p', 0.05, 'individual', undefined, R.FBW_P_INDIVIDUAL_DELETED, R.FBW_P_INDIVIDUAL_RESULT, R.FBW_P_INDIVIDUAL_COEF],
      [R.FBW_COEF, R.FBW_VCOV, 7, 'p', 0.2, 'residual', undefined, R.FBW_P20_RESIDUAL_DELETED, R.FBW_P20_RESIDUAL_RESULT, R.FBW_P20_RESIDUAL_COEF],
      [R.FBW_COEF, R.FBW_VCOV, 7, 'aic', 2, 'individual', Math.log(300), R.FBW_AICS2_LOGN_INDIVIDUAL_DELETED, R.FBW_AICS2_LOGN_INDIVIDUAL_RESULT, R.FBW_AICS2_LOGN_INDIVIDUAL_COEF],
      [R.FBW_OLS_COEF, R.FBW_OLS_VCOV, 8, 'aic', 0, 'residual', 2, R.FBW_OLS_AIC_RESIDUAL_DELETED, R.FBW_OLS_AIC_RESIDUAL_RESULT, R.FBW_OLS_AIC_RESIDUAL_COEF],
      [R.FBW_OLS_COEF, R.FBW_OLS_VCOV, 8, 'p', 0.1, 'individual', undefined, R.FBW_OLS_P_INDIVIDUAL_DELETED, R.FBW_OLS_P_INDIVIDUAL_RESULT, R.FBW_OLS_P_INDIVIDUAL_COEF],
    ];
    for (const [coef, vcov, stride, rule, threshold, type, k, deleted, result, want] of cases) {
      const s = wasm.fastbw(vector(coef), vector(vcov), 1, terms(), rule, threshold, type, k);
      const tag = `${rule} ${threshold} ${type}`;
      expect(s.deleted.map((d: any) => d.term), tag).toEqual(deleted);
      s.deleted.forEach((d: any, r: number) =>
        allNear([d.chisq, d.df, d.pValue, d.residualChisq, d.residualDf, d.residualP, d.aic],
          result.slice(r * stride, r * stride + 7), 1e-9, `${tag} step ${r}`),
      );
      allNear(s.coefficients, want, 1e-9, `${tag} coef`);
    }
    expect(() => wasm.fastbw(vector(R.FBW_COEF), vector(R.FBW_VCOV), 1, terms(), 'bic')).toThrow();
  });

  it('cluster-robust covariances match sandwich::vcovCL', () => {
    const { x, n } = design([R.CL_X1, R.CL_X2]);
    const y = vector(R.CL_Y);
    const ols = new wasm.OlsRegressor().fit(x, n, 2, y);
    const types: [string, number[]][] = [['hc0', R.CL_LM_HC0], ['hc1', R.CL_LM_HC1], ['hc2', R.CL_LM_HC2], ['hc3', R.CL_LM_HC3]];
    for (const [type, want] of types) {
      matNear(ols.clusterVcov(x, n, [R.CL_G1], { type, cadjust: true }), want, 1e-10, 1e-6, `OLS ${type}`);
    }
    matNear(ols.clusterVcov(x, n, [R.CL_G1], { type: 'hc0', cadjust: false }), R.CL_LM_HC0_NOADJ, 1e-10, 1e-6, 'noadj');
    matNear(ols.clusterVcov(x, n, [R.CL_G1, R.CL_G2], 'cr1'), R.CL_LM_TWOWAY_HC1, 1e-10, 1e-6, 'two-way HC1');
    matNear(ols.clusterVcov(x, n, [R.CL_G1, R.CL_G2], 'hc0'), R.CL_LM_TWOWAY_HC0, 1e-10, 1e-6, 'two-way HC0');
    matNear(ols.clusterVcov(x, n, [new Uint32Array(R.CL_G1)], 'robcov'), R.CL_ROBCOV, 1e-10, 1e-6, 'robcov');
    matNear(ols.clusterVcov(x, n, [R.CL_G1]), R.CL_LM_HC1, 1e-10, 1e-6, 'default cr1');
    const wm = new wasm.WlsRegressor();
    wm.setWeights(vector(R.CL_W));
    matNear(wm.fit(x, n, 2, y).clusterVcov(x, n, [R.CL_G1], 'cr1'), R.CL_WLS_HC1, 1e-10, 1e-6, 'WLS');
    const m0 = new wasm.OlsRegressor();
    m0.setWithIntercept(false);
    matNear(m0.fit(x, n, 2, y).clusterVcov(x, n, [R.CL_G1], 'cr1'), R.CL_LM0_HC1, 1e-10, 1e-6, 'no intercept');
    // the free function on the residuals agrees
    const res = vector(ols.getResult().residuals);
    matNear(wasm.lmClusterVcov(x, n, 2, res, undefined, true, undefined, [R.CL_G1], 'cr1'), R.CL_LM_HC1, 1e-10, 1e-6, 'lmClusterVcov');
    const yc = vector(R.CL_YC);
    const mu = vector(R.CL_POIS_MU);
    matNear(wasm.glmClusterVcov('poisson', x, n, 2, true, yc, mu, undefined, [R.CL_G1], 'hc0'), R.CL_POIS_HC0, 1e-6, 1e-6, 'poisson HC0');
    matNear(wasm.glmClusterVcov('poisson', x, n, 2, true, yc, mu, undefined, [R.CL_G1], 'cr1'), R.CL_POIS_HC1, 1e-6, 1e-6, 'poisson HC1');
    matNear(wasm.glmClusterVcov('poisson', x, n, 2, true, yc, mu, undefined, [R.CL_G1, R.CL_G2], 'hc0'), R.CL_POIS_TWOWAY, 1e-8, 1e-6, 'poisson two-way');
    expect(() => ols.clusterVcov(x, n, [R.CL_G1.slice(1)])).toThrow();
    expect(() => ols.clusterVcov(x, n, [R.CL_G1], 'cr3')).toThrow();
  });

  it('GLM likelihood-ratio, score tests and profile CIs match R', () => {
    const full = design([R.LIK_X1, R.LIK_X2, R.LIK_X3]);
    const red = design([R.LIK_X1]);
    const add = design([R.LIK_X2, R.LIK_X3]);
    const n = full.n;
    const cases: [string, number[], string | number, string][] = [['binomial', R.LIK_YB, 1, 'BIN'], ['gamma', R.LIK_YG, 'estimated', 'GAMMA']];
    for (const [family, yv, dispersion, k] of cases) {
      const ref = (s: string) => (R as any)[`LIK_${k}_${s}`];
      const opts = { dispersion };
      const f = wasm.fitGlm(family, full.x, n, 3, vector(yv), opts);
      const r = wasm.fitGlm(family, red.x, n, 1, vector(yv), opts);
      const fr = f.getResult();
      allNear(fr.coefficients, ref('COEF'), 1e-9, `${k} coef`);
      matNear(fr.vcov, ref('VCOV'), 1e-8, 1, `${k} vcov`);
      near(fr.deviance, ref('DEV'), 1e-10, `${k} deviance`);
      near(fr.dispersion, ref('DISP'), 1e-8, `${k} dispersion`);
      const t = wasm.lrTest(r.likelihood(), f.likelihood());
      expect(t.df).toBe(2);
      near(t.statistic, ref('LR'), 1e-8, `${k} LR`);
      near(t.pValue, ref('LR_P'), 1e-6, `${k} LR p`);
      const m = f.modelLrTest();
      expect(m.df).toBe(3);
      near(m.statistic, (ref('NULLDEV') - ref('DEV')) / ref('DISP'), 1e-8, `${k} model LR`);
      const s = wasm.glmScoreTest(family, red.x, n, 1, vector(yv), add.x, 2, fr.dispersion);
      near(s.test.statistic, ref('RAO') / ref('DISP'), 1e-7, `${k} Rao`);
      near(s.test.pValue, ref('RAO_P'), 1e-6, `${k} Rao p`);
      const z = wasm.glmScoreTest(family, red.x, n, 1, vector(yv), add.x, 2, r.getResult().dispersion).z;
      allNear(z, ref('SCORE_Z'), 1e-7, `${k} score z`);
      for (const [level, want] of [[0.95, ref('CI95')], [0.9, ref('CI90')]] as [number, number[]][]) {
        const ci = wasm.profileConfint(family, full.x, n, 3, vector(yv), level, opts);
        allNear(ci.flat(), want, 1e-6, `${k} ${level} CI`);
      }
    }
    // the dedicated regressors expose the same likelihood summary
    const b = new wasm.BinomialRegressor();
    b.setTolerance(1e-12);
    const bf = b.fit(full.x, n, 3, vector(R.LIK_YB));
    near(bf.likelihood().minusTwoLoglik, R.LIK_BIN_DEV, 1e-8, 'binomial -2 logLik');
    near(bf.modelLrTest().statistic, R.LIK_BIN_NULLDEV - R.LIK_BIN_DEV, 1e-8, 'binomial model LR');
    expect(() => wasm.fitGlm('binomial', full.x, n, 3, vector(R.LIK_YB), { dispersion: 'free' })).toThrow();
  });

  it('Rubin pooling and the D1 test match Hmisc / mice', () => {
    const m = R.POOL_M[0];
    const P = 4;
    const est = Array.from({ length: m }, (_, i) => R.POOL_COEFS.slice(i * P, i * P + P));
    // the fixture stores each covariance column-major
    const vc = Array.from({ length: m }, (_, i) => {
      const v = R.POOL_VCOVS.slice(i * P * P, (i + 1) * P * P);
      return Array.from({ length: P * P }, (_, k) => v[(k % P) * P + Math.floor(k / P)]);
    });
    const f = wasm.poolRubin(est, vc);
    const r = f.getResult();
    allNear(r.coefficients, R.POOL_POOL_COEF, 1e-12, 'coef');
    allNear(r.lambda, R.POOL_POOL_FMI, 1e-12, 'lambda');
    allNear(r.dfRubin, R.POOL_POOL_DF, 1e-10, 'df');
    expect(r.dfBarnardRubin).toBeUndefined();
    near(r.stdErrors[1], Math.sqrt(r.vcov[1][1]), 1e-14, 'se');
    const br = wasm.poolRubin(est, vc, 146).getResult();
    allNear(br.riv, R.POOL_MICE_RIV, 1e-12, 'riv');
    allNear(br.dfBarnardRubin, R.POOL_MICE_DF, 1e-10, 'BR df');
    for (const [idx, want] of [[[1, 2], R.POOL_D1_X1X2], [[1, 2, 3], R.POOL_D1_ALL]] as [number[], number[]][]) {
      const t = f.waldD1(new Uint32Array(idx));
      allNear([t.f, t.df1, t.df2, t.pValue, t.riv], want, 1e-9, 'D1');
    }
  });

  it('predictSe matches predict(se.fit = TRUE)', () => {
    const { x, n } = design([R.PSE_X1, R.PSE_X2]);
    const nx = design([R.PSE_NEW_X1, R.PSE_NEW_X2]).x;
    const lin = (f: any, k: string) => {
      const ref = (s: string) => (R as any)[`PSE_${k}_${s}`];
      const c = f.predictSe(nx, 5);
      allNear(c.fit, ref('FIT'), 1e-9, `${k} fit`);
      allNear(c.seFit, ref('SE'), 1e-9, `${k} se`);
      allNear(c.lower, ref('CONF_LWR'), 1e-9, `${k} lwr`);
      allNear(c.upper, ref('CONF_UPR'), 1e-9, `${k} upr`);
      near(c.residualScale, ref('SIGMA')[0], 1e-9, `${k} sigma`);
      expect([c.scale, c.interval]).toEqual(['response', 'confidence']);
      const p = f.predictSe(nx, 5, { interval: 'prediction', level: 0.9 });
      allNear(p.lower, ref('PRED90_LWR'), 1e-9, `${k} pred lwr`);
      allNear(p.upper, ref('PRED90_UPR'), 1e-9, `${k} pred upr`);
    };
    lin(new wasm.OlsRegressor().fit(x, n, 2, vector(R.PSE_Y_LIN)), 'OLS');
    const wm = new wasm.WlsRegressor();
    wm.setWeights(vector(R.PSE_W));
    lin(wm.fit(x, n, 2, vector(R.PSE_Y_LIN)), 'WLS');

    const b = new wasm.BinomialRegressor();
    b.setTolerance(1e-12);
    b.setMaxIterations(200);
    const bf = b.fit(x, n, 2, vector(R.PSE_Y_BIN));
    const l = bf.predictSe(nx, 5, { scale: 'link' });
    const r = bf.predictSe(nx, 5);
    allNear(l.fit, R.PSE_BIN_ETA, 1e-7, 'eta');
    allNear(l.seFit, R.PSE_BIN_SE_ETA, 1e-7, 'se eta');
    allNear(r.fit, R.PSE_BIN_MU, 1e-7, 'mu');
    allNear(r.seFit, R.PSE_BIN_SE_MU, 1e-7, 'se mu');
    allNear(r.lower, R.PSE_BIN_MU_LWR, 1e-7, 'mu lwr');
    allNear(r.upper, R.PSE_BIN_MU_UPR, 1e-7, 'mu upr');
    expect(l.df).toBe(Infinity);
    expect(() => bf.predictSe(nx, 5, { interval: 'prediction' })).toThrow();
    expect(() => bf.predictSe(nx, 5, { scale: 'probability' })).toThrow();

    // the variance-factor helpers reproduce the OLS intervals
    const ols = new wasm.OlsRegressor().fit(x, n, 2, vector(R.PSE_Y_LIN));
    const sigma2 = R.PSE_OLS_SIGMA[0] ** 2;
    const M = ols.varianceFactor().flat();
    const iv = wasm.intervalsFromVarianceFactor(nx, 5, 2, vector(M), vector(R.PSE_OLS_FIT), sigma2, n - 3, 0.95, 'confidence', true);
    allNear(iv.lower, R.PSE_OLS_CONF_LWR, 1e-9, 'variance factor lwr');
    const xtx = wasm.computeXtxInverse(x, n, 2, true).flat();
    allNear(xtx, M, 1e-9, '(XtX)^-1 = M');
    const pi = wasm.computePredictionIntervals(nx, 5, 2, vector(xtx), vector(R.PSE_OLS_FIT), sigma2, n - 3, 0.9, 'prediction', true);
    allNear(pi.lower, R.PSE_OLS_PRED90_LWR, 1e-9, 'prediction interval');
    allNear(ols.leverageNew(nx, 5), wasm.leverageNew(vector(M), nx, 5, 2, true), 1e-12, 'leverageNew');
    matNear(ols.vcov(), M.map((v: number) => v * sigma2), 1e-9, 1e-12, 'vcov = sigma2 M');
    matNear(wasm.computeVarianceFactor(x, n, 2, undefined, true, undefined, 0), M, 1e-9, 1e-12, 'computeVarianceFactor');
  });

  it('nested F, HC standard errors, coefficient helpers and anovaTable', () => {
    const X1 = [1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9];
    const X2 = [0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5];
    const X3 = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];
    const Y = [2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4];
    const full = design([X1, X2, X3]).x;
    const t = wasm.nestedFTest(vector(X1), 1, full, 3, vector(Y));
    near(t.fStatistic, 3.91675014697775055e1, 1e-9, 'F');
    expect([t.df1, t.df2]).toEqual([2, 8]);
    near(t.pValue, 7.37245814150955264e-5, 1e-9, 'p');
    near(t.rssReduced, 2.70743110048637554e1, 1e-9, 'rss reduced');
    near(t.deltaAdjRSquared, 9.92283908364675837e-1 - 9.33383120598241467e-1, 1e-9, 'delta adj R2');
    const ti = wasm.nestedFTestIndices(full, 3, vector(Y), new Uint32Array([0]));
    near(ti.fStatistic, t.fStatistic, 1e-12, 'indices');

    const xs = [1, 2, 3, 4, 5];
    const ys = [2.1, 4.3, 5.8, 8.2, 9.9];
    const ols = new wasm.OlsRegressor().fit(vector(xs), 5, 1, vector(ys));
    const want: [string, number, number][] = [['hc0', 0.12237, 0.03178], ['hc1', 0.15796, 0.04103], ['hc2', 0.15489, 0.04185], ['hc3', 0.20343, 0.05722]];
    for (const [type, se0, se1] of want) {
      const hc = ols.hcStandardErrors(vector(xs), 5, type);
      expect(Math.abs(hc.interceptStdError - se0)).toBeLessThan(0.01);
      expect(Math.abs(hc.stdErrors[0] - se1)).toBeLessThan(0.01);
      const inf = ols.hcInference(vector(xs), 5, type);
      near(inf.stdErrors[0], hc.stdErrors[0], 1e-12, 'hcInference');
      expect(inf.hcType).toBe(type);
      near(inf.intercept.stdError, hc.interceptStdError, 1e-12, 'intercept');
      const free = wasm.computeHcStandardErrors(vector(xs), 5, 1, vector(ols.getResult().residuals), true, type);
      near(free.stdErrors[0], hc.stdErrors[0], 1e-12, 'free HC');
    }
    const r = ols.getResult();
    const mse = r.residuals.reduce((s: number, v: number) => s + v * v, 0) / 3;
    const se = wasm.standardErrorsWithIntercept(vector(xs), 5, 1, mse);
    near(se.stdErrors[0], r.stdErrors[0], 1e-10, 'se');
    const tt = wasm.tStatistics(vector(r.coefficients), vector(se.stdErrors));
    allNear(wasm.coefficientPValues(tt, 3), r.pValues, 1e-10, 'p values');
    const ci = wasm.coefficientConfidenceIntervals(vector(r.coefficients), vector(se.stdErrors), 3, 0.95);
    expect(ci.lower[0]).toBeLessThan(r.coefficients[0]);
    expect(wasm.coefficientStandardErrors(vector(xs), 5, 1, mse).length).toBe(1);

    const at = wasm.anovaTable(full, 12, 3, vector(Y));
    expect(at.terms.length).toBe(3);
    expect(at.anovaType).toBe('i');
    const at2 = wasm.anovaTable(full, 12, 3, vector(Y), [[0], [1, 2]], 'II');
    expect(at2.terms[1].df).toBe(2);
    near(at2.terms[1].fStatistic, t.fStatistic, 1e-9, 'type II F = nested F');
    expect(() => wasm.anovaTable(full, 12, 3, vector(Y), undefined, 'III')).toThrow();
  });

  it('model approximation and shrinkage', () => {
    const t = new wasm.TermMap();
    t.addMain('a', new Uint8Array([0]));
    t.addMain('b', new Uint8Array([0]));
    const n = 30;
    const a = Array.from({ length: n }, (_, i) => Math.sin(i));
    const b = Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    const lp = a.map((v, i) => 1 + 2 * v + 0.1 * b[i]);
    const x = design([a, b]).x;
    const vcov = [1, 0, 0, 0, 1, 0, 0, 0, 1].map((v) => v * 0.01);
    const ap = wasm.approximateModel(x, n, vector(lp), vector(vcov), t, new Uint32Array([0]));
    expect(ap.keptTerms).toEqual([0]);
    expect(ap.rSquared).toBeGreaterThan(0.95);
    const path = wasm.approximationPath(x, n, vector(lp), t);
    expect(path.length).toBeGreaterThan(0);
    near(wasm.heuristicShrinkage(20, 4), 0.8, 1e-12, 'shrinkage');
    allNear(wasm.shrinkSlopes(vector([1, 2, 4]), 1, 0.5), [1, 1, 2], 1e-14, 'shrink slopes');
  });
});
