// #287 phase 5c: mediation (mediation::mediate / lavaan / bda), eb_shrink
// (metafor), quantile inference (quantreg::summary.rq / predict.rq),
// streaming fit states and the regressor-level path / fit-from-state methods
// (tests/r_validation_mediation.rs, r_validation_eb_shrink.rs,
// r_validation_quantile_inference.rs, r_validation_quantile_predict_interval.rs,
// issue122_sufficient_stats.rs, r_validation_reg_path.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

describe('mediate', () => {
  const covariates = [R.MD_C1, R.MD_C2];

  it('point estimates match mediate and lavaan', () => {
    const r = wasm.mediate(R.MD_Y, R.MD_M, R.MD_X, { covariates, nSims: 2000, seed: 42 });
    allNear([r.a, r.b, r.cPrime], [R.MD_A, R.MD_B, R.MD_C_PRIME], 1e-10, 'paths');
    allNear([r.acme.estimate, r.ade.estimate, r.total.estimate, r.propMediated.estimate],
      [R.MD_PERC_ACME, R.MD_PERC_ADE, R.MD_PERC_TOTAL, R.MD_PERC_PROP], 1e-10, 'effects');
    near(r.acme.estimate, R.MD_LAVAAN_AB, 1e-8, 'lavaan');
    allNear([r.rSquaredMediator, r.rSquaredOutcome], [R.MD_R2_MEDIATOR, R.MD_R2_OUTCOME], 1e-10, 'r2');
    expect([r.nObs, r.nSims, r.method, r.bootCi]).toEqual([R.MD_N, 2000, 'bootstrap', 'percentile']);
    // our bootstrap SE within simulation error of R's draws
    expect(Math.abs(r.acme.stdError - R.MD_PERC_ACME_SD)).toBeLessThan(0.1 * R.MD_PERC_ACME_SD);
  });

  it('summaries of R draws match mediate exactly', () => {
    const p = wasm.mediationEffectFromDraws(R.MD_PERC_ACME, R.MD_PERC_ACME_DRAWS, 0.95);
    allNear([p.confLow, p.confHigh], R.MD_PERC_ACME_CI, 1e-10, 'percentile');
    near(p.pValue, R.MD_PERC_ACME_P, 1e-10, 'p');
    near(p.stdError, R.MD_PERC_ACME_SD, 1e-10, 'sd');
    const b = wasm.mediationEffectFromDraws(R.MD_PERC_ACME, R.MD_PERC_ACME_DRAWS, 0.95, 'bca');
    allNear([b.confLow, b.confHigh], R.MD_BCA_ACME_CI, 1e-10, 'bca');
    expect(() => wasm.mediationEffectFromDraws(0, [], 0.95)).toThrow();
  });

  it('continuous treatment, Sobel and options', () => {
    const c = wasm.mediate(R.MD_YC, R.MD_MC, R.MD_XC, { covariates, nSims: 200, treatValue: 2, controlValue: 0.5 });
    allNear([c.acme.estimate, c.ade.estimate], [R.MD_CONT_ACME, R.MD_CONT_ADE], 1e-10, 'continuous');
    const s = wasm.mediate(R.MD_Y, R.MD_M, R.MD_X, { method: 'sobel' });
    allNear([s.acme.statistic, s.acme.pValue], [R.MD_SOBEL_NOCOV_Z, R.MD_SOBEL_NOCOV_P], 1e-8, 'sobel');
    expect(s.nSims).toBe(0);
    const sc = wasm.mediate(R.MD_Y, R.MD_M, R.MD_X, { method: 'sobel', covariates });
    near(sc.ade.stdError, R.MD_SE_C_PRIME, 1e-10, 'ade se');
    near(sc.acme.stdError, Math.sqrt(R.MD_A ** 2 * R.MD_SE_B ** 2 + R.MD_B ** 2 * R.MD_SE_A ** 2), 1e-10, 'acme se');
    const qb = wasm.mediate(R.MD_Y, R.MD_M, R.MD_X, { method: 'quasi-bayesian', covariates, nSims: 500 });
    expect(qb.method).toBe('quasi_bayesian');
    expect(() => wasm.mediate(R.MD_Y, R.MD_M, R.MD_X, { method: 'magic' })).toThrow(/method/);
    expect(() => wasm.mediate(R.MD_Y, R.MD_M.slice(1), R.MD_X)).toThrow();
  });
});

describe('ebShrink', () => {
  const yi = [0.1, 0.3, 0.35, 0.65, 1.0];
  const sei = [0.3, 0.1, 0.5, 0.2, 0.4];

  it('matches metafor rma(DL), fixed effect and fixed τ²', () => {
    const r = wasm.ebShrink(yi, sei);
    allNear([r.mu, r.muSe, r.tauSquared, r.q, r.iSquared],
      [4.223980166707431e-1, 1.260284589424375e-1, 2.394587965990844e-2, 5.737201081799203, 3.027959203504876e-1], 1e-10, 'DL');
    expect(r.nGroups).toBe(5);
    expect(r.groups.length).toBe(5);
    const fe = wasm.ebShrink(yi, sei, { method: 'none' });
    allNear([fe.mu, fe.muSe], [3.758587967356235e-1, 8.265842980736918e-2], 1e-10, 'FE');
    expect(fe.groups.every((g: any) => g.shrunken === fe.mu)).toBe(true);
    const fx = wasm.ebShrink(yi, sei, { tauSquared: 0.05 });
    allNear([fx.mu, fx.muSe, fx.tauSquared], [4.385608856088561e-1, 1.5247043146894e-1, 0.05], 1e-10, 'fixed');
    expect(() => wasm.ebShrink(yi, sei.slice(1))).toThrow();
    expect(() => wasm.ebShrink(yi, sei, { method: 'reml' })).toThrow(/method/);
  });
});

describe('quantile inference', () => {
  const N = R.QI_N;
  const x = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) x.set([R.QI_X1[i], R.QI_X2[i]], 2 * i);

  it('iid and nid match summary.rq', () => {
    R.QI_TAUS.forEach((tau: number, k: number) => {
      const q = new wasm.QuantileRegressor();
      q.setTau(tau);
      const fit = q.fit(x, N, 2, R.QI_Y);
      for (const se of ['iid', 'nid']) {
        const ref = (s: string) => (R as any)[`QI_T${k + 1}_${se.toUpperCase()}_${s}`];
        const inf = fit.inference(x, N, R.QI_Y, se, 0.95).getResult();
        expect(Math.max(...inf.estimate.map((v: number, j: number) => Math.abs(v - ref('COEF')[j])))).toBeLessThan(1e-8);
        expect(Math.max(...inf.stdError.map((v: number, j: number) => Math.abs(v - ref('SE')[j])))).toBeLessThan(1e-8);
        expect(Math.max(...inf.pValue.map((v: number, j: number) => Math.abs(v - ref('P')[j])))).toBeLessThan(1e-8);
        expect(inf.seType).toBe(se);
        // the free function gives the same table
        const free = wasm.quantileInference(x, N, 2, R.QI_Y, tau, true, se, 0.95).getResult();
        allNear(free.stdError, inf.stdError, 1e-12, 'free');
      }
    });
  });

  it('process, bootstrap and bandwidth', () => {
    const proc = wasm.quantileProcess(x, N, 2, R.QI_Y, R.QI_PROCESS_TAUS, true, 'nid', 0.9);
    expect(Math.max(...proc.flatMap((r: any) => r.estimate).map((v: number, i: number) => Math.abs(v - R.QI_PROCESS_COEF[i])))).toBeLessThan(1e-8);
    const noSe = wasm.quantileProcess(x, N, 2, R.QI_Y, [0.5], true, null, 0.95);
    expect(noSe[0].stdError.every((v: number) => Number.isNaN(v))).toBe(true);
    const boot = (seed: number) =>
      wasm.quantileInference(x, N, 2, R.QI_Y, 0.5, true, { type: 'boot', seed, r: 400 }, 0.95).getResult();
    const a = boot(7);
    expect(a.stdError).toEqual(boot(7).stdError);
    expect(a.nBoot).toBe(400);
    expect(a.seType).toEqual({ type: 'boot', seed: 7, r: 400 });
    a.stdError.forEach((s: number, j: number) => expect(Math.abs(s / R.QI_BOOT_MEDIAN_SE[j] - 1)).toBeLessThan(0.2));
    expect(wasm.hallSheatherBandwidth(0.5, 100)).toBeGreaterThan(0);
    expect(() => wasm.quantileInference(x, N, 2, R.QI_Y, 0.5, true, 'xy', 0.95)).toThrow(/se/);
  });

  it('predicted-quantile intervals match predict.rq', () => {
    const xq = new Float64Array(120);
    for (let i = 0; i < 60; i++) xq.set([R.QP_X1[i], R.QP_X2[i]], 2 * i);
    const nx = new Float64Array([R.QP_NEW_X1[0], R.QP_NEW_X2[0], R.QP_NEW_X1[1], R.QP_NEW_X2[1], R.QP_NEW_X1[2], R.QP_NEW_X2[2]]);
    const check = (tau: number, intercept: boolean, se: string, level: number, want: number[]) => {
      const inf = wasm.quantileInference(xq, 60, 2, R.QP_Y, tau, intercept, se, 0.95);
      const p = inf.predictInterval(nx, 3, level);
      allNear([...p.fit, ...p.lower, ...p.upper], want, 1e-8, `${se} ${tau} ${level}`);
    };
    check(0.5, true, 'iid', 0.95, R.QP_IID_TAU50_L95);
    check(0.5, true, 'iid', 0.8, R.QP_IID_TAU50_L80);
    check(0.25, true, 'nid', 0.95, R.QP_NID_TAU25_L95);
    check(0.5, false, 'nid', 0.9, R.QP_NID_NOINT_TAU50_L90);
    const inf = wasm.quantileInference(xq, 60, 2, R.QP_Y, 0.5, true, 'iid', 0.95);
    expect(() => inf.predictInterval(nx, 3, 1)).toThrow();
  });
});

describe('streaming fit states and regressor paths', () => {
  const [N, P] = [R.PT_N, R.PT_P];

  it('OLS / ridge from SufficientStats and MomentAccumulator equal the row fits', () => {
    const ols = new wasm.OlsRegressor();
    const direct = ols.fit(R.PT_X, N, P, R.PT_Y);
    // two chunks, merged, serialised and restored
    const half = N / 2;
    const a = wasm.SufficientStats.fromData(R.PT_X.slice(0, half * P), half, P, R.PT_Y.slice(0, half));
    const b = new wasm.SufficientStats(P);
    for (let i = half; i < N; i++) b.push(R.PT_X.slice(i * P, i * P + P), R.PT_Y[i]);
    const state = wasm.SufficientStats.fromBytes(a.merged(b).toBytes());
    expect([state.nObservations(), state.nFeatures(), state.isWeighted()]).toEqual([N, P, false]);
    const fs = ols.fitFromState(state);
    allNear(fs.getCoefficients(), direct.getCoefficients(), 1e-10, 'ols state');
    near(fs.getIntercept(), direct.getIntercept(), 1e-10, 'ols intercept');
    const acc = new wasm.MomentAccumulator(P);
    acc.pushRows(R.PT_X, N, R.PT_Y);
    expect(acc.n()).toBe(N);
    allNear(ols.fitFromAccumulator(acc).getCoefficients(), direct.getCoefficients(), 1e-9, 'ols acc');
    allNear(ols.fitFromMoments(acc.xtx(), acc.xty(), acc.sumX(), acc.sumY(), acc.n()).getCoefficients(),
      direct.getCoefficients(), 1e-9, 'ols moments');
    const ridge = new wasm.RidgeRegressor();
    ridge.setLambda(0.5);
    allNear(ridge.fitFromState(state).getCoefficients(), ridge.fit(R.PT_X, N, P, R.PT_Y).getCoefficients(), 1e-9, 'ridge state');
    expect(() => wasm.SufficientStats.fromBytes(new Uint8Array([1, 2, 3]))).toThrow();
    expect(() => acc.pushRow([1, 2], 1)).toThrow();
  });

  it('ridge / elastic net / lars paths delegate to the path solver', () => {
    const ridge = new wasm.RidgeRegressor();
    const p = ridge.path(R.PT_X, N, P, R.PT_Y, R.PT_RIDGE_LAMBDA);
    allNear(p.getResult().intercepts, R.PT_RIDGE_A0, 1e-9, 'ridge path');
    const enet = new wasm.ElasticNetRegressor();
    enet.setAlpha(0.5);
    const cv = enet.cvPath(R.PT_X, N, P, R.PT_Y, R.PT_ENET_LAMBDA, new Uint32Array(R.PT_FOLDID)).getResult();
    expect(cv.lambdaMin).toBe(R.PT_ENET_LMIN);
    allNear(enet.path(R.PT_X, N, P, R.PT_Y, R.PT_ENET_LAMBDA).getResult().intercepts, R.PT_ENET_A0, 1e-6, 'enet path');
    const lars = new wasm.LarsRegressor().fit(R.PT_X, N, P, R.PT_Y);
    const lp = lars.path();
    const last = lp.len() - 1;
    allNear(lp.coefficientsAt(last), lars.getCoefficients(), 1e-8, 'lars end = OLS');
    expect(lp.getResult().df[0]).toBe(0);
  });
});
