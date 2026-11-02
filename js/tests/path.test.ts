// #287 phase 5b: regularisation paths and CV against glmnet's objective /
// cv.glmnet / grpreg (tests/r_validation_reg_path.rs,
// r_validation_cv_mae.rs), fold generators against scikit-learn
// (tests/cv_splits_validation.rs) and pentrace against rms
// (tests/r_validation_pentrace.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe('regularisation paths', () => {
  const [N, P] = [R.PT_N, R.PT_P];
  /** Column-major p × L of a path's coefficients. */
  const beta = (path: any) => {
    const c = path.getResult().coefficients;
    const L = c[0].length;
    return Array.from({ length: L }, (_, l) => c.map((row: number[]) => row[l])).flat();
  };
  const cases = [
    ['ridge', { alpha: 0 }, R.PT_RIDGE_LAMBDA, R.PT_RIDGE_BETA, R.PT_RIDGE_A0, 1e-9],
    ['ridge std', { alpha: 0, standardize: true }, R.PT_RIDGE_STD_LAMBDA, R.PT_RIDGE_STD_BETA, R.PT_RIDGE_STD_A0, 1e-9],
    ['enet', { alpha: 0.5 }, R.PT_ENET_LAMBDA, R.PT_ENET_BETA, R.PT_ENET_A0, 1e-6],
    ['enet std', { alpha: 0.5, standardize: true }, R.PT_ENET_STD_LAMBDA, R.PT_ENET_STD_BETA, R.PT_ENET_STD_A0, 1e-6],
    ['lasso std', { alpha: 1, standardize: true }, R.PT_LASSO_STD_LAMBDA, R.PT_LASSO_STD_BETA, R.PT_LASSO_STD_A0, 1e-6],
  ] as const;

  it('paths and default grids match glmnet', () => {
    for (const [name, opts, lam, b, a0, tol] of cases) {
      const grid = wasm.lambdaGrid(R.PT_X, N, P, R.PT_Y, opts, lam.length);
      expect(maxAbs(grid, lam)).toBeLessThanOrEqual(1e-10 * lam[0]);
      const path = wasm.fitPath(R.PT_X, N, P, R.PT_Y, lam, opts);
      expect(path.len()).toBe(lam.length);
      expect(maxAbs(beta(path), b)).toBeLessThan(tol);
      expect(maxAbs(path.getResult().intercepts, a0)).toBeLessThan(tol);
      expect(maxAbs(path.coefficientsAt(0), b.slice(0, P))).toBeLessThan(tol);
      void name;
    }
    const g = wasm.fitPath(R.PT_X, N, P, R.PT_YS, R.PT_GRP_LASSO_LAMBDA, { standardize: true });
    expect(maxAbs(beta(g), R.PT_GRP_LASSO_BETA)).toBeLessThan(1e-5);
    // predictions: n × L, intercept + x β at every penalty
    const path = wasm.fitPath(R.PT_X, N, P, R.PT_Y, R.PT_ENET_LAMBDA, { alpha: 0.5 });
    const pred = path.predict(R.PT_X, N);
    const L = path.len();
    const r = path.getResult();
    let lp = r.intercepts[3];
    for (let j = 0; j < P; j++) lp += R.PT_X[j] * r.coefficients[j][3];
    near(pred[3], lp, 1e-12, 'predict');
    expect(pred.length).toBe(N * L);
    expect(() => path.coefficientsAt(L)).toThrow();
  });

  it('cv matches cv.glmnet aggregates', () => {
    const folds = new Uint32Array(R.PT_FOLDID);
    for (const [opts, lam, cvm, cvsd, lmin, l1se] of [
      [{ alpha: 0 }, R.PT_RIDGE_LAMBDA, R.PT_RIDGE_CVM, R.PT_RIDGE_CVSD, R.PT_RIDGE_LMIN, R.PT_RIDGE_L1SE],
      [{ alpha: 0, standardize: true }, R.PT_RIDGE_STD_LAMBDA, R.PT_RIDGE_STD_CVM, R.PT_RIDGE_STD_CVSD, R.PT_RIDGE_STD_LMIN, R.PT_RIDGE_STD_L1SE],
      [{ alpha: 0.5 }, R.PT_ENET_LAMBDA, R.PT_ENET_CVM, R.PT_ENET_CVSD, R.PT_ENET_LMIN, R.PT_ENET_L1SE],
      [{ alpha: 0.5, standardize: true }, R.PT_ENET_STD_LAMBDA, R.PT_ENET_STD_CVM, R.PT_ENET_STD_CVSD, R.PT_ENET_STD_LMIN, R.PT_ENET_STD_L1SE],
    ] as const) {
      const cv = wasm.cvPath(R.PT_X, N, P, R.PT_Y, lam, folds, opts).getResult();
      expect(maxAbs(cv.cvMean, cvm)).toBeLessThan(1e-7);
      expect(maxAbs(cv.cvSe, cvsd)).toBeLessThan(1e-7);
      expect([cv.lambdaMin, cv.lambda1se, cv.nFolds]).toEqual([lmin, l1se, 5]);
    }
    // explicit splits give the same curve
    const viaSplits = wasm.cvPathSplits(R.PT_X, N, P, R.PT_Y, R.PT_ENET_LAMBDA, wasm.splitsFromFoldIds(folds), { alpha: 0.5 });
    expect(viaSplits.getResult().lambdaMin).toBe(R.PT_ENET_LMIN);
    expect(viaSplits.path().len()).toBe(R.PT_ENET_LAMBDA.length);
  });

  it('MAE cv matches cv.glmnet', () => {
    const splits = wasm.splitsFromFoldIds(new Uint32Array(R.CM_FOLDID));
    for (const [alpha, lam, cvm, cvsd, sel] of [
      [1, R.CM_LASSO_LAMBDA, R.CM_LASSO_CVM, R.CM_LASSO_CVSD, R.CM_LASSO_LMIN_L1SE],
      [0.5, R.CM_ENET_LAMBDA, R.CM_ENET_CVM, R.CM_ENET_CVSD, R.CM_ENET_LMIN_L1SE],
    ] as const) {
      const cv = wasm.cvPathSplits(R.CM_X_FLAT, 60, 4, R.CM_Y, lam, splits, { alpha, tol: 1e-20 }, 'mae').getResult();
      expect(maxAbs(cv.cvMean, cvm)).toBeLessThan(1e-9);
      expect(maxAbs(cv.cvSe, cvsd)).toBeLessThan(1e-9);
      expect([cv.lambdaMin, cv.lambda1se]).toEqual(sel);
    }
    expect(() => wasm.cvPathSplits(R.CM_X_FLAT, 60, 4, R.CM_Y, R.CM_ENET_LAMBDA, splits, {}, 'rmse')).toThrow(/metric/);
  });

  it('seeded folds are reproducible', () => {
    const a = wasm.kfoldIds(N, 10, 99);
    expect(a).toEqual(wasm.kfoldIds(N, 10, 99));
    expect(Math.max(...a)).toBe(9);
    expect(() => wasm.kfoldIds(N, 10, -1)).toThrow(/seed/);
  });
});

describe('fold generators', () => {
  it('match scikit-learn', () => {
    expect(Array.from(wasm.groupKfoldIds(R.CS_GKF_GROUPS, 4, 'rows', 123))).toEqual(R.CS_GKF_FOLDS);
    expect(Array.from(wasm.logoIds(R.CS_GKF_GROUPS))).toEqual(R.CS_LOGO_FOLDS);
    expect(Array.from(wasm.stratifiedKfoldIds(R.CS_SKF_STRATA, 5))).toEqual(R.CS_SKF_FOLDS);
    expect(Array.from(wasm.stratifiedGroupKfoldIds(R.CS_SGKF_GROUPS, R.CS_SGKF_STRATA, 3))).toEqual(R.CS_SGKF_FOLDS);
    // string keys behave like their sorted numeric counterparts
    const asStr = R.CS_GKF_GROUPS.map((g: number) => `g${String(g + 100).padStart(4, '0')}`);
    expect(Array.from(wasm.logoIds(asStr))).toEqual(R.CS_LOGO_FOLDS);
    expect(wasm.groupCodes(['b', 'a', 'b'])).toEqual({ codes: [1, 0, 1], nGroups: 2 });
    expect(() => wasm.logoIds([1, 'a'])).toThrow(/numbers or of strings/);
  });

  it('time-series splits match scikit-learn', () => {
    const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);
    const cases: [number, number, number, number, number, number[][]][] = [
      [23, 4, 0, 0, 0, [[0, 7, 7, 11], [0, 11, 11, 15], [0, 15, 15, 19], [0, 19, 19, 23]]],
      [30, 3, 5, 2, 0, [[0, 13, 15, 20], [0, 18, 20, 25], [0, 23, 25, 30]]],
      [40, 5, 4, 1, 10, [[9, 19, 20, 24], [13, 23, 24, 28], [17, 27, 28, 32], [21, 31, 32, 36], [25, 35, 36, 40]]],
      [17, 2, 0, 3, 6, [[0, 4, 7, 12], [3, 9, 12, 17]]],
    ];
    for (const [n, k, ts, gap, mts, bounds] of cases) {
      const opts: any = { nSplits: k, gap };
      if (ts > 0) opts.testSize = ts;
      if (mts > 0) opts.maxTrainSize = mts;
      const s = wasm.timeSeriesSplits(n, opts);
      expect(s.length).toBe(bounds.length);
      s.forEach((sp: any, i: number) => {
        const [a, b, c, d] = bounds[i];
        expect(sp.train).toEqual(range(a, b));
        expect(sp.test).toEqual(range(c, d));
      });
    }
    // rows sharing a time key stay together
    const keys = [3, 1, 2, 1, 3, 2, 4, 4, 5, 5];
    for (const sp of wasm.groupTimeSeriesSplits(keys, { nSplits: 2 })) {
      const tr = new Set(sp.train.map((i: number) => keys[i]));
      expect(sp.test.every((i: number) => !tr.has(keys[i]))).toBe(true);
      expect(Math.max(...sp.train.map((i: number) => keys[i]))).toBeLessThan(Math.min(...sp.test.map((i: number) => keys[i])));
    }
    const strata = wasm.quantileStrata(Array.from({ length: 20 }, (_, i) => i), 4);
    expect(new Set(strata).size).toBe(4);
  });

  it('nested CV keeps outer test rows out of inner fits', () => {
    const n = 60;
    const x = new Float64Array(n * 3);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 3; j++) x[3 * i + j] = (((i + 1) * (j + 2) * 7) % 13) / 3;
      y[i] = 1 + x[3 * i] - 0.5 * x[3 * i + 1] + 0.2 * x[3 * i + 2] + ((i * 5) % 7) / 2;
    }
    const outer = wasm.splitsFromFoldIds(new Uint32Array(wasm.kfoldIds(n, 4, 1)));
    const groups = Array.from({ length: n }, (_, i) => Math.floor(i / 3));
    for (const inner of [
      { type: 'kfold', k: 3, seed: 2 },
      { type: 'group', groups, k: 3 },
      { type: 'logo', groups: groups.map((g) => g % 5) },
      { type: 'stratified', strata: groups.map((g) => g % 2), k: 3 },
      { type: 'timeSeries', nSplits: 3 },
    ]) {
      const all = wasm.nestedInnerSplits(n, outer, inner);
      all.forEach((splits: any[], o: number) => {
        const test = new Set(outer[o].test);
        for (const s of splits) expect([...s.train, ...s.test].some((i: number) => test.has(i))).toBe(false);
      });
    }
    const nested = wasm.nestedCvPath(x, n, 3, y, null, outer, { alpha: 0.5 },
      { inner: { type: 'kfold', k: 3, seed: 5 }, rule: '1se', nLambda: 20 });
    expect(nested.folds.length).toBe(4);
    expect(nested.metric).toBe('mse');
    near(nested.mean, nested.folds.reduce((a: number, f: any) => a + f.testMetric, 0) / 4, 1e-12, 'mean');
    expect(nested.folds[0].inner.lambdas.length).toBe(20);
    expect(() => wasm.nestedInnerSplits(n, outer, { type: 'bogus' })).toThrow(/inner folds/);
  });
});

describe('pentrace', () => {
  const [N, P] = [300, 9];
  // lrm(y ~ rcs(x1, 4) * x2 + g)
  const design = () => {
    const t = new wasm.TermMap();
    const x1 = t.addMain('x1', new Uint8Array([0, 1, 1]));
    const x2 = t.addMain('x2', new Uint8Array([0]));
    t.addMain('g', new Uint8Array([0, 0]));
    t.addInteraction(x1, x2, false);
    return t;
  };
  const cat = new Uint32Array([2]);
  const full = { simple: 1, nonlinear: 4, interaction: 2, nonlinearInteraction: 8 };

  it('penalty matrix, multipliers and the penalized lrm match rms', () => {
    const t = design();
    allNear(wasm.basePenaltyMatrix(R.PN_X, N, P, t, cat), R.PN_BASE_PENALTY, 1e-12, 'P0');
    expect(Array.from(wasm.penaltyMultipliers(t, full))).toEqual(R.PN_MULTIPLIERS);
    expect(Array.from(wasm.resolvePenalty({ simple: 2, nonlinear: 5 }))).toEqual([2, 5, 5, 5]);
    expect(Array.from(wasm.resolvePenalty(3))).toEqual([3, 3, 3, 3]);
    const pm = wasm.penaltyMatrix(R.PN_X, N, P, t, cat, full);
    const ord = new wasm.OrdinalRegressor();
    ord.setPenaltyMatrix(pm);
    const f = ord.fit(R.PN_X, N, P, R.PN_Y);
    const r = f.getResult();
    near(r.intercepts[0], R.PN_PEN_COEF[0], 1e-5, 'intercept');
    allNear(r.coefficients, R.PN_PEN_COEF.slice(1), 1e-5, 'coef');
    const byTerm = wasm.effectiveDfByTerm(t, R.PN_PEN_EFFECTIVE_DF);
    expect(byTerm.length).toBe(4);
    near(byTerm[0], R.PN_PEN_EFFECTIVE_DF.slice(0, 3).reduce((a: number, b: number) => a + b), 1e-12, 'x1 term');
    expect(() => wasm.basePenaltyMatrix(R.PN_X, N, P, t, new Uint32Array([7]))).toThrow();
  });

  it('pentrace matches rms', () => {
    const t = design();
    const grid = wasm.penaltyGrid([0, 2], [0, 4, 16], [], []);
    expect(grid.length).toBe(R.PN_TRACE_PENALTY.length / 2);
    grid.forEach((g: any, i: number) => {
      expect([g.simple, g.nonlinear]).toEqual([R.PN_TRACE_PENALTY[2 * i], R.PN_TRACE_PENALTY[2 * i + 1]]);
    });
    const tr = wasm.pentrace(R.PN_X, N, P, R.PN_Y, t, cat, grid);
    const r = tr.getResult();
    r.rows.forEach((row: any, i: number) =>
      allNear([row.df, row.aic, row.bic, row.aicC], R.PN_TRACE_STATS.slice(4 * i, 4 * i + 4), 1e-5, `row ${i}`));
    const best = r.rows[r.best].penalty;
    expect([best.simple, best.nonlinear]).toEqual(R.PN_TRACE_BEST);
    expect(tr.bestFit().getResult().coefficients.length).toBe(P);
    const uniform = Array.from({ length: R.PN_UNIFORM_STATS.length / 5 }, (_, i) => R.PN_UNIFORM_STATS[5 * i]);
    const u = wasm.pentrace(R.PN_X, N, P, R.PN_Y, t, cat, uniform, 'logistic', 'aic.c').getResult();
    u.rows.forEach((row: any, i: number) =>
      allNear([row.df, row.aic, row.bic, row.aicC], R.PN_UNIFORM_STATS.slice(5 * i + 1, 5 * i + 5), 1e-5, `uniform ${i}`));
    expect(u.rows[u.best].penalty.simple).toBe(R.PN_UNIFORM_BEST[0]);
    const bic = wasm.pentrace(R.PN_X, N, P, R.PN_Y, t, cat, uniform, 'logistic', 'bic').getResult();
    expect(bic.rows[bic.best].bic).toBe(Math.max(...bic.rows.map((x: any) => x.bic)));
    expect(() => wasm.pentrace(R.PN_X, N, P, R.PN_Y, t, cat, [])).toThrow();
    expect(() => wasm.pentrace(R.PN_X, N, P, R.PN_Y, t, cat, uniform, 'logistic', 'gcv')).toThrow(/criterion/);
  });

  it('penalized ols and its pentrace match rms', () => {
    const t = design();
    const pm = wasm.penaltyMatrix(R.PN_X, N, P, t, cat, full);
    const f = wasm.penalizedOls(R.PN_X, N, P, R.PN_Y_C, pm);
    allNear(f.coefficients, R.PN_OLS_COEF, 1e-10, 'coef');
    allNear(f.effectiveDfDiagonal, R.PN_OLS_EFFECTIVE_DF, 1e-10, 'edf');
    allNear(f.vcov.flat(), R.PN_OLS_VAR, 1e-10, 'vcov');
    allNear(f.vcovSandwich.flat(), R.PN_OLS_SANDWICH, 1e-10, 'sandwich');
    allNear([f.lrChisq, f.effectiveDf, f.rSquared, f.sigma], R.PN_OLS_STATS, 1e-10, 'stats');
    const tr = wasm.pentraceOls(R.PN_X, N, P, R.PN_Y_C, t, cat, wasm.penaltyGrid([0, 2], [0, 4, 16], [], []));
    tr.rows.forEach((row: any, i: number) =>
      allNear([row.df, row.aic, row.bic, row.aicC], R.PN_OLS_TRACE_STATS.slice(4 * i, 4 * i + 4), 1e-10, `ols row ${i}`));
    const best = tr.rows[tr.best].penalty;
    expect([best.simple, best.nonlinear]).toEqual(R.PN_OLS_TRACE_BEST);
    expect(tr.bestFit.effectiveDf).toBe(tr.rows[tr.best].df);
    expect(() => wasm.penalizedOls(R.PN_X, N, P, R.PN_Y_C, [1, 0, 0, 1])).toThrow(/penalty/);
  });
});
