// #287 phase 4b: validate / calibrate / bootcov / calibrate.cph against rms,
// replaying R's resamples (tests/r_validation_validate.rs,
// r_validation_bootcov.rs, r_validation_calibrate_cph.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

const n = R.VA_X1.length;
const x = new Float64Array(n * 3);
for (let i = 0; i < n; i++) [R.VA_X1[i], R.VA_X2[i], R.VA_X3[i]].forEach((v, j) => (x[3 * i + j] = v));
const chunks = (idx: number[], len: number) => {
  const out: number[][] = [];
  for (let i = 0; i < idx.length; i += len) out.push(idx.slice(i, i + len));
  return out;
};

/** R's table is (index × [orig, training, test, optimism, corrected, n]). */
function check(id: string, t: any, want: number[], tol: number) {
  const r = t.getResult();
  r.names.forEach((name: string, k: number) => {
    const row = want.slice(6 * k, 6 * k + 6);
    const g = t.get(name);
    near(g.orig, row[0], tol, `${id} ${name} orig`);
    near(g.training, row[1], tol, `${id} ${name} training`);
    near(g.test, row[2], tol, `${id} ${name} test`);
    near(g.optimism, row[3], tol * 10, `${id} ${name} optimism`);
    near(g.corrected, row[4], tol, `${id} ${name} corrected`);
    expect(g.n).toBe(row[5]);
    near(r.corrected[k], row[4], tol, `${id} ${name} table`);
  });
}

describe('validate', () => {
  it('bootstrap, cross-validation and .632 match rms::validate', () => {
    const boot = wasm.Resamples.bootstrap(n, chunks(R.VA_BOOT_IDX, n));
    expect(boot.method()).toBe('bootstrap');
    check('lrm boot', wasm.validateLrm(x, n, 3, R.VA_YB, boot), R.VA_LRM_BOOT, 1e-5);
    check('ols boot', wasm.validateOls(x, n, 3, R.VA_YC, boot), R.VA_OLS_BOOT, 1e-9);
    const cv = wasm.Resamples.crossvalidation(new Uint32Array(R.VA_CV_PERM), 10);
    expect(cv.splits().length).toBe(10);
    check('lrm cv', wasm.validateLrm(x, n, 3, R.VA_YB, cv), R.VA_LRM_CV, 1e-5);
    check('ols cv', wasm.validateOls(x, n, 3, R.VA_YC, cv), R.VA_OLS_CV, 1e-9);
    const p632 = wasm.Resamples.point632(n, chunks(R.VA_P632_IDX, n));
    expect(p632.weights().length).toBe(30);
    check('lrm .632', wasm.validateLrm(x, n, 3, R.VA_YB, p632), R.VA_LRM_632, 1e-5);
    check('ols .632', wasm.validateOls(x, n, 3, R.VA_YC, p632), R.VA_OLS_632, 1e-9);
    expect(wasm.validateOls(x, n, 3, R.VA_YC, boot).get('nope')).toBeUndefined();
  });

  it('seeded resamples run end to end', () => {
    const t = wasm.validateLrm(x, n, 3, R.VA_YB, wasm.Resamples.bootstrapSeeded(n, 20, 42)).get('Dxy');
    expect(t.n).toBe(20);
    expect(t.optimism).toBeGreaterThan(0);
    near(t.orig - t.optimism, t.corrected, 1e-12, 'corrected');
    expect(wasm.Resamples.crossvalidationSeeded(n, 5, 1).n()).toBe(n);
    expect(wasm.Resamples.point632Seeded(n, 40, 7).method()).toBe('point632');
    expect(() => wasm.Resamples.bootstrapSeeded(n, 5, -1)).toThrow(/seed/);
    expect(() => wasm.Resamples.crossvalidation(new Uint32Array([0, 0, 1, 2]), 2)).toThrow(/permutation/);
  });

  it('predabResample with JS callbacks reproduces validateOls', () => {
    const r = wasm.Resamples.bootstrap(n, chunks(R.VA_BOOT_IDX, n));
    const ols = new wasm.OlsRegressor();
    const fit = (rows: Uint32Array) => {
      const xs = new Float64Array(rows.length * 3);
      const ys = new Float64Array(rows.length);
      rows.forEach((i, k) => {
        xs.set(x.subarray(3 * i, 3 * i + 3), 3 * k);
        ys[k] = R.VA_YC[i];
      });
      return ols.fit(xs, rows.length, 3, ys).predict(x, n); // predictions on all rows
    };
    const measure = (pred: Float64Array, rows: Uint32Array, apparent: boolean) => {
      const p = Array.from(rows, (i) => pred[i]);
      const y = Array.from(rows, (i) => R.VA_YC[i]);
      const m = y.length;
      const ybar = y.reduce((a, b) => a + b) / m;
      const sst = y.reduce((a, v) => a + (v - ybar) ** 2, 0);
      const sse = y.reduce((a, v, k) => a + (v - p[k]) ** 2, 0);
      let a = 0, b = 1;
      if (!apparent) {
        const pbar = p.reduce((s, v) => s + v) / m;
        const sxx = p.reduce((s, v) => s + (v - pbar) ** 2, 0);
        const sxy = p.reduce((s, v, k) => s + (v - pbar) * (y[k] - ybar), 0);
        b = sxy / sxx;
        a = ybar - b * pbar;
      }
      return [1 - sse / sst, sse / m, a, b];
    };
    const t = wasm.predabResample(r, ['R-square', 'MSE', 'Intercept', 'Slope'], fit, measure);
    const want = wasm.validateOls(x, n, 3, R.VA_YC, r);
    for (const name of ['R-square', 'MSE', 'Intercept', 'Slope']) {
      near(t.get(name).corrected, want.get(name).corrected, 1e-9, name);
    }
    expect(() => wasm.predabResample(r, ['a'], () => { throw new Error('boom'); }, measure)).toThrow(/fit threw: boom/);
    expect(() => wasm.predabResample(r, ['a', 'b'], fit, () => [1])).toThrow(/2 numbers/);
  });
});

describe('calibrate', () => {
  it('matches rms::calibrate on the default and a user grid', () => {
    const r = wasm.Resamples.bootstrap(n, chunks(R.VA_BOOT_IDX, n));
    const cases = [
      ['lrm', wasm.calibrateLrm(x, n, 3, R.VA_YB, r), R.VA_CAL_LRM_PREDY, R.VA_CAL_LRM_ORIG, R.VA_CAL_LRM_CORR, R.VA_CAL_LRM_OPT, R.VA_CAL_LRM_N, R.VA_CAL_LRM_ERR],
      ['ols', wasm.calibrateOls(x, n, 3, R.VA_YC, r), R.VA_CAL_OLS_PREDY, R.VA_CAL_OLS_ORIG, R.VA_CAL_OLS_CORR, R.VA_CAL_OLS_OPT, R.VA_CAL_OLS_N, R.VA_CAL_OLS_ERR],
      ['lrm grid', wasm.calibrateLrm(x, n, 3, R.VA_YB, r, R.VA_PREDY_LRM_GRID), R.VA_PREDY_LRM_GRID, R.VA_CAL_LRM_GRID_ORIG, R.VA_CAL_LRM_GRID_CORR, R.VA_CAL_LRM_GRID_OPT, R.VA_CAL_LRM_GRID_N, R.VA_CAL_LRM_GRID_ERR],
      ['ols grid', wasm.calibrateOls(x, n, 3, R.VA_YC, r, R.VA_PREDY_OLS_GRID), R.VA_PREDY_OLS_GRID, R.VA_CAL_OLS_GRID_ORIG, R.VA_CAL_OLS_GRID_CORR, R.VA_CAL_OLS_GRID_OPT, R.VA_CAL_OLS_GRID_N, R.VA_CAL_OLS_GRID_ERR],
    ] as const;
    for (const [id, cal, predy, orig, corr, opt, nn, err] of cases) {
      allNear(cal.predy, predy, 1e-8, `${id} predy`);
      allNear(cal.calibratedOrig, orig, 1e-7, `${id} orig`);
      allNear(cal.calibratedCorrected, corr, 1e-6, `${id} corrected`);
      allNear(cal.optimism, opt, 1e-5, `${id} optimism`);
      expect(cal.n).toEqual(nn);
      allNear([cal.meanAbsError, cal.meanSquaredError, cal.quantile90AbsError], err, 1e-6, `${id} errors`);
    }
    expect(() => wasm.calibrateOls(x, n, 3, R.VA_YC.slice(1), r)).toThrow(/y values/);
  });

  it('calibrate.cph matches rms', () => {
    const N = R.CC_TIME.length;
    const b = R.CC_BOOT_IDX.length / N;
    const r = wasm.Resamples.bootstrap(N, chunks(R.CC_BOOT_IDX, N));
    expect(b).toBe(r.splits().length);
    const event = new Uint8Array(R.CC_EVENT.map((e: number) => (e === 1 ? 1 : 0)));
    const cal = wasm.calibrateCph(R.CC_X, N, 3, R.CC_TIME, event, R.CC_U_M[0], r, R.CC_U_M[1], undefined,
      { eps: 1e-4, maxIter: 10 });
    allNear(cal.predicted, R.CC_PREDICTED, 1e-9, 'predicted');
    const g = cal.km.length;
    const col = (k: number) => Array.from({ length: g }, (_, i) => R.CC_CAL[i * 10 + k]);
    const fields = ['indexOrig', 'training', 'test', 'optimism', 'corrected', 'n', 'meanPredicted', 'km', 'kmCorrected', 'stdErr'];
    fields.forEach((f, k) => allNear(cal[f], col(k), 1e-9, f));
    const two = wasm.calibrateCph(R.CC_X, N, 3, R.CC_TIME, event, 5, wasm.Resamples.bootstrapSeeded(N, 5, 3), 60, [0, 0.5, 1]);
    expect(two.km.length).toBe(2);
    expect(() => wasm.calibrateCph(R.CC_X, N, 3, R.CC_TIME, event, 5, r, 60, [0.5])).toThrow(/cuts/);
  });
});

describe('bootcov', () => {
  const m = R.BC_Y.length;
  let logistic: any;
  const fit = (rows: Uint32Array) => {
    if (!logistic) {
      logistic = new wasm.LogisticRegression();
      logistic.setTolerance(1e-12);
    }
    const xs = new Float64Array(rows.length * 2);
    const ys = new Float64Array(rows.length);
    rows.forEach((i, k) => {
      xs[2 * k] = R.BC_X1[i];
      xs[2 * k + 1] = R.BC_X2[i];
      ys[k] = R.BC_Y[i];
    });
    const f = logistic.fit(xs, rows.length, 2, ys);
    return [f.getIntercept(), ...f.getCoefficients()];
  };

  it('ordinary bootstrap matches rms::bootcov', () => {
    const b = wasm.bootcov(chunks(R.BC_BOOT_IDX, m), fit);
    const r = b.getResult();
    expect(r.failed).toBe(0);
    allNear(b.getVcov(), R.BC_BOOT_VAR, 1e-4, 'vcov');
    allNear(r.vcov.flat(), R.BC_BOOT_VAR, 1e-4, 'vcov rows');
    allNear(b.percentileCi(0.95).flat(), R.BC_BOOT_Q, 1e-5, 'percentile');
    allNear(b.linearCombinationCi([0, 1, 0], 0.95), R.BC_BOOT_Q.slice(2, 4), 1e-5, 'lincomb');
    expect(() => b.linearCombinationCi([1, 0], 0.95)).toThrow(/weights/);
  });

  it('cluster bootstrap matches rms::bootcov', () => {
    let at = 0;
    const samples = R.BC_CL_LENS.map((l: number) => R.BC_CL_ROWS.slice(at, (at += l)));
    allNear(wasm.bootcov(samples, fit).getVcov(), R.BC_CL_VAR, 1e-4, 'cluster vcov');
  });

  it('seeded samples, failures and rank intervals', () => {
    expect(wasm.bootstrapRows(10, 3, 1).length).toBe(3);
    const rows = wasm.clusterBootstrapRows(new Uint32Array(R.BC_ID), 20, 9);
    expect(rows.every((r: number[]) => r.length % 4 === 0)).toBe(true);
    expect(wasm.bootcov(rows, fit).getVcov()[4]).toBeGreaterThan(0);
    let calls = 0;
    const flaky = wasm.bootcov(rows, (r: Uint32Array) => (calls++ % 2 ? null : fit(r)));
    expect(flaky.getResult().failed).toBe(10);
    expect(() => wasm.bootcov(rows, () => 'x')).toThrow(/numbers/);
    const k = R.VA_RANK_ORIG.length;
    const got = wasm.rankIntervals(R.VA_RANK_ORIG, R.VA_RANK_REPS, 0.9);
    expect(got.length).toBe(k);
    allNear(got.flat(), R.VA_RANK_TABLE, 1e-12, 'ranks');
  });
});
