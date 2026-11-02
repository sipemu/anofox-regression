// #287 phase 3: survfit / survdiff (tests/r_validation_survival.rs) and
// coxph, its residuals, cox.zph and survfit.coxph (tests/r_validation_cox.rs,
// tests/r_validation_coxdiag.rs) against R's survival / rms.
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

const ev = (v: number[]) => new Uint8Array(v.map((s) => (s === 1 ? 1 : 0)));
const flat = (m: number[][]) => m.flat();
const cols = (n: number, ...c: number[][]) => {
  const x = new Float64Array(n * c.length);
  for (let i = 0; i < n; i++) for (let j = 0; j < c.length; j++) x[i * c.length + j] = c[j][i];
  return x;
};
const tableOf = (t: any) => [
  t.records, t.nMax, t.nStart, t.events, t.rmean, t.seRmean, t.median, t.medianLower, t.medianUpper,
];

describe('survfit / kaplanMeier', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });
  const time = new Float64Array(R.SV_TIME);
  const event = ev(R.SV_STATUS);

  it('Kaplan–Meier matches survfit', () => {
    const c = wasm.kaplanMeier(time, event);
    const r = c.getResult();
    allNear(r.time, R.SV_KM_TIME, 1e-12, 'time');
    allNear(r.nRisk, R.SV_KM_NRISK, 1e-12, 'n.risk');
    allNear(r.nEvent, R.SV_KM_NEVENT, 1e-12, 'n.event');
    allNear(r.nCensor, R.SV_KM_NCENSOR, 1e-12, 'n.censor');
    allNear(r.surv, R.SV_KM_SURV, 1e-12, 'surv');
    allNear(r.stdErr, R.SV_KM_SE, 1e-12, 'std.err');
    allNear(r.lower, R.SV_KM_LOWER, 1e-10, 'lower');
    allNear(r.upper, R.SV_KM_UPPER, 1e-10, 'upper');
    allNear(r.cumhaz, R.SV_KM_CUMHAZ, 1e-12, 'cumhaz');
    allNear(r.stdChaz, R.SV_KM_SECHAZ, 1e-12, 'std.chaz');
    expect(r.stratum).toBeUndefined();
    allNear(tableOf(c.summary(Math.max(...R.SV_TIME))), R.SV_KM_TABLE, 1e-10, 'table');
    const m = c.median();
    near(m.estimate, R.SV_KM_TABLE[6], 1e-12, 'median');
    const [rmean, se] = c.restrictedMean(Math.max(...R.SV_TIME));
    near(rmean, R.SV_KM_TABLE[4], 1e-10, 'rmean');
    near(se, R.SV_KM_TABLE[5], 1e-10, 'se rmean');
    expect(c.survivalAt(0)).toBe(1);
    expect(c.survivalAt(R.SV_KM_TIME[10])).toBe(r.surv[10]);
  });

  it('confidence types and Fleming–Harrington', () => {
    let r = wasm.survfit(time, event, undefined, { confLevel: 0.9, confType: 'log-log' })[0].getResult();
    allNear(r.lower, R.SV_KM_LOGLOG_LOWER, 1e-10, 'loglog lower');
    allNear(r.upper, R.SV_KM_LOGLOG_UPPER, 1e-10, 'loglog upper');
    r = wasm.survfit(time, event, undefined, { confType: 'plain' })[0].getResult();
    allNear(r.lower, R.SV_KM_PLAIN_LOWER, 1e-10, 'plain lower');
    allNear(r.upper, R.SV_KM_PLAIN_UPPER, 1e-10, 'plain upper');
    r = wasm.survfit(time, event, undefined, { hazard: 'fleming-harrington' })[0].getResult();
    allNear(r.cumhaz, R.SV_FH_CUMHAZ, 1e-12, 'FH cumhaz');
    allNear(r.stdChaz, R.SV_FH_SECHAZ, 1e-12, 'FH std.chaz');
  });

  it('strata and the summary table', () => {
    const curves = wasm.survfit(time, event, { strata: R.SV_SEX });
    expect(curves.length).toBe(2);
    let at = 0;
    curves.forEach((c: any, s: number) => {
      expect(c.getStratum()).toBe(s);
      const m = R.SV_KMS_STRATA[s];
      const r = c.getResult();
      allNear(r.time, R.SV_KMS_TIME.slice(at, at + m), 1e-12, `time ${s}`);
      allNear(r.surv, R.SV_KMS_SURV.slice(at, at + m), 1e-12, `surv ${s}`);
      at += m;
      allNear(tableOf(c.summary(R.SV_KMS_RMEAN_END[0])), R.SV_KMS_TABLE.slice(9 * s, 9 * s + 9), 1e-10, `table ${s}`);
    });
  });

  it('case weights (Greenwood) and left truncation', () => {
    let r = wasm.survfit(time, event, { weights: R.SV_W }, { robust: false })[0].getResult();
    allNear(r.surv, R.SV_KMW_SURV, 1e-12, 'weighted surv');
    allNear(r.stdErr, R.SV_KMW_SE, 1e-12, 'weighted se');
    r = wasm
      .survfit(new Float64Array(R.SV_LT_STOP), ev(R.SV_LT_EVENT), { entry: R.SV_LT_ENTRY })[0]
      .getResult();
    allNear(r.time, R.SV_LT_TIME, 1e-12, 'LT time');
    allNear(r.nRisk, R.SV_LT_NRISK, 1e-12, 'LT n.risk');
    allNear(r.surv, R.SV_LT_SURV, 1e-12, 'LT surv');
    allNear(r.stdErr, R.SV_LT_SE, 1e-12, 'LT se');
  });

  it('median midpoint rule', () => {
    const c = wasm.kaplanMeier(new Float64Array([1, 2, 3, 4, 5, 6]), new Uint8Array(6).fill(1));
    allNear(tableOf(c.summary(6)), R.SV_MID_TABLE, 1e-12, 'midpoint');
  });

  it('rejects bad input', () => {
    expect(() => wasm.kaplanMeier(time, event.slice(1))).toThrow(/event values/);
    expect(() => wasm.survfit(time, event, undefined, { confType: 'arcsin' })).toThrow(/confType/);
  });
});

describe('survdiff', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });
  const time = new Float64Array(R.SV_TIME);
  const event = ev(R.SV_STATUS);
  const sex = new Uint32Array(R.SV_SEX);
  const ecog = new Uint32Array(R.SV_ECOG);

  it('log-rank tests match R', () => {
    let d = wasm.survdiff(time, event, sex);
    allNear(d.observed, R.SV_LR2_OBS, 1e-12, 'obs');
    allNear(d.expected, R.SV_LR2_EXP, 1e-12, 'exp');
    allNear(flat(d.variance), R.SV_LR2_VAR, 1e-10, 'var');
    near(d.chisq, R.SV_LR2_CHISQ, 1e-10, 'chisq');
    near(d.pValue, R.SV_LR2_P, 1e-6, 'p');
    expect(d.df).toBe(1);
    d = wasm.survdiff(time, event, ecog, undefined, 0);
    allNear(d.observed, R.SV_LRK_OBS, 1e-12, 'k obs');
    allNear(flat(d.variance), R.SV_LRK_VAR, 1e-10, 'k var');
    near(d.chisq, R.SV_LRK_CHISQ, 1e-10, 'k chisq');
    near(d.pValue, R.SV_LRK_P, 1e-6, 'k p');
  });

  it('Peto–Peto and stratified', () => {
    near(wasm.survdiff(time, event, ecog, undefined, 1).chisq, R.SV_PETO_CHISQ, 1e-10, 'peto');
    const d = wasm.survdiff(time, event, ecog, sex, 0);
    near(d.chisq, R.SV_STRAT_CHISQ, 1e-10, 'strat chisq');
    near(d.pValue, R.SV_STRAT_P, 1e-6, 'strat p');
  });
});

describe('coxph', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });
  const n = R.CX_TIME.length;
  const time = new Float64Array(R.CX_TIME);
  const event = ev(R.CX_STATUS);
  const x = cols(n, R.CX_AGE, R.CX_SEX, R.CX_ECOG, R.CX_WTLOSS);
  const opts = (ties: string) => ({ ties, eps: 1e-12, maxIter: 50 });
  const fit = (data?: any, ties = 'efron', o: any = {}) =>
    wasm.coxph(x, n, 4, time, event, data, { ...opts(ties), ...o });

  it('Efron fit matches coxph and cph', () => {
    const f = fit();
    const r = f.getResult();
    expect(r.converged).toBe(true);
    expect(f.getConverged()).toBe(true);
    allNear(r.coefficients, R.CX_EFRON_COEF, 1e-8, 'coef');
    allNear(f.getVcov(), R.CX_EFRON_VAR, 1e-7, 'var');
    allNear(flat(r.var), R.CX_EFRON_VAR, 1e-7, 'var rows');
    allNear(r.loglik, R.CX_EFRON_LOGLIK, 1e-11, 'loglik');
    near(r.scoreTest, R.CX_EFRON_SCORE, 1e-9, 'score');
    near(r.waldTest, R.CX_EFRON_WALD, 1e-7, 'wald');
    near(f.concordance(), R.CX_EFRON_CONCORDANCE, 1e-12, 'C');
    near(r.lrTest, R.CX_CPH_STATS[0], 1e-9, 'LR');
    near(f.r2(), R.CX_CPH_STATS[1], 1e-9, 'R2');
    near(f.dxy(), R.CX_CPH_STATS[2], 1e-12, 'Dxy');
    near(f.testPValue(r.lrTest), f.modelLrTest().pValue, 1e-12, 'LR p');
    near(f.likelihood().minusTwoLoglik, -2 * R.CX_EFRON_LOGLIK[1], 1e-11, '-2LL');
    expect(f.linearPredictors().length).toBe(n);
  });

  it('Breslow, weights + offset, strata, ridge', () => {
    allNear(fit(undefined, 'breslow').getCoefficients(), R.CX_BRESLOW_COEF, 1e-8, 'breslow');
    const w = fit({ weights: R.CX_W, offset: R.CX_OFF });
    allNear(w.getCoefficients(), R.CX_WEIGHTED_COEF, 1e-8, 'weighted');
    allNear(w.getVcov(), R.CX_WEIGHTED_VAR, 1e-7, 'weighted var');
    const s = wasm.coxph(cols(n, R.CX_AGE, R.CX_ECOG, R.CX_WTLOSS), n, 3, time, event,
      { strata: R.CX_SEX }, opts('efron'));
    allNear(s.getCoefficients(), R.CX_STRATA_COEF, 1e-8, 'strata');
    allNear(s.getVcov(), R.CX_STRATA_VAR, 1e-7, 'strata var');
    const rd = fit(undefined, 'efron', { ridge: 2 });
    allNear(rd.getCoefficients(), R.CX_RIDGE_COEF, 1e-8, 'ridge');
    allNear(rd.getVcov(), R.CX_RIDGE_VAR, 1e-7, 'ridge var');
  });

  it('score residuals and the robust variance', () => {
    const f = fit();
    allNear(flat(f.scoreResiduals()), R.CX_SCORE_RESID, 1e-7, 'score resid');
    allNear(flat(f.robustVcov(new Uint32Array(R.CX_ID))), R.CX_ROBUST_VAR, 1e-7, 'robust');
  });

  it('predicted survival matches survfit.coxph', () => {
    const f = fit();
    const m = R.CX_PRED_TIME.length;
    [[60, 1, 1, 5], [70, 2, 2, 10]].forEach((row, c) => {
      const s = f.predictSurvival(new Float64Array(row));
      const part = (v: number[]) => v.slice(c * m, (c + 1) * m);
      allNear(s.time, R.CX_PRED_TIME, 0, 'time');
      allNear(s.nRisk, R.CX_PRED_NRISK, 0, 'n.risk');
      allNear(s.surv, part(R.CX_PRED_SURV), 1e-8, `surv ${c}`);
      allNear(s.cumhaz, part(R.CX_PRED_CUMHAZ), 1e-8, `cumhaz ${c}`);
      allNear(s.stdErr, part(R.CX_PRED_SE), 1e-7, `se ${c}`);
      allNear(s.lower, part(R.CX_PRED_LOWER), 1e-7, `lower ${c}`);
      allNear(s.upper, part(R.CX_PRED_UPPER), 1e-7, `upper ${c}`);
    });
    allNear(fit(undefined, 'breslow').predictSurvival(new Float64Array([60, 1, 1, 5])).surv,
      R.CX_PREDB_SURV, 1e-8, 'breslow surv');
  });

  it('counting-process data', () => {
    const hn = R.CX_H_STOP.length;
    const f = wasm.coxph(cols(hn, R.CX_H_AGE, R.CX_H_TRANSPLANT, R.CX_H_SURGERY), hn, 3,
      new Float64Array(R.CX_H_STOP), ev(R.CX_H_EVENT), { entry: R.CX_H_START }, opts('efron'));
    allNear(f.getCoefficients(), R.CX_HEART_COEF, 1e-8, 'heart coef');
    allNear(f.getVcov(), R.CX_HEART_VAR, 1e-7, 'heart var');
    allNear(f.getResult().loglik, R.CX_HEART_LOGLIK, 1e-11, 'heart loglik');
    allNear(flat(f.robustVcov(new Uint32Array(R.CX_H_ID))), R.CX_HEART_ROBUST_VAR, 1e-7, 'heart robust');
    allNear(f.martingaleResiduals(), R.CD_HEART_MART, 1e-8, 'heart martingale');
    const z = f.coxZph();
    const got = [...z.terms, z.global].flatMap((t: any) => [t.chisq, t.df, t.pValue]);
    allNear(got, R.CD_HEART_ZPH_KM, 1e-6, 'heart zph');
  });

  it('residuals and cox.zph match R', () => {
    const f = fit();
    allNear(f.martingaleResiduals(), R.CD_EFRON_MART, 1e-8, 'martingale');
    allNear(f.devianceResiduals(), R.CD_EFRON_DEV, 1e-8, 'deviance');
    const sch = f.schoenfeldResiduals();
    allNear(flat(sch.residuals), R.CD_EFRON_SCH, 1e-8, 'schoenfeld');
    expect(sch.rows.length).toBe(sch.time.length);
    allNear(flat(f.schoenfeldResiduals(true).residuals), R.CD_EFRON_SCSCH, 1e-6, 'scaledsch');
    allNear(flat(f.dfbeta()), R.CD_EFRON_DFBETA, 1e-7, 'dfbeta');
    allNear(flat(f.dfbetas()), R.CD_EFRON_DFBETAS, 1e-7, 'dfbetas');
    const zph = (t?: string) => {
      const z = f.coxZph(t);
      return [...z.terms, z.global].flatMap((r: any) => [r.chisq, r.df, r.pValue]);
    };
    allNear(zph(), R.CD_EFRON_ZPH_KM, 1e-6, 'km');
    allNear(zph('rank'), R.CD_EFRON_ZPH_RANK, 1e-6, 'rank');
    allNear(zph('identity'), R.CD_EFRON_ZPH_IDENTITY, 1e-6, 'identity');
    allNear(zph('log'), R.CD_EFRON_ZPH_LOG, 1e-6, 'log');
    allNear(f.coxZph('km').x, R.CD_EFRON_ZPH_X, 1e-12, 'zph x');
    const grouped = f.coxZph('km', [[0, 1], [2, 3]]);
    expect(grouped.terms.map((t: any) => t.df)).toEqual([2, 2]);
    near(grouped.global.chisq, R.CD_EFRON_ZPH_KM[12], 1e-7, 'grouped global');
  });

  it('rejects bad input', () => {
    expect(() => fit(undefined, 'exact')).toThrow(/ties/);
    expect(() => wasm.coxph(x, n, 3, time, event)).toThrow(/Expected/);
    expect(() => fit().coxZph('sqrt')).toThrow(/transform/);
    expect(() => fit().predictSurvival(new Float64Array([1, 2]))).toThrow();
  });
});
