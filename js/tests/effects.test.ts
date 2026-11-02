// #287 phase 4a: spline bases (tests/r_validation_rcs.rs), Design, and
// effects / contrasts / Predict / nomogram / simultaneous bands
// (tests/r_validation_effects.rs, r_validation_nomogram.rs,
// r_validation_simultaneous.rs) against Hmisc and rms.
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

describe('splines', () => {
  it('default knots match rcspline.eval', () => {
    [R.RC_KNOTS_SMALL_3, R.RC_KNOTS_SMALL_4, R.RC_KNOTS_SMALL_5, R.RC_KNOTS_SMALL_6, R.RC_KNOTS_SMALL_7]
      .forEach((want, i) => allNear(wasm.rcsDefaultKnots(R.RC_X_SMALL, i + 3), want, 1e-14, `nk=${i + 3}`));
    allNear(wasm.rcsDefaultKnots(R.RC_X_LARGE, 5), R.RC_KNOTS_LARGE_5, 1e-14, 'large');
    allNear(wasm.rcsDefaultKnots(R.RC_X_TIEDLOW, 4), R.RC_KNOTS_TIEDLOW_4, 1e-14, 'tied low');
    allNear(wasm.rcsDefaultKnots(R.RC_X_FEWU, 5), R.RC_KNOTS_FEWU_5, 1e-14, 'few unique');
    allNear(wasm.defaultKnotQuantiles(5), [0.05, 0.275, 0.5, 0.725, 0.95], 1e-15, 'quantiles');
    expect(() => wasm.rcsDefaultKnots([1, 2, 3], 3)).toThrow();
  });

  it('rcs basis for every norm and rms::rcs', () => {
    const rcs = wasm.Rcs.fromData(R.RC_X_SMALL, 5);
    allNear(rcs.knots(), R.RC_KNOTS_SMALL_5, 1e-14, 'knots');
    expect(rcs.norm()).toBe('range');
    for (const [norm, want] of [
      ['none', R.RC_BASIS_SMALL_5_NORM0],
      ['last-interval', R.RC_BASIS_SMALL_5_NORM1],
      ['range', R.RC_BASIS_SMALL_5_NORM2],
    ] as const) {
      rcs.setNorm(norm);
      allNear(rcs.basis(R.RC_X_SMALL, 'x').columns(), want, 1e-12, norm);
    }
    expect(() => rcs.setNorm('cube')).toThrow(/norm/);
    const b = wasm.Rcs.fromData(R.RC_X_LARGE, 4).basis(R.RC_X_LARGE, 'x');
    expect(b.names()).toEqual(['x', "x'", "x''"]);
    expect(Array.from(b.nonlinear())).toEqual([0, 1, 1]);
    expect(Array.from(b.nonlinearColumns())).toEqual([1, 2]);
    expect([b.nRows(), b.nColumns()]).toEqual([300, 3]);
    allNear(b.columns(), R.RC_RMS_RCS_LARGE_4, 1e-12, 'rms rcs');
  });

  it('restate matches rcspline.restate', () => {
    const rcs = wasm.Rcs.fromData(R.RC_X_SMALL, 5);
    const r = rcs.restate(R.RC_RESTATE_COEF_IN[0], R.RC_RESTATE_COEF_IN.slice(1));
    const res = r.getResult();
    allNear([res.intercept, res.linear, ...res.cubic], R.RC_RESTATE_COEF_OUT, 1e-12, 'restate');
    const cols = rcs.basis(R.RC_X_SMALL, 'x').columns();
    R.RC_X_SMALL.forEach((x: number, i: number) => {
      let fit = R.RC_RESTATE_COEF_IN[0];
      for (let j = 0; j < 4; j++) fit += R.RC_RESTATE_COEF_IN[j + 1] * cols[4 * i + j];
      near(r.eval(x), fit, 1e-10, `eval ${i}`);
    });
  });

  it('lsp and %ia% match rms', () => {
    const b = new wasm.Lsp(R.RC_LSP_KNOTS).basis(R.RC_X_SMALL, 'x');
    expect(b.names()).toEqual(['x', "x'", "x''", "x'''"]);
    allNear(b.columns(), R.RC_LSP_SMALL, 1e-14, 'lsp');
    const a = wasm.Rcs.fromData(R.RC_X_SMALL, 4).basis(R.RC_X_SMALL, 'x1');
    const c = wasm.Rcs.fromData(R.RC_IA_X2, 3).basis(R.RC_IA_X2, 'x2');
    const ia = wasm.restrictedInteraction(a, c);
    expect(ia.names()).toEqual(['x1 * x2', "x1 * x2'", "x1' * x2", "x1'' * x2"]);
    allNear(ia.columns(), R.RC_IA, 1e-12, 'ia');
  });

  it('Basis constructor and Design', () => {
    const g = new wasm.Basis([0, 1, 0, 0, 0, 1], 3, ['g=2', 'g=3'], new Uint8Array([0, 0]));
    expect(() => new wasm.Basis([0, 1], 1, ['a'], new Uint8Array([0]))).toThrow();
    const d = new wasm.Design();
    const x = d.add('x', wasm.Basis.linear([1, 2, 3], 'x'));
    const t = d.add('g', g);
    expect(d.addInteraction(x, t)).toBe(2);
    expect(d.columnNames()).toEqual(['x', 'g=2', 'g=3', 'x * g=2', 'x * g=3']);
    expect([d.nRows(), d.nColumns()]).toEqual([3, 5]);
    expect(Array.from(d.matrix())).toEqual([1, 0, 1, 0, 1, 2, 0, 0, 0, 0, 3, 0, 1, 0, 3]);
    expect(d.terms().nColumns()).toBe(5);
    expect(d.block(1).names()).toEqual(['g=2', 'g=3']);
    expect(d.block(9)).toBeUndefined();
    expect(() => d.add('z', wasm.Basis.linear([1, 2], 'z'))).toThrow();
  });
});

describe('ModelSpec effects', () => {
  // y ~ rcs(x1, 4) * x2 + g, rms column order
  const spec = (knots: number[]) => {
    const s = new wasm.ModelSpec();
    const x1 = s.rcs('x1', new wasm.Rcs(knots), R.EF_X1);
    const x2 = s.linear('x2', R.EF_X2);
    s.categorical('g', R.EF_G);
    s.interaction(x1, x2);
    return s;
  };

  it('datadist limits', () => {
    const p = spec(R.EF_OLS_KNOTS).predictors();
    expect(p.map((q: any) => q.kind)).toEqual(['rcs', 'linear', 'categorical']);
    [R.EF_LIM_X1, R.EF_LIM_X2].forEach((want, k) => {
      const l = p[k].limits;
      allNear([l.lowEffect, l.adjustTo, l.highEffect, l.lowPrediction, l.highPrediction], want.slice(0, 5), 1e-12, `lim ${k}`);
    });
    const l = wasm.numericLimits(R.EF_X2);
    near(l.adjustTo, R.EF_LIM_X2[1], 1e-12, 'numericLimits');
  });

  const check = (id: string, coef: number[], vcov: number[], knots: number[], opts: any,
    summary: number[], contrast: number[], double: number[], cg: number[],
    px: number[], py: number[], pl: number[], pu: number[]) => {
    const s = spec(knots);
    const rows = s.summary(coef, vcov, true, opts);
    expect(rows.length * 7).toBe(summary.length);
    rows.forEach((r: any, i: number) => {
      const w = summary.slice(7 * i, 7 * i + 7);
      allNear([r.low, r.high], [w[0], w[1]], 1e-12, `${id} ${r.label} low/high`);
      allNear([r.effect.estimate, r.effect.se, r.effect.lower, r.effect.upper], [w[3], w[4], w[5], w[6]], 1e-9, `${id} ${r.label}`);
    });
    // by name and by index
    const c = s.contrast(coef, vcov, true, { x1: 60, x2: 2 }, [[0, 45], [1, 2]], opts);
    allNear([c.estimate, c.se, c.lower, c.upper, c.statistic], contrast.slice(0, 5), 1e-6, `${id} contrast`);
    const d = s.doubleContrast(coef, vcov, true, { x1: 60, x2: 8 }, { x1: 45, x2: 8 }, { x1: 60, x2: 2 }, { x1: 45, x2: 2 }, opts);
    allNear([d.estimate, d.se, d.lower, d.upper], double.slice(0, 4), 1e-9, `${id} double`);
    near(d.pValue, double[4], 1e-6, `${id} double p`);
    const g = s.contrast(coef, vcov, true, { g: 3 }, { g: 2 }, opts);
    allNear([g.estimate, g.se, g.lower, g.upper], cg.slice(0, 4), 1e-9, `${id} contrast g`);
    const p = s.predictGrid(coef, vcov, true, 0, undefined, opts);
    allNear(p.values, px, 1e-12, `${id} grid`);
    allNear(p.estimates.map((e: any) => e.estimate), py, 1e-9, `${id} yhat`);
    allNear(p.estimates.map((e: any) => e.lower), pl, 1e-8, `${id} lower`);
    allNear(p.estimates.map((e: any) => e.upper), pu, 1e-8, `${id} upper`);
  };

  it('ols effects match rms (t reference)', () => {
    check('ols', R.EF_OLS_COEF, R.EF_OLS_VCOV, R.EF_OLS_KNOTS, { df: R.EF_X1.length - R.EF_OLS_COEF.length },
      R.EF_OLS_SUMMARY, R.EF_OLS_CONTRAST, R.EF_OLS_DOUBLE, R.EF_OLS_CONTRAST_G,
      R.EF_OLS_PRED_X1, R.EF_OLS_PRED_YHAT, R.EF_OLS_PRED_LOWER, R.EF_OLS_PRED_UPPER);
  });

  it('lrm effects and odds ratios match rms (normal reference)', () => {
    const s = spec(R.EF_LRM_KNOTS);
    s.summary(R.EF_LRM_COEF, R.EF_LRM_VCOV, true).forEach((r: any, i: number) => {
      allNear([Math.exp(r.effect.estimate), Math.exp(r.effect.lower), Math.exp(r.effect.upper)],
        R.EF_LRM_OR.slice(3 * i, 3 * i + 3), 1e-9, `OR ${r.label}`);
    });
    const c = s.contrast(R.EF_LRM_COEF, R.EF_LRM_VCOV, true, { x1: 60, x2: 2 }, { x1: 45, x2: 2 });
    allNear([c.estimate, c.se], R.EF_LRM_CONTRAST.slice(0, 2), 1e-6, 'lrm contrast');
    const p = s.predictGrid(R.EF_LRM_COEF, R.EF_LRM_VCOV, true, 0, undefined, { level: 0.95 });
    allNear(p.estimates.map((e: any) => e.estimate), R.EF_LRM_PRED_YHAT, 1e-9, 'lrm yhat');
    allNear(p.estimates.map((e: any) => e.lower), R.EF_LRM_PRED_LOWER, 1e-8, 'lrm lower');
  });

  it('lsp and %ia% specifications match rms', () => {
    for (const [id, coef, vcov, summary, contrast] of [
      ['lsp', R.EF_LSP_COEF, R.EF_LSP_VCOV, R.EF_LSP_SUMMARY, R.EF_LSP_CONTRAST],
      ['ia', R.EF_IA_COEF, R.EF_IA_VCOV, R.EF_IA_SUMMARY, R.EF_IA_CONTRAST],
    ] as const) {
      const s = new wasm.ModelSpec();
      const x1 = s.rcs('x1', new wasm.Rcs(R.EF_OLS_KNOTS), R.EF_X1);
      const x2 = id === 'lsp'
        ? s.lsp('x2', new wasm.Lsp([3, 6]), R.EF_X2)
        : s.rcs('x2', new wasm.Rcs(R.EF_IA_KNOTS_X2), R.EF_X2);
      s.categorical('g', R.EF_G);
      if (id === 'ia') s.restrictedInteraction(x1, x2);
      const opts = { df: R.EF_X1.length - coef.length };
      s.summary(coef, vcov, true, opts).forEach((r: any, i: number) => {
        const w = summary.slice(7 * i, 7 * i + 7);
        allNear([r.effect.estimate, r.effect.se, r.effect.lower, r.effect.upper], [w[3], w[4], w[5], w[6]], 1e-9, `${id} ${r.label}`);
      });
      const c = s.contrast(coef, vcov, true, { x1: 60, x2: 7 }, { x1: 45, x2: 2 }, opts);
      allNear([c.estimate, c.se, c.lower, c.upper], contrast, 1e-9, `${id} contrast`);
      if (id === 'ia') {
        const p = s.predictGrid(coef, vcov, true, 1, undefined, opts);
        allNear(p.values, R.EF_IA_PRED_X2, 1e-12, 'ia grid');
        allNear(p.estimates.map((e: any) => e.estimate), R.EF_IA_PRED_YHAT, 1e-9, 'ia yhat');
        allNear(p.estimates.map((e: any) => e.upper), R.EF_IA_PRED_UPPER, 1e-8, 'ia upper');
      }
    }
  });

  it('design and rows follow the specification', () => {
    const s = spec(R.EF_OLS_KNOTS);
    const d = s.design([R.EF_X1, R.EF_X2, R.EF_G]);
    expect(d.nColumns()).toBe(R.EF_OLS_COEF.length - 1);
    expect(d.nRows()).toBe(R.EF_X1.length);
    const row = s.row({ x1: 60 }, true);
    expect(row.length).toBe(R.EF_OLS_COEF.length);
    expect(row[0]).toBe(1);
    expect(() => s.row({ nope: 1 }, true)).toThrow(/nope/);
    expect(() => s.design([R.EF_X1])).toThrow(/predictor columns/);
    expect(() => s.summary([1, 2], [1, 0, 0, 1], true)).toThrow(/coefficients/);
    expect(() => s.interaction(0, 7)).toThrow(/predictor 7/);
  });
});

describe('nomogram', () => {
  const plogis = (x: number) => 1 / (1 + Math.exp(-x));

  it('matches rms::nomogram', () => {
    const s = new wasm.ModelSpec();
    s.rcs('age', wasm.Rcs.fromData(R.NM_AGE, 4), R.NM_AGE);
    s.categorical('sex', R.NM_SEX);
    s.linear('chol', R.NM_CHOL);
    s.linear('grade', R.NM_GRADE);
    const nom = s.nomogram(R.NM_BETA, true, R.NM_LP);
    const r = nom.getResult();
    const want = [
      [R.NM_AGE_VALUES, R.NM_AGE_XBETA, R.NM_AGE_POINTS],
      [R.NM_SEX_VALUES, R.NM_SEX_XBETA, R.NM_SEX_POINTS],
      [R.NM_CHOL_VALUES, R.NM_CHOL_XBETA, R.NM_CHOL_POINTS],
      [R.NM_GRADE_VALUES, R.NM_GRADE_XBETA, R.NM_GRADE_POINTS],
    ];
    r.axes.forEach((a: any, k: number) => {
      allNear(a.values, want[k][0], 1e-12, `${a.name} values`);
      allNear(a.xbeta, want[k][1], 1e-10, `${a.name} xbeta`);
      allNear(a.points, want[k][2], 1e-10, `${a.name} points`);
    });
    allNear([r.scale, r.intercept], R.NM_SCALE_INTERCEPT, 1e-10, 'scale');
    allNear(r.totalPoints, R.NM_TOTAL_POINTS, 1e-12, 'total');
    allNear(r.lpAt, R.NM_LP_AT, 1e-12, 'lp.at');
    allNear(r.lpPoints, R.NM_LP_POINTS, 1e-10, 'lp points');
    const risk = nom.functionAxis(plogis, 10);
    allNear(risk.values, R.NM_RISK_AT, 1e-12, 'risk ticks');
    allNear(risk.points, R.NM_RISK_POINTS, 1e-10, 'risk points');
    const total = r.axes.reduce((t: number, a: any) => t + a.points[0], 0);
    const lp = R.NM_BETA[0] + r.axes.reduce((t: number, a: any) => t + a.xbeta[0], 0);
    near(nom.pointsToLp(total), lp, 1e-10, 'points → lp');
    near(nom.lpToPoints(lp), total, 1e-8, 'lp → points');
    expect(() => nom.functionAxis(() => { throw new Error('boom'); })).toThrow(/boom/);
    expect(() => nom.functionAxis(() => 'x')).toThrow(/number/);
    expect(() => nom.functionAxis(Math.log)).toThrow(/finite/);
  });

  it('explicit ticks; interactions are rejected', () => {
    const s = new wasm.ModelSpec();
    const a = s.linear('chol', R.NM_CHOL);
    const b = s.linear('grade', R.NM_GRADE);
    const nom = s.nomogram([0.5, 0.01, 0.3], true, [0, 3], [[150, 250], null], 100, 10).getResult();
    expect(nom.axes[0].values).toEqual([150, 250]);
    expect(nom.axes[1].values).toEqual([1, 2, 3, 4]);
    s.interaction(a, b);
    expect(() => s.nomogram([0.5, 0.01, 0.3], true, [0, 3])).toThrow(/main effects/);
  });
});

describe('simultaneous bands', () => {
  const spec = (ageRcs: boolean, withW: boolean) => {
    const s = new wasm.ModelSpec();
    if (ageRcs) s.rcs('age', wasm.Rcs.fromData(R.SB_AGE, 4), R.SB_AGE);
    else s.linear('age', R.SB_AGE);
    s.linear('sex', R.SB_SEX);
    if (withW) s.linear('w', R.SB_W);
    return s;
  };
  const check = (tag: string, s: any, k: number, grid: number[], beta: number[], vcov: number[],
    opts: any, pred: number[], crit: number[]) => {
    const r = s.predictGridSimultaneous(beta, vcov, true, k, grid, opts);
    near(r.critical, crit[1], 2e-4, `${tag} crit`);
    const g = grid.length;
    r.estimates.forEach((e: any, i: number) => {
      near(e.estimate, pred[i], 1e-10, `${tag} yhat ${i}`);
      near(e.lower, pred[g + i], 2e-3, `${tag} lower ${i}`);
      near(e.upper, pred[2 * g + i], 2e-3, `${tag} upper ${i}`);
    });
  };

  it('match Predict(conf.type = "simultaneous")', () => {
    check('LRM', spec(true, false), 0, R.SB_LRM_GRID, R.SB_LRM_BETA, R.SB_LRM_VCOV, undefined, R.SB_LRM_PRED, R.SB_LRM_CRIT);
    check('OLS', spec(false, true), 2, R.SB_OLS_GRID, R.SB_OLS_BETA, R.SB_OLS_VCOV, { df: 220 - 4 }, R.SB_OLS_PRED, R.SB_OLS_CRIT);
    check('CAT', spec(true, false), 1, R.SB_CAT_GRID, R.SB_CAT_BETA, R.SB_CAT_VCOV, undefined, R.SB_CAT_PRED, R.SB_CAT_CRIT);
  });

  it('rank one is pointwise; bad input is NaN', () => {
    const v = [1, 0.3, 0.3, 1];
    near(wasm.simultaneousCritical([[1, 0], [2, 0]], v), 1.959963984540054, 1e-12, 'rank 1');
    expect(wasm.simultaneousCritical([[1, 0], [2, 0]], v, { level: 1.5 })).toBeNaN();
  });
});
