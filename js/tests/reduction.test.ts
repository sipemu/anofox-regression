// #287 phase 5a: data reduction, transformations and imputation against
// Hmisc / stats / MASS (tests/r_validation_reduction.rs, r_validation_pca.rs,
// r_validation_transcan.rs, r_validation_redun.rs, r_validation_transform.rs,
// r_validation_areg.rs, r_validation_aregimpute.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule } from './setup';
import { near, allNear } from './rel';
import * as R from './references';

let wasm: any;
beforeAll(async () => {
  wasm = await getWasmModule();
});

/** Column-major `p × p` from arrays of rows. */
const colMajor = (m: number[][]) => m[0].map((_, j) => m.map((row) => row[j])).flat();
const spearmanRow = (s: any) => [s.rho2, s.f, s.df1, s.df2, s.pValue, s.n, s.adjustedRho2];

describe('spearman2, hoeffdingD, hclust, varclus', () => {
  const [N, P] = [80, 6];
  const col = (j: number) => Array.from({ length: N }, (_, i) => R.RD_X[i * P + j]);

  it('spearman2 matches Hmisc', () => {
    for (const [p, want] of [[1, R.RD_SP1], [2, R.RD_SP2]] as const) {
      for (let j = 0; j < P; j++) {
        allNear(spearmanRow(wasm.spearman2(col(j), R.RD_Y, p)), want.slice(7 * j, 7 * j + 7), 1e-10, `p=${p} x${j}`);
      }
    }
    allNear(spearmanRow(wasm.spearman2Categorical(new Uint32Array(col(5)), R.RD_Y)), R.RD_SPF, 1e-10, 'factor');
    expect(() => wasm.spearman2(col(0), R.RD_Y, 3)).toThrow();
  });

  it('hoeffdingD matches Hmisc', () => {
    const h = wasm.hoeffdingD(R.RD_X, N, P);
    allNear(colMajor(h.d), R.RD_HOEFFD_D, 1e-12, 'D');
    allNear(colMajor(h.pValue).map((v: number) => (Number.isNaN(v) ? -1 : v)), R.RD_HOEFFD_P, 1e-10, 'P');
  });

  const tree = (tag: string, h: any, merge: number[], height: number[], order: number[]) => {
    const r = h.getResult();
    expect(r.merge.flat()).toEqual(merge);
    allNear(r.height, height, 1e-12, `${tag} height`);
    expect(r.order.map((i: number) => i + 1)).toEqual(order);
  };

  it('varclus matches Hmisc', () => {
    for (const [s, sim, merge, height, order] of [
      ['spearman', R.RD_VC_SPEARMAN_SIM, R.RD_VC_SPEARMAN_MERGE, R.RD_VC_SPEARMAN_HEIGHT, R.RD_VC_SPEARMAN_ORDER],
      ['pearson', R.RD_VC_PEARSON_SIM, R.RD_VC_PEARSON_MERGE, R.RD_VC_PEARSON_HEIGHT, R.RD_VC_PEARSON_ORDER],
      ['hoeffding', R.RD_VC_HOEFFDING_SIM, R.RD_VC_HOEFFDING_MERGE, R.RD_VC_HOEFFDING_HEIGHT, R.RD_VC_HOEFFDING_ORDER],
    ] as const) {
      const v = wasm.varclus(R.RD_X, N, P, { similarity: s });
      allNear(colMajor(v.sim()), sim, 1e-12, `${s} sim`);
      tree(s, v.hclust(), merge, height, order);
    }
    const v = wasm.varclus(R.RD_X, N, P, { transform: 'abs', linkage: 'average' }).hclust().getResult();
    expect(v.merge.flat()).toEqual(R.RD_VC_ABS_AVG_MERGE);
    allNear(v.height, R.RD_VC_ABS_AVG_HEIGHT, 1e-12, 'abs average');
    expect(() => wasm.varclus(R.RD_X, N, P, { similarity: 'kendall' })).toThrow(/similarity/);
  });

  it('hclust matches R for every linkage', () => {
    const cases = [
      ['ward.D', R.RD_HC_WARDD_MERGE, R.RD_HC_WARDD_HEIGHT, R.RD_HC_WARDD_ORDER],
      ['single', R.RD_HC_SINGLE_MERGE, R.RD_HC_SINGLE_HEIGHT, R.RD_HC_SINGLE_ORDER],
      ['complete', R.RD_HC_COMPLETE_MERGE, R.RD_HC_COMPLETE_HEIGHT, R.RD_HC_COMPLETE_ORDER],
      ['average', R.RD_HC_AVERAGE_MERGE, R.RD_HC_AVERAGE_HEIGHT, R.RD_HC_AVERAGE_ORDER],
      ['mcquitty', R.RD_HC_MCQUITTY_MERGE, R.RD_HC_MCQUITTY_HEIGHT, R.RD_HC_MCQUITTY_ORDER],
      ['median', R.RD_HC_MEDIAN_MERGE, R.RD_HC_MEDIAN_HEIGHT, R.RD_HC_MEDIAN_ORDER],
      ['centroid', R.RD_HC_CENTROID_MERGE, R.RD_HC_CENTROID_HEIGHT, R.RD_HC_CENTROID_ORDER],
      ['ward.D2', R.RD_HC_WARDD2_MERGE, R.RD_HC_WARDD2_HEIGHT, R.RD_HC_WARDD2_ORDER],
    ] as const;
    for (const [l, merge, height, order] of cases) tree(l, wasm.hclust(R.RD_HC_DIST, 8, l), merge, height, order);
    const h = wasm.hclust(R.RD_HC_DIST, 8);
    expect(Array.from(h.cutHeight(Infinity))).toEqual(new Array(8).fill(0));
    expect(Math.max(...h.cutHeight(h.getResult().height[2])) + 1).toBe(5);
    expect(() => wasm.hclust(R.RD_HC_DIST, 8, 'ward')).toThrow(/linkage/);
  });
});

describe('principal components', () => {
  const [N, P] = [80, 5];
  /** Sign turning R's column c (column-major, `rows` rows) into ours. */
  const sign = (ours: number[][], r: number[], rows: number, c: number) =>
    Math.sign(ours.reduce((s, row, i) => s + row[c] * r[c * rows + i], 0));

  it('match prcomp', () => {
    for (const [scale, sdev, rot, scores] of [
      [true, R.PC_SC_SDEV, R.PC_SC_ROT, R.PC_SC_SCORES],
      [false, R.PC_UN_SDEV, R.PC_UN_ROT, R.PC_UN_SCORES],
    ] as const) {
      const pc = wasm.principalComponents(R.PC_X, N, P, scale);
      const r = pc.getResult();
      allNear(r.sdev, sdev, 1e-12, 'sdev');
      for (let c = 0; c < P; c++) {
        const s = sign(r.rotation, rot, P, c);
        allNear(r.rotation.map((row: number[]) => row[c]), rot.slice(c * P, c * P + P).map((v) => s * v), 1e-10, `rot ${c}`);
        allNear(r.scores.map((row: number[]) => row[c]), scores.slice(c * N, c * N + N).map((v) => s * v), 1e-10, `scores ${c}`);
      }
      const total = sdev.reduce((a, v) => a + v * v, 0);
      near(pc.proportionOfVariance()[0], (sdev[0] * sdev[0]) / total, 1e-12, 'pve');
      if (scale) {
        const pred = pc.transform(R.PC_NEW, 3, 2);
        for (let c = 0; c < 2; c++) {
          const s = sign(r.rotation, rot, P, c);
          for (let i = 0; i < 3; i++) near(pred[2 * i + c], s * R.PC_SC_PRED[c * 3 + i], 1e-10, `pred ${i},${c}`);
        }
      }
    }
  });

  it('cluster scores are first components', () => {
    const s = wasm.clusterScores(R.PC_X, N, P, [[0, 1], [2, 3], [4]], 1);
    expect(s.length).toBe(N * 3);
    for (let c = 0; c < 3; c++) {
      const ours = Array.from({ length: N }, (_, i) => s[3 * i + c]);
      const want = R.PC_CL_SCORES.slice(c * N, c * N + N);
      const sg = Math.sign(ours.reduce((a, v, i) => a + v * want[i], 0));
      allNear(ours, want.map((v) => sg * v), 1e-10, `cluster ${c}`);
    }
    expect(() => wasm.clusterScores(R.PC_X, N, P, [[P]], 1)).toThrow();
  });
});

describe('transcan and redun', () => {
  it('transcan matches Hmisc, with and without missing values', () => {
    const [N, P] = [150, 5];
    const cat = new Uint8Array([0, 0, 0, 1, 0]);
    for (const [x, trans, rsq, rsqAdj] of [
      [R.TC_X, R.TC_FULL_TRANS, R.TC_FULL_RSQ, R.TC_FULL_RSQADJ],
      [R.TC_XM, R.TC_MISS_TRANS, R.TC_MISS_RSQ, R.TC_MISS_RSQADJ],
    ] as const) {
      const t = wasm.transcan(x, N, P, cat);
      for (let j = 0; j < P; j++) {
        const ours = t.transformed.map((row: number[]) => row[j]);
        const want = trans.slice(j * N, j * N + N);
        const s = Math.sign(ours.reduce((a: number, v: number, i: number) => a + v * want[i], 0));
        allNear(ours, want.map((v) => s * v), 1e-6, `trans ${j}`);
      }
      allNear(t.rSquared, rsq, 1e-8, 'rsq');
      allNear(t.rSquaredAdj, rsqAdj, 1e-8, 'rsq adj');
    }
    expect(() => wasm.transcan(R.TC_X, 150, 5, new Uint8Array(2))).toThrow();
  });

  it('redun matches Hmisc', () => {
    const [N, P] = [200, 7];
    const kinds = (linA2: boolean) => ['numeric', 'numeric', 'numeric', linA2 ? 'linear' : 'numeric', 'categorical', 'numeric', 'numeric'];
    const cases = [
      ['DEF', { r2: 0.8 }, false],
      ['ADJ', { r2: 0.8, adjusted: true, nk: 4 }, false],
      ['NOTL', { r2: 0.7, tlinear: false }, false],
      ['ALLCAT', { r2: 0.5, allcat: true }, false],
      ['RANK', { r2: 0.7, rank: true }, false],
      ['QRANK', { r2: 0.7, qrank: true }, false],
      ['LIN', { r2: 0.8 }, true],
    ] as const;
    for (const [tag, opts, lin] of cases) {
      const r = wasm.redun(R.RU_X, N, P, kinds(lin), opts);
      const ref = (k: string) => (R as any)[`RU_${tag}_${k}`];
      expect(r.removed).toEqual(ref('OUT'));
      expect(r.kept).toEqual(ref('IN'));
      allNear(r.rSquared, ref('RSQ'), 1e-8, `${tag} rsq`);
      allNear(r.rSquaredFirst, ref('RSQ1'), 1e-8, `${tag} rsq1`);
      allNear(r.rSquaredLater.flat(), ref('LATER'), 1e-8, `${tag} later`);
      expect(r.tooFew).toEqual([]);
    }
    expect(() => wasm.redun(R.RU_X, N, P, kinds(false).map(() => 'ordinal'))).toThrow(/kind/);
  });
});

describe('Box–Cox and smearing', () => {
  const N = 120;
  const ols = (g: (v: number) => number) => {
    const gy = new Float64Array(R.TF_Y.map(g));
    const fit = new wasm.OlsRegressor().fit(R.TF_X, N, 2, gy);
    const lp = fit.predict(R.TF_X, N);
    return { lp6: Array.from(lp.slice(0, 6)) as number[], r: Array.from(gy, (v, i) => v - lp[i]) };
  };

  it('Box–Cox profile matches MASS::boxcox', () => {
    const p = wasm.boxCox(R.TF_X, N, 2, R.TF_Y);
    allNear(p.logLikelihood, R.TF_BC_GRID_LOGLIK, 1e-12, 'grid');
    allNear([p.lambdaHat], [R.TF_BC_MLE[0]], 1e-7, 'lambda');
    allNear([p.maxLogLikelihood, p.ciLower, p.ciUpper], R.TF_BC_MLE.slice(1), 1e-9, 'mle');
    allNear(wasm.boxCox(R.TF_X, N, 2, R.TF_Y, { lambdas: [-0.015, 0, 0.01] }).logLikelihood, R.TF_BC_SMALL, 1e-12, 'small');
    const x1 = new Float64Array(N * 3);
    for (let i = 0; i < N; i++) x1.set([R.TF_X[2 * i], R.TF_X[2 * i + 1], 1], 3 * i);
    allNear(wasm.boxCox(x1, N, 3, R.TF_Y, { lambdas: [0.3, 0.5], withIntercept: false }).logLikelihood,
      R.TF_BC_NOINT_CHECK, 1e-12, 'no intercept');
    const open = wasm.boxCox(R.TF_X, N, 2, R.TF_Y, { lambdas: [0.45, 0.5, 0.55] });
    expect(open.ciLower).toBeUndefined();
    expect(() => wasm.boxCox(R.TF_X, N, 2, R.TF_Y, { confidenceLevel: 1 })).toThrow();
  });

  it('smearing matches Hmisc::smearingEst', () => {
    const { lp6, r } = ols(Math.sqrt);
    allNear(lp6, R.TF_SM_LP, 1e-10, 'lp');
    const sq = (z: number) => z * z;
    allNear(wasm.smearingEstimate(lp6, r, sq), R.TF_SM_MEAN, 1e-10, 'mean');
    allNear(wasm.smearingEstimate(lp6, r, sq, 'median'), R.TF_SM_MEDIAN, 1e-10, 'median');
    allNear(wasm.smearingEstimate(lp6, r, sq, 0.9), R.TF_SM_Q90, 1e-10, 'q90');
    allNear(wasm.smearingEstimate(lp6, r, sq, 'fitted'), R.TF_SM_FITTED, 1e-10, 'fitted');
    const lg = ols(Math.log);
    allNear(wasm.smearingEstimate(lg.lp6, lg.r, Math.exp, 'mean'), R.TF_SM_LOG_MEAN, 1e-10, 'log');
    const bc = ols((v) => wasm.boxCoxTransform(v, 0.3));
    allNear(wasm.smearingEstimate(bc.lp6, bc.r, (z: number) => wasm.boxCoxInverse(z, 0.3)), R.TF_SM_BC_MEAN, 1e-10, 'box-cox');
    near(wasm.boxCoxInverse(wasm.boxCoxTransform(3.7, 0), 0), 3.7, 1e-12, 'round trip');
    expect(() => wasm.smearingEstimate(lp6, r, () => { throw new Error('nope'); })).toThrow(/nope/);
    expect(() => wasm.smearingEstimate(lp6, r, sq, 1.5)).toThrow();
  });
});

describe('areg', () => {
  const [N, P] = [150, 3];
  const smooth = { nk: 4, xTypes: ['spline', 'spline', 'categorical'], yType: 'spline' };

  it('smooth response matches Hmisc::areg', () => {
    const f = wasm.areg(R.AR_X, N, P, R.AR_Y, smooth);
    const r = f.getResult();
    allNear(r.xCoefficients, R.AR_S_XCOEF, 1e-8, 'xcoef');
    allNear(r.yCoefficients, R.AR_S_YCOEF, 1e-8, 'ycoef');
    allNear([r.rSquared, r.mad, r.med], R.AR_S_STATS, 1e-8, 'stats');
    allNear(r.transformedY, R.AR_S_TY, 1e-8, 'ty');
    allNear(r.linearPredictors, R.AR_S_LP, 1e-8, 'lp');
    allNear(r.transformedX.flat(), R.AR_S_TX, 1e-8, 'tx');
    expect([r.xDf, r.yDf]).toEqual([[3, 3, 2], 3]);
    allNear(f.yKnots(), R.AR_S_YKNOTS, 1e-12, 'y knots');
    allNear(f.transformY(R.AR_S_YTRANS_AT), R.AR_S_YTRANS, 1e-8, 'ytrans');
    allNear(f.inverseY(R.AR_S_LP.slice(0, 8)), R.AR_S_YINV, 1e-8, 'yinv');
    allNear(f.predict(R.AR_X, N), R.AR_S_LP, 1e-8, 'predict');
    const b = R.AR_BOOT_IDX.length / N;
    const idx = Array.from({ length: b }, (_, k) => R.AR_BOOT_IDX.slice(k * N, (k + 1) * N));
    const boot = f.bootstrap(idx);
    allNear([boot.r2Optimism, boot.r2Corrected, boot.madOptimism, boot.madCorrected, boot.medOptimism, boot.medCorrected],
      R.AR_BOOT_STATS, 1e-8, 'bootstrap');
  });

  it('linear response is least squares; bad input rejected', () => {
    const f = wasm.areg(R.AR_X, N, P, R.AR_Y, { nk: 5, xTypes: ['s', 'l', 'c'], yType: 'l' }).getResult();
    allNear(f.xCoefficients, R.AR_L_XCOEF, 1e-8, 'xcoef');
    expect(f.yCoefficients).toEqual([0, 1]);
    allNear([f.rSquared, f.mad, f.med], R.AR_L_STATS, 1e-8, 'stats');
    expect(() => wasm.areg(R.AR_X, N, P, R.AR_Y, { xTypes: ['s'] })).toThrow();
    expect(() => wasm.areg(R.AR_X, N, P, R.AR_Y, { yType: 'q' })).toThrow(/type/);
  });
});

describe('aregImpute', () => {
  const [N, P] = [120, 5];
  const kinds = ['numeric', 'numeric', 'numeric', 'numeric', 'categorical'];
  const check = (tag: string, k: string[], opts: any, x1: number[], x2: number[], g: number[], rsq: number[]) => {
    const a = wasm.aregImpute(R.AI_X, N, P, k, opts);
    const r = a.getResult();
    allNear(r.imputed[1].flat(), x1, 1e-12, `${tag} x1`);
    allNear(r.imputed[2].flat(), x2, 1e-12, `${tag} x2`);
    allNear(r.imputed[4].flat(), g, 1e-12, `${tag} g`);
    allNear([1, 2, 4].map((c) => r.rSquared[c]), rsq, 1e-8, `${tag} R²`);
    const c = a.complete(R.AI_X, N, 0);
    expect(c.some((v: number) => Number.isNaN(v))).toBe(false);
    expect(c[0]).toBe(R.AI_X[0]);
    expect(() => a.complete(R.AI_X, N, 99)).toThrow(/range/);
  };

  it('replays aregImpute (weighted, closest, I(x1), tlinear, burnin)', () => {
    check('weighted', kinds, { seed: 7 }, R.AI_W_X1, R.AI_W_X2, R.AI_W_G, R.AI_W_RSQ);
    check('closest', kinds, { nImpute: 3, matching: 'closest', seed: 8 }, R.AI_C_X1, R.AI_C_X2, R.AI_C_G, R.AI_C_RSQ);
    const lin = [...kinds];
    lin[1] = 'linear';
    check('I(x1)', lin, { nImpute: 3, seed: 9 }, R.AI_L_X1, R.AI_L_X2, R.AI_L_G, R.AI_L_RSQ);
    check('tlinear', kinds, { nImpute: 3, tlinear: false, seed: 10 }, R.AI_T_X1, R.AI_T_X2, R.AI_T_G, R.AI_T_RSQ);
    check('burnin', kinds, { nImpute: 2, burnin: 1, seed: 11 }, R.AI_B_X1, R.AI_B_X2, R.AI_B_G, R.AI_B_RSQ);
    expect(() => wasm.aregImpute(R.AI_X, N, P, kinds.slice(0, 2))).toThrow();
    expect(() => wasm.aregImpute(R.AI_X, N, P, kinds, { matching: 'random' })).toThrow(/matching/);
  });
});
