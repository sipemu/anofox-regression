// #287: GlsRegressor against nlme::gls (tests/r_validation_gls.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { allNear, near } from './rel';
import * as R from './references';

const N = 164;
const P = 4;

describe('GlsRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  const x = vector(R.GLS_X);
  const y = vector(R.GLS_Y);
  const id = new Uint32Array(R.GLS_ID);
  const visits = new Uint32Array(R.GLS_VISIT);

  function fit(configure: (m: any) => void, time?: number[]) {
    const m = new wasm.GlsRegressor();
    configure(m);
    return m.fit(x, N, P, y, id, time ? vector(time) : undefined);
  }

  function check(
    tag: string,
    f: any,
    coef: number[],
    vcov: number[],
    sigma: number[],
    ll: number[],
    cor: number[],
    vr: number[],
    loose = false,
  ) {
    const [tol, tolPar] = loose ? [1e-4, 1e-3] : [1e-7, 1e-6];
    const r = f.getResult();
    allNear(r.coefficients, coef, tol, `${tag} coef`);
    allNear(r.vcov.flat(), vcov, tol, `${tag} vcov`);
    near(r.sigma, sigma[0], tol, `${tag} sigma`);
    allNear([r.logLikelihood, r.aic, r.bic], ll, 1e-7, `${tag} logLik / AIC / BIC`);
    allNear(r.correlationParams, cor, tolPar, `${tag} cor`);
    allNear(r.varianceParams, vr, tolPar, `${tag} var`);
    return r;
  }

  function checkIntervals(tag: string, r: any, expected: number[], tol = 2e-3) {
    const iv = r.varCovIntervals;
    const got = [...iv.correlation, ...iv.variance, iv.sigma].flatMap((i: any) => [i.lower, i.estimate, i.upper]);
    allNear(got, expected, tol, `${tag} intervals`);
  }

  it('continuous AR(1) matches nlme, with residuals, predictions and variogram', () => {
    const f = fit((m) => m.setCorrelation('car1'), R.GLS_TIME);
    const r = check('CAR1', f, R.GLS_CAR1_COEF, R.GLS_CAR1_VCOV, R.GLS_CAR1_SIGMA, R.GLS_CAR1_LOGLIK, R.GLS_CAR1_COR, []);
    checkIntervals('CAR1', r, R.GLS_CAR1_INTERVALS);
    allNear(f.residuals('pearson'), R.GLS_CAR1_PEARSON, 1e-4, 'pearson');
    allNear(f.residuals('normalized'), R.GLS_CAR1_NORMALIZED, 1e-4, 'normalized');
    const nd = [
      [1, 0, 0],
      [3, 1, 0.5],
      [5, 1, -1],
    ].flatMap(([t, trt, xv]) => [t, trt, xv, t * trt]);
    const pr = f.predictWithSe(vector(nd), 3);
    allNear(pr.fit, R.GLS_CAR1_PREDICT, 1e-4, 'predict');
    allNear(f.predict(vector(nd), 3), pr.fit, 1e-14, 'predict = predictWithSe');
    expect(pr.se.every((s: number) => s > 0)).toBe(true);
    const v = f.variogram(vector(R.GLS_TIME), 'normalized', 20);
    allNear(v.flatMap((b: any) => [b.variog, b.dist, b.nPairs]), R.GLS_CAR1_VARIOGRAM, 1e-4, 'variogram');
  });

  it('AR(1) by ML matches nlme', () => {
    const f = fit((m) => {
      m.setCorrelation('ar1');
      m.setMethod('ml');
    }, R.GLS_VISIT);
    const r = check('AR1', f, R.GLS_AR1_ML_COEF, R.GLS_AR1_ML_VCOV, R.GLS_AR1_ML_SIGMA, R.GLS_AR1_ML_LOGLIK, R.GLS_AR1_ML_COR, []);
    checkIntervals('AR1', r, R.GLS_AR1_ML_INTERVALS);
  });

  it('compound symmetry with varIdent matches nlme', () => {
    const f = fit((m) => {
      m.setCorrelation('compound_symmetry');
      m.setVarIdent(visits);
    });
    const r = check(
      'CS', f, R.GLS_CS_IDENT_COEF, R.GLS_CS_IDENT_VCOV, R.GLS_CS_IDENT_SIGMA,
      R.GLS_CS_IDENT_LOGLIK, R.GLS_CS_IDENT_COR, R.GLS_CS_IDENT_VAR, true,
    );
    checkIntervals('CS', r, R.GLS_CS_IDENT_INTERVALS);
  });

  it('varPower and varExp match nlme', () => {
    let f = fit((m) => m.setVarPower(vector(R.GLS_AX)));
    const r = check('POWER', f, R.GLS_POWER_COEF, R.GLS_POWER_VCOV, R.GLS_POWER_SIGMA, R.GLS_POWER_LOGLIK, [], R.GLS_POWER_VAR);
    checkIntervals('POWER', r, R.GLS_POWER_INTERVALS);
    const xcol = Array.from({ length: N }, (_, i) => R.GLS_X[i * P + 2]);
    f = fit((m) => {
      m.setVarExp(vector(xcol));
      m.setMethod('ml');
    });
    check('VEXP', f, R.GLS_VEXP_ML_COEF, R.GLS_VEXP_ML_VCOV, R.GLS_VEXP_ML_SIGMA, R.GLS_VEXP_ML_LOGLIK, [], R.GLS_VEXP_ML_VAR);
  });

  it('independent errors are OLS', () => {
    const f = fit(() => {});
    const r = check('INDEP', f, R.GLS_INDEP_COEF, R.GLS_INDEP_VCOV, R.GLS_INDEP_SIGMA, R.GLS_INDEP_LOGLIK, [], []);
    checkIntervals('INDEP', r, R.GLS_INDEP_INTERVALS);
    expect(r.nParams).toBe(P + 2);
    expect(r.confIntervalLower[1]).toBeLessThan(r.coefficients[1]);
    expect(r.confIntervalUpper[1]).toBeGreaterThan(r.coefficients[1]);
    expect(r.pValues.every((p: number) => p >= 0 && p <= 1)).toBe(true);
    allNear(f.getCoefficients(), r.coefficients, 0, 'getCoefficients');
  });

  it('rejects bad input and unknown options', () => {
    const m = new wasm.GlsRegressor();
    expect(() => m.fit(x, N, P, y, new Uint32Array(N - 1))).toThrow();
    expect(() => m.setCorrelation('toeplitz')).toThrow();
    expect(() => m.setMethod('gee')).toThrow();
    m.setCorrelation('compound_symmetry');
    expect(() => m.fit(x, N, P, y, new Uint32Array(Array.from({ length: N }, (_, i) => i)))).toThrow();
    const f = fit(() => {});
    expect(() => f.residuals('deviance')).toThrow();
  });
});
