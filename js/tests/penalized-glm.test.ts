// #287: PenalizedGlmRegressor against glm, MASS::glm.nb and an R
// posterior-mode fit (tests/r_validation_penalized_glm_aft.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { near } from './rel';
import * as R from './references';

describe('PenalizedGlmRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  const n = R.PG_X1.length;
  const x = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    x[2 * i] = R.PG_X1[i];
    x[2 * i + 1] = R.PG_X2[i];
  }

  function fit(family: string, y: number[], configure?: (m: any) => void, link?: string, param?: number) {
    const m = new wasm.PenalizedGlmRegressor();
    m.setFamily(family, link, param);
    m.setComputeInference(true);
    m.setTolerance(1e-10);
    configure?.(m);
    return m.fit(x, n, 2, vector(y));
  }

  function check(f: any, coef: number[], se: number[], dev: number[], llAic: number[], what: string) {
    expect(f.getConverged(), what).toBe(true);
    const r = f.getResult();
    const shift = coef.length - r.coefficients.length;
    if (shift === 1) {
      near(f.getIntercept(), coef[0], 1e-6, `${what} b0`);
      near(r.inference.interceptStdError, se[0], 1e-5, `${what} se b0`);
    } else {
      expect(f.getIntercept()).toBeUndefined();
    }
    r.coefficients.forEach((b: number, j: number) => {
      near(b, coef[j + shift], 1e-6, `${what} b${j + 1}`);
      near(r.inference.stdErrors[j], se[j + shift], 1e-5, `${what} se${j + 1}`);
    });
    near(f.getDeviance(), dev[0], 1e-7, `${what} deviance`);
    near(r.nullDeviance, dev[1], 1e-7, `${what} null deviance`);
    near(r.logLikelihood, llAic[0], 1e-7, `${what} logLik`);
    near(r.aic, llAic[1], 1e-7, `${what} AIC`);
    return r;
  }

  it('Poisson (log, sqrt, offset, no intercept) matches glm', () => {
    let r = check(fit('poisson', R.PG_Y_POIS), R.PG_POIS_COEF, R.PG_POIS_SE, R.PG_POIS_DEV, R.PG_POIS_LL_AIC, 'poisson');
    expect([r.family, r.link]).toEqual(['poisson', 'log']);
    check(fit('poisson', R.PG_Y_POIS, undefined, 'sqrt'), R.PG_POIS_SQRT_COEF, R.PG_POIS_SQRT_SE, R.PG_POIS_SQRT_DEV, R.PG_POIS_SQRT_LL_AIC, 'sqrt');
    const off = fit('poisson', R.PG_Y_POIS, (m) => m.setOffset(vector(R.PG_OFFSET)));
    check(off, R.PG_POIS_OFF_COEF, R.PG_POIS_OFF_SE, R.PG_POIS_OFF_DEV, R.PG_POIS_OFF_LL_AIC, 'offset');
    const mu = off.predictWithOffset(x, n, vector(R.PG_OFFSET));
    off.getResult().fittedValues.forEach((m: number, i: number) => near(mu[i], m, 1e-12, 'offset prediction'));
    r = check(
      fit('poisson', R.PG_Y_POIS, (m) => m.setWithIntercept(false)),
      R.PG_POIS_NOINT_COEF, R.PG_POIS_NOINT_SE, R.PG_POIS_NOINT_DEV, R.PG_POIS_NOINT_LL_AIC, 'no intercept',
    );
    expect(r.intercept).toBeUndefined();
  });

  it('binomial links, logistic, gamma and Tweedie match glm', () => {
    const links: [string, number[], number[], number[], number[]][] = [
      ['logit', R.PG_BIN_LOGIT_COEF, R.PG_BIN_LOGIT_SE, R.PG_BIN_LOGIT_DEV, R.PG_BIN_LOGIT_LL_AIC],
      ['probit', R.PG_BIN_PROBIT_COEF, R.PG_BIN_PROBIT_SE, R.PG_BIN_PROBIT_DEV, R.PG_BIN_PROBIT_LL_AIC],
      ['cloglog', R.PG_BIN_CLOGLOG_COEF, R.PG_BIN_CLOGLOG_SE, R.PG_BIN_CLOGLOG_DEV, R.PG_BIN_CLOGLOG_LL_AIC],
    ];
    for (const [link, coef, se, dev, ll] of links) {
      const r = check(fit('binomial', R.PG_Y_BIN, undefined, link), coef, se, dev, ll, `binomial ${link}`);
      expect(r.separation.hasSeparation).toBe(false);
    }
    const lg = check(fit('logistic', R.PG_Y_BIN), R.PG_BIN_LOGIT_COEF, R.PG_BIN_LOGIT_SE, R.PG_BIN_LOGIT_DEV, R.PG_BIN_LOGIT_LL_AIC, 'logistic');
    expect(lg.accuracy).toBeGreaterThanOrEqual(0.5);
    const g = fit('gamma', R.PG_Y_GAMMA);
    check(g, R.PG_GAMMA_COEF, R.PG_GAMMA_SE, R.PG_GAMMA_DEV, R.PG_GAMMA_LL_AIC, 'gamma');
    const t = fit('tweedie', R.PG_Y_GAMMA, undefined, undefined, 2.0);
    t.getCoefficients().forEach((b: number, j: number) => near(b, g.getCoefficients()[j], 1e-6, 'tweedie p=2'));
  });

  it('negative binomial matches MASS::glm.nb, estimated and fixed theta', () => {
    const f = fit('negative_binomial', R.PG_Y_NB);
    check(f, R.PG_NB_COEF, R.PG_NB_SE, R.PG_NB_DEV, R.PG_NB_LL_AIC, 'glm.nb');
    near(f.getResult().theta, R.PG_NB_THETA, 1e-5, 'theta');
    const fixed = fit('negative_binomial', R.PG_Y_NB, undefined, undefined, 2.5);
    const r = fixed.getResult();
    near(r.theta, 2.5, 1e-12, 'fixed theta');
    [0, 1].forEach((j) => {
      near(r.coefficients[j], R.PG_NB_FIXED_COEF[j + 1], 1e-6, 'nb fixed');
      near(r.inference.stdErrors[j], R.PG_NB_FIXED_SE[j + 1], 1e-5, 'nb fixed se');
    });
  });

  it('Gaussian priors give the posterior mode and the three covariances', () => {
    const priors = [{ kind: 'flat' }, { kind: 'normal', loc: 0.5, scale: 0.5 }, { kind: 'normal', loc: 0, scale: 0.25 }];
    const cases: [string, number[]][] = [
      ['laplace', R.PG_POIS_PRIOR_SE_LAPLACE],
      ['sandwich', R.PG_POIS_PRIOR_SE_SANDWICH],
      ['naive', R.PG_POIS_PRIOR_SE_NAIVE],
    ];
    for (const [vcov, se] of cases) {
      const f = fit('poisson', R.PG_Y_POIS, (m) => {
        m.setPriors(priors);
        m.setVcovType(vcov);
        m.setTolerance(1e-12);
      });
      near(f.getIntercept(), R.PG_POIS_PRIOR_COEF[0], 1e-7, 'MAP b0');
      near(f.getCoefficients()[0], R.PG_POIS_PRIOR_COEF[1], 1e-7, 'MAP b1');
      near(f.getCoefficients()[1], R.PG_POIS_PRIOR_COEF[2], 1e-7, 'MAP b2');
      const v = f.getResult().inference.vcov;
      [0, 1, 2].forEach((j) => near(Math.sqrt(v[j][j]), se[j], 1e-6, `${vcov} se ${j}`));
    }
  });

  it('lambda shrinks, quasi-Poisson scales the SEs, predictions agree, bad options throw', () => {
    const plain = fit('poisson', R.PG_Y_POIS);
    const ridge = fit('poisson', R.PG_Y_POIS, (m) => m.setLambda(100));
    expect(Math.abs(ridge.getCoefficients()[0])).toBeLessThan(Math.abs(plain.getCoefficients()[0]));
    const quasi = fit('poisson', R.PG_Y_POIS, (m) => m.setEstimateDispersion(true));
    expect(quasi.getResult().dispersion).not.toBe(1);
    const eta = plain.predictLinear(x, n);
    const mu = plain.predict(x, n);
    near(mu[0], Math.exp(eta[0]), 1e-12, 'predict = exp(eta)');
    expect(plain.getIterations()).toBeGreaterThan(0);
    const m = new wasm.PenalizedGlmRegressor();
    expect(() => m.setFamily('gaussian')).toThrow();
    expect(() => m.setFamily('poisson', 'logit')).toThrow();
    expect(() => m.setFamily('tweedie')).toThrow();
  });
});
