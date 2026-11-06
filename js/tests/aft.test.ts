// #287: AftRegressor against survival::survreg
// (tests/r_validation_penalized_glm_aft.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';
import { near } from './rel';
import * as R from './references';

describe('AftRegressor', () => {
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
  const event = new Uint8Array(R.PG_EVENT.map((e) => (e === 1 ? 1 : 0)));

  const cases: [string, number[], number, number[], number[], number[]][] = [
    ['weibull', R.PG_AFT_WEIBULL_COEF, R.PG_AFT_WEIBULL_SCALE, R.PG_AFT_WEIBULL_LL, R.PG_AFT_WEIBULL_SE, R.PG_AFT_WEIBULL_Q],
    ['lognormal', R.PG_AFT_LOGNORMAL_COEF, R.PG_AFT_LOGNORMAL_SCALE, R.PG_AFT_LOGNORMAL_LL, R.PG_AFT_LOGNORMAL_SE, R.PG_AFT_LOGNORMAL_Q],
    ['loglogistic', R.PG_AFT_LOGLOGISTIC_COEF, R.PG_AFT_LOGLOGISTIC_SCALE, R.PG_AFT_LOGLOGISTIC_LL, R.PG_AFT_LOGLOGISTIC_SE, R.PG_AFT_LOGLOGISTIC_Q],
    ['exponential', R.PG_AFT_EXPONENTIAL_COEF, R.PG_AFT_EXPONENTIAL_SCALE, R.PG_AFT_EXPONENTIAL_LL, R.PG_AFT_EXPONENTIAL_SE, R.PG_AFT_EXPONENTIAL_Q],
  ];

  for (const [dist, coef, scale, ll, se, q] of cases) {
    it(`${dist} matches survreg`, () => {
      const m = new wasm.AftRegressor();
      m.setDistribution(dist);
      m.setComputeInference(true);
      const f = m.fit(x, n, 2, vector(R.PG_TIME), event);
      expect(f.getConverged()).toBe(true);
      expect(f.getIterations()).toBeGreaterThan(0);
      const r = f.getResult();
      expect(r.distribution).toBe(dist);
      near(f.getIntercept(), coef[0], 1e-6, 'intercept');
      near(f.getCoefficients()[0], coef[1], 1e-6, 'b1');
      near(f.getCoefficients()[1], coef[2], 1e-6, 'b2');
      near(f.getScale(), scale, 1e-6, 'scale');
      near(r.nullLogLikelihood, ll[0], 1e-7, 'null logLik');
      near(f.getLogLikelihood(), ll[1], 1e-7, 'logLik');
      near(r.inference.interceptStdError, se[0], 1e-5, 'se b0');
      near(r.inference.stdErrors[0], se[1], 1e-5, 'se b1');
      near(r.inference.stdErrors[1], se[2], 1e-5, 'se b2');
      if (se.length === 4) near(r.inference.logScaleStdError, se[3], 1e-5, 'se log scale');
      else expect(r.inference.logScaleStdError).toBeUndefined();
      expect(r.nEvents + r.nCensored).toBe(n);
      const x0 = vector([0.3, -0.2]);
      [0.25, 0.5, 0.9].forEach((p, k) => {
        const qp = f.predictQuantile(x0, 1, p)[0];
        near(qp, q[k], 1e-6, `q${p}`);
        near(f.predictSurvival(x0, 1, qp)[0], 1 - p, 1e-9, `S(q${p})`);
        near(f.predictCdf(x0, 1, qp)[0], p, 1e-9, `F(q${p})`);
      });
      near(f.predictMedian(x0, 1)[0], q[1], 1e-6, 'median');
    });
  }

  it('fromParts predicts as the fit (0.12.1)', () => {
    const f = new wasm.AftRegressor().fit(x, n, 2, vector(R.PG_TIME), event);
    const r = f.getResult();
    const g = wasm.FittedAft.fromParts(r.distribution, r.intercept, vector(r.coefficients), r.scale, undefined);
    expect(Array.from(g.predictMedian(x, n))).toEqual(Array.from(f.predictMedian(x, n)));
  });

  it('residuals and score contributions have the fit dimensions', () => {
    const f = new wasm.AftRegressor().fit(x, n, 2, vector(R.PG_TIME), event);
    const z = f.standardizedResiduals(x, n, vector(R.PG_TIME));
    const cs = f.coxSnellResiduals(x, n, vector(R.PG_TIME));
    expect(z.length).toBe(n);
    expect(cs.every((v: number) => v >= 0)).toBe(true);
    const eta = f.predictLinear(x, n);
    near(z[0], (Math.log(R.PG_TIME[0]) - eta[0]) / f.getScale(), 1e-12, 'z0');
    // intercept + 2 coefficients + log scale; scores sum to ~0 at the MLE
    const s = f.scoreContributions(x, n, vector(R.PG_TIME), event);
    expect(s.length).toBe(n * 4);
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += s[i * 4 + j];
      expect(Math.abs(sum)).toBeLessThan(1e-4);
    }
  });

  it('priors shrink the coefficients; bad options throw', () => {
    const m = new wasm.AftRegressor();
    m.setPriors([{ kind: 'flat' }, { kind: 'normal', loc: 0, scale: 0.01 }, { kind: 'normal', loc: 0, scale: 0.01 }]);
    const f = m.fit(x, n, 2, vector(R.PG_TIME), event);
    expect(Math.abs(f.getCoefficients()[0])).toBeLessThan(0.05);
    expect(() => m.setDistribution('gompertz')).toThrow();
    expect(() => m.setVcovType('bogus')).toThrow();
    expect(() => m.setPriors([{ kind: 'normal' }])).toThrow();
    expect(() => m.fit(x, n, 2, vector(R.PG_TIME), new Uint8Array(n - 1))).toThrow();
  });
});
