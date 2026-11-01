// #258: offsets, prior weights and na.action through the WASM bindings,
// against the R references of the Rust validation (see references.ts).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';
import * as R from './references';

const col = (v: number[]) => ({ x: vector(v), nRows: v.length, nCols: 1 });
const cols2 = (a: number[], b: number[]) => {
  const x = new Float64Array(a.length * 2);
  a.forEach((v, i) => {
    x[2 * i] = v;
    x[2 * i + 1] = b[i];
  });
  return { x, nRows: a.length, nCols: 2 };
};
const rel = (got: number, want: number, tol: number) =>
  expectClose(got, want, tol * (1 + Math.abs(want)));

function expectPadded(got: ArrayLike<number>, want: number[], tol: number) {
  expect(got.length).toBe(want.length);
  want.forEach((w, i) => {
    if (Number.isNaN(w)) {
      expect(Number.isNaN(got[i])).toBe(true);
    } else {
      expectClose(got[i], w, tol);
    }
  });
}

describe('offsets, weights and na.action through WASM', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  const logExposure = R.EXPOSURE_POISSON_OFFSET_R.map(Math.log);

  it('Poisson offset matches glm(offset = log(exposure))', () => {
    const { x, nRows, nCols } = col(R.X_POISSON_OFFSET_R);
    const m = new wasm.PoissonRegressor();
    m.setOffset(vector(logExposure));
    m.setTolerance(1e-13);
    const f = m.fit(x, nRows, nCols, vector(R.Y_POISSON_OFFSET_R));
    const r = f.getResult();
    rel(r.intercept, R.POISSON_OFFSET[0], 1e-7);
    rel(r.coefficients[0], R.POISSON_OFFSET[1], 1e-7);
    rel(r.stdErrors[0], R.POISSON_OFFSET[3], 1e-7);
    rel(r.deviance, R.POISSON_OFFSET[4], 1e-7);
    // predictWithOffset adds the offset on the link scale
    const pred = f.predictWithOffset(vector([3]), 1, vector([Math.log(30)]));
    expectClose(pred[0], 30 * Math.exp(r.intercept + 3 * r.coefficients[0]), 1e-9);
    f.free();
    m.free();
  });

  it('Tweedie(p = 1) offset reproduces the Poisson glm', () => {
    const { x, nRows, nCols } = col(R.X_POISSON_OFFSET_R);
    const m = new wasm.TweedieRegressor();
    m.setVarPower(1);
    m.setLinkPower(0);
    m.setOffset(vector(logExposure));
    m.setTolerance(1e-13);
    const f = m.fit(x, nRows, nCols, vector(R.Y_POISSON_OFFSET_R));
    rel(f.getIntercept(), R.POISSON_OFFSET[0], 1e-7);
    rel(f.getCoefficients()[0], R.POISSON_OFFSET[1], 1e-7);
    rel(f.getDeviance(), R.POISSON_OFFSET[4], 1e-7);
    f.free();
    m.free();
  });

  it('negative binomial offset: null deviance matches glm(negative.binomial(1.5), offset = )', () => {
    // the null deviance is fitted with the offset, so it checks that it is used
    const { x, nRows, nCols } = cols2(R.ND190_X1, R.ND190_X2);
    const m = new wasm.NegativeBinomialRegressor();
    m.setTheta(1.5);
    m.setOffset(vector(R.ND190_OFF));
    m.setTolerance(1e-12);
    const f = m.fit(x, nRows, nCols, vector(R.ND190_Y));
    rel(f.getResult().nullDeviance, R.ND190_ND_NB_OFF, 1e-9);
    const pred = f.predictWithOffset(vector([R.ND190_X1[0], R.ND190_X2[0]]), 1, vector([R.ND190_OFF[0]]));
    expectClose(pred[0], f.getResult().fittedValues[0], 1e-10);
    f.free();
    m.free();
  });

  it('Gamma offset matches glm(Gamma("log"), offset = log(expo))', () => {
    const { x, nRows, nCols } = col(R.P5_X1);
    const m = new wasm.GammaRegressor();
    m.setOffset(vector(R.P5_EXPO.map(Math.log)));
    m.setTolerance(1e-13);
    const f = m.fit(x, nRows, nCols, vector(R.P5_Y_GAMMA_OFFSET));
    const r = f.getResult();
    rel(r.intercept, R.P5_GAMMA_OFFSET[0], 1e-7);
    rel(r.coefficients[0], R.P5_GAMMA_OFFSET[1], 1e-7);
    rel(r.stdErrors[0], R.P5_GAMMA_OFFSET[3], 1e-7);
    rel(f.getDeviance(), R.P5_GAMMA_OFFSET[8], 1e-7);
    f.free();
    m.free();
  });

  it('binomial trial weights match glm(cbind(s, m - s) ~ x)', () => {
    const { x, nRows, nCols } = col(R.B193_X);
    const p = R.B193_S.map((s, i) => s / R.B193_M[i]);
    for (const [link, aic] of [
      ['logit', R.B193_PROP_LOGIT_AIC],
      ['probit', R.B193_PROP_PROBIT_AIC],
    ] as const) {
      const m = new wasm.BinomialRegressor();
      m.setLink(link);
      m.setWeights(vector(R.B193_M));
      m.setTolerance(1e-12);
      const f = m.fit(x, nRows, nCols, vector(p));
      rel(f.getResult().aic, aic, 1e-8);
      if (link === 'logit') rel(f.getResult().bic, R.B193_PROP_LOGIT_BIC, 1e-8);
      f.free();
      m.free();
    }
  });

  it('binomial case weights on a 0/1 response match glm(weights = )', () => {
    const { x, nRows, nCols } = col(R.B193_X);
    const m = new wasm.BinomialRegressor();
    m.setWeights(vector(R.B193_WB));
    m.setTolerance(1e-12);
    const f = m.fit(x, nRows, nCols, vector(R.B193_YB));
    const r = f.getResult();
    rel(r.aic, R.B193_CASEW_AIC, 1e-8);
    rel(r.bic, R.B193_CASEW_BIC, 1e-8);
    f.free();
    m.free();
  });

  it('OLS na.exclude pads residuals and fitted values as lm(na.action = na.exclude)', () => {
    const { x, nRows, nCols } = cols2(R.NA205_X1, R.NA205_X2);
    const m = new wasm.OlsRegressor();
    m.setNaAction('exclude');
    const f = m.fit(x, nRows, nCols, vector(R.NA205_Y));
    const r = f.getResult();
    expect(r.nObservations).toBe(nRows - 3);
    expectPadded(r.residuals, R.NA205_OLS_RESID, 1e-10);
    expectPadded(r.fittedValues, R.NA205_OLS_FITTED, 1e-10);
    f.free();
    m.free();
  });

  it('Poisson na.exclude pads residuals and fitted values as glm(na.action = na.exclude)', () => {
    const { x, nRows, nCols } = cols2(R.NA205_X1, R.NA205_X2);
    const m = new wasm.PoissonRegressor();
    m.setNaAction('exclude');
    m.setTolerance(1e-12);
    const f = m.fit(x, nRows, nCols, vector(R.NA205_YC));
    const r = f.getResult();
    expectPadded(r.residuals, R.NA205_POIS_RESID, 1e-8);
    expectPadded(r.fittedValues, R.NA205_POIS_FITTED, 1e-8);
    f.free();
    m.free();
  });

  it('na.omit (default) drops the incomplete rows without padding', () => {
    const { x, nRows, nCols } = cols2(R.NA205_X1, R.NA205_X2);
    const m = new wasm.OlsRegressor();
    const f = m.fit(x, nRows, nCols, vector(R.NA205_Y));
    expect(f.getResult().residuals.length).toBe(nRows - 3);
    f.free();
    m.free();
  });

  it('na.fail errors on NaN, unknown actions are rejected', () => {
    const { x, nRows, nCols } = cols2(R.NA205_X1, R.NA205_X2);
    for (const name of ['OlsRegressor', 'WlsRegressor', 'RidgeRegressor', 'ElasticNetRegressor', 'GammaRegressor']) {
      const m = new wasm[name]();
      m.setNaAction('fail');
      const y = name === 'GammaRegressor' ? R.NA205_Y.map((v) => v + 1) : R.NA205_Y;
      expect(() => m.fit(x, nRows, nCols, vector(y))).toThrow();
      expect(() => m.setNaAction('drop')).toThrow();
      m.free();
    }
  });

  it('a NaN row together with an offset or weights is an error, not a silent drop', () => {
    const { x, nRows, nCols } = cols2(R.NA205_X1, R.NA205_X2);
    const pois = new wasm.PoissonRegressor();
    pois.setOffset(vector(new Array(nRows).fill(0)));
    expect(() => pois.fit(x, nRows, nCols, vector(R.NA205_YC))).toThrow();
    pois.free();
    const bin = new wasm.BinomialRegressor();
    bin.setWeights(vector(new Array(nRows).fill(1)));
    expect(() => bin.fit(x, nRows, nCols, vector(R.NA205_YC.map((v) => (v > 2 ? 1 : 0))))).toThrow();
    bin.free();
  });

  it('offset and weights of the wrong length are rejected', () => {
    const { x, nRows, nCols } = col(R.X_POISSON_OFFSET_R);
    const m = new wasm.PoissonRegressor();
    m.setOffset(vector([0, 0]));
    expect(() => m.fit(x, nRows, nCols, vector(R.Y_POISSON_OFFSET_R))).toThrow();
    m.free();
    const b = new wasm.BinomialRegressor();
    b.setWeights(vector([1]));
    expect(() => b.fit(x, nRows, nCols, vector(R.Y_POISSON_OFFSET_R.map((v) => (v > 60 ? 1 : 0))))).toThrow();
    b.free();
  });
});
