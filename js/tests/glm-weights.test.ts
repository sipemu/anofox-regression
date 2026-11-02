// #308: prior weights for the Poisson, Gamma and Tweedie GLMs, against R's
// glm(weights = ) (tests/fixtures/glm_weights_validation.rs).
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, expectClose } from './setup';
import {
  GW308_X1, GW308_X2, GW308_W, GW308_Y_POIS, GW308_Y_GAMMA, GW308_Y_TW,
  GW308_POIS_COEF, GW308_POIS_SE, GW308_POIS_AIC, GW308_GAMMA_COEF,
  GW308_GAMMA_DEV, GW308_TW_COEF,
} from './references';

const N = GW308_X1.length;
const X = new Float64Array(N * 2);
for (let i = 0; i < N; i++) {
  X[2 * i] = GW308_X1[i];
  X[2 * i + 1] = GW308_X2[i];
}
const W = new Float64Array(GW308_W);

function coefs(f: any): number[] {
  return [f.getIntercept(), ...Array.from(f.getCoefficients() as Float64Array)];
}

describe('GLM prior weights (#308)', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('weighted Poisson matches glm(weights = )', () => {
    const reg = new wasm.PoissonRegressor();
    reg.setWeights(W);
    reg.setTolerance(1e-12);
    const f = reg.fit(X, N, 2, new Float64Array(GW308_Y_POIS));
    coefs(f).forEach((b, j) => expectClose(b, GW308_POIS_COEF[j], 1e-6));
    const r = f.getResult();
    expectClose(r.stdErrors[0], GW308_POIS_SE[1], 1e-6);
    expectClose(r.aic, GW308_POIS_AIC, 1e-8);
    expect(Array.from(f.priorWeights())).toEqual(GW308_W);
  });

  it('weighted Gamma and Tweedie match glm(weights = )', () => {
    const g = new wasm.GammaRegressor();
    g.setWeights(W);
    g.setTolerance(1e-12);
    const fg = g.fit(X, N, 2, new Float64Array(GW308_Y_GAMMA));
    coefs(fg).forEach((b, j) => expectClose(b, GW308_GAMMA_COEF[j], 1e-6));
    expectClose(fg.getResult().deviance, GW308_GAMMA_DEV, 1e-8);

    const t = new wasm.TweedieRegressor();
    t.setVarPower(1.5);
    t.setLinkPower(0);
    t.setWeights(W);
    t.setTolerance(1e-12);
    t.setMaxIterations(200);
    const ft = t.fit(X, N, 2, new Float64Array(GW308_Y_TW));
    coefs(ft).forEach((b, j) => expectClose(b, GW308_TW_COEF[j], 1e-6));
    expect(ft.priorWeights().length).toBe(N);
  });
});
