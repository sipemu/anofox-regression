// #307: multinomial (softmax) logistic regression and L1 / elastic-net
// penalties, against scikit-learn 1.9.1 LogisticRegression.
import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, expectClose } from './setup';

const X = [0.497, -0.138, 0.648, 1.523, -0.234, -0.234, 1.579, 0.767, -0.469, 0.543, -0.463, -0.466, 0.242, -1.913, -1.725, -0.562, -1.013, 0.314, -0.908, -1.412, 1.466, -0.226, 0.068, -1.425, -0.544, 0.111, -1.151, 0.376, -0.601, -0.292, -0.602, 1.852, -0.013, -1.058, 0.823, -1.221, 0.209, -1.96, -1.328, 0.197, 0.738, 0.171, -0.116, -0.301, -1.479, -0.72, -0.461, 1.057, 0.344, -1.763, 0.324, -0.385, -0.677, 0.612, 1.031, 0.931, -0.839, -0.309, 0.331, 0.976, -0.479, -0.186, -1.106, -1.196, 0.813, 1.356, -0.072, 1.004, 0.362, -0.645, 0.361, 1.538, -0.036, 1.565, -2.62, 0.822, 0.087, -0.299, 0.092, -1.988, -0.22, 0.357, 1.478, -0.518, -0.808, -0.502, 0.915, 0.329, -0.53, 0.513, 0.097, 0.969, -0.702, -0.328, -0.392, -1.464, 0.296, 0.261, 0.005, -0.235, -1.415, -0.421, -0.343, -0.802, -0.161, 0.404, 1.886, 0.175, 0.258, -0.074, -1.919, -0.027, 0.06, 2.463, -0.192, 0.302, -0.035, -1.169, 1.143, 0.752, 0.791, -0.909, 1.403, -1.402, 0.587, 2.19, -0.991, -0.566, 0.1, -0.503, -1.551, 0.069, -1.062, 0.474, -0.919, 1.55, -0.783, -0.322, 0.814, -1.231, 0.227, 1.307, -1.607, 0.185, 0.26, 0.782, -1.237, -1.32, 0.522, 0.297, 0.25, 0.346, -0.68, 0.232, 0.293, -0.714, 1.866, 0.474, -1.191, 0.657, -0.975, 0.787, 1.159, -0.821, 0.963, 0.413, 0.822, 1.897, -0.245, -0.754, -0.89, -0.816, -0.077, 0.341, 0.277, 0.827, 0.013, 1.454, -0.265, 2.72];
const YB = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
const Y3 = [2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0];

describe('MultinomialLogisticRegressor', () => {
  let wasm: any;
  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  it('multinomial L2 matches sklearn lbfgs', () => {
    const f = new wasm.MultinomialLogisticRegressor().fit(new Float64Array(X), 60, 3, new Float64Array(Y3));
    expect(f.getConverged()).toBe(true);
    expect(Array.from(f.classes())).toEqual([0, 1, 2]);
    const b = f.coefficients();
    expectClose(b[0][0], 0.5453637893962024, 1e-7);
    expectClose(b[2][2], 0.5985701427348307, 1e-7);
    const b0 = f.intercepts();
    expectClose(b0[1], -0.5389281686445537, 1e-7);
    const p = f.predictProba(new Float64Array(X.slice(0, 6)), 2);
    expectClose(p[0][2], 0.45269591202565906, 1e-7);
    expect(Array.from(f.predict(new Float64Array(X.slice(0, 6)), 2))).toEqual([2, 0]);
    expect(f.score(new Float64Array(X), 60, new Float64Array(Y3))).toBeGreaterThan(0.5);
    expect(f.objective()).toBeGreaterThan(-f.logLikelihood());
    expect(f.decisionFunction(new Float64Array(X.slice(0, 3)), 1)[0].length).toBe(3);
  });

  it('binary L1 gives exact zeros and matches sklearn saga', () => {
    const reg = new wasm.MultinomialLogisticRegressor();
    reg.setC(0.5);
    reg.setL1Ratio(1.0);
    const f = reg.fit(new Float64Array(X), 60, 3, new Float64Array(YB));
    const b = f.coefficients();
    expect(b.length).toBe(1);
    expectClose(b[0][0], 0.8906884616279431, 1e-7);
    expectClose(f.intercepts()[0], 0.31326346483484513, 1e-7);
    reg.setC(0.05);
    const z = reg.fit(new Float64Array(X), 60, 3, new Float64Array(YB));
    expect(z.coefficients()[0]).toEqual([0, 0, 0]);
  });
});
