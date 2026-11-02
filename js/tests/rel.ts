// Relative-tolerance comparisons as in the Rust R validation tests.

/** |a − b| ≤ tol · (1 + |b|). */
export function near(a: number, b: number, tol: number, what = ''): void {
  if (!(Math.abs(a - b) <= tol * (1 + Math.abs(b)))) {
    throw new Error(`${what}: ${a} vs ${b} (tol ${tol})`);
  }
}

/** Elementwise `near`, with a length check. */
export function allNear(a: ArrayLike<number>, b: ArrayLike<number>, tol: number, what = ''): void {
  if (a.length !== b.length) throw new Error(`${what}: length ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) near(a[i], b[i], tol, `${what}[${i}]`);
}
