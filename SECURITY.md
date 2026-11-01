# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Do not open a public
issue for a security problem.

Include what is affected (crate or npm package, version), how to reproduce it
and the impact you expect. You will get an acknowledgement within a week; a
fix or mitigation for a confirmed issue is released as soon as practical and
credited in the advisory unless you prefer otherwise.

## Scope

`anofox-regression` is a numerical library with no network or file access and
`#![forbid(unsafe_code)]`. Relevant reports include panics, unbounded memory or
time on crafted input, and wrong results that a caller cannot detect (for
example `Ok` with silently invalid estimates). Dependencies are checked
against the RustSec advisory database in CI (`cargo deny`).

## Supported versions

Security fixes go into the latest release. Before 1.0 there are no backports
to older minor versions.
