# R environment of the reference fixtures

Recorded with `Rscript validation/record_r_versions.R` on 2026-11-01 after `validation/regenerate_r_fixtures.sh` reproduced the fixtures in
 `validation/r_fixtures.txt` (values compared by `validation/compare_fixtures.py`).

| Package | Version |
|---|---|
| R | 4.6.1 |
| bda | not installed |
| broom | 1.0.13 |
| car | 3.1.5 |
| greybox | 2.0.9 |
| grpreg | 3.6.0 |
| Hmisc | 5.2.6 |
| lavaan | not installed |
| lme4 | 2.0.1 |
| lmtest | 0.9.40 |
| MASS | 7.3.65 |
| mediation | not installed |
| mice | not installed |
| multcomp | 1.4.30 |
| nlme | 3.1.169 |
| numDeriv | 2016.8.1.1 |
| quantreg | 6.1 |
| rms | 8.1.1 |
| sandwich | 3.1.1 |
| statmod | 1.5.2 |
| survival | 3.8.6 |

On this date all fixtures in `validation/r_fixtures.txt` reproduced exactly
except `mediation_validation.rs`, whose package was not installed locally;
the scheduled `validation` workflow installs every package above and checks
all of them. Fixtures written by other means (scikit-learn under
`validation/python/`, and the R generators that write their own files) and
the reference values pasted into test files record their versions in their
headers.
