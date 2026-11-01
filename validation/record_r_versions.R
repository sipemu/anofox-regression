#!/usr/bin/env Rscript
# Record the R and package versions the reference fixtures were checked
# with (validation/R_VERSIONS.md). Run from the repository root:
#
#   Rscript validation/record_r_versions.R > validation/R_VERSIONS.md
pkgs <- c("bda", "broom", "car", "greybox", "grpreg", "Hmisc", "lavaan", "lme4",
          "lmtest", "MASS", "mediation", "mice", "multcomp", "nlme", "numDeriv",
          "quantreg", "rms", "sandwich", "statmod", "survival")
cat("# R environment of the reference fixtures\n\n")
cat("Recorded with `Rscript validation/record_r_versions.R` on", format(Sys.Date()),
    "after `validation/regenerate_r_fixtures.sh` reproduced the fixtures in\n",
    "`validation/r_fixtures.txt` (values compared by `validation/compare_fixtures.py`).\n\n")
cat("| Package | Version |\n|---|---|\n")
cat(sprintf("| R | %s.%s |\n", R.version$major, R.version$minor))
for (p in pkgs) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) "not installed")
  cat(sprintf("| %s | %s |\n", p, v))
}
