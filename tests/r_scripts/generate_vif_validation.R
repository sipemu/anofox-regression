# Reference values for tests/issue196_gvif.rs (#196): car::vif VIF and GVIF.
suppressMessages(library(car))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
d <- mtcars
v("WT", d$wt); v("HP", d$hp); v("DISP", d$disp); v("DRAT", d$drat); v("CYL", d$cyl); v("GEAR", d$gear)
# Plain VIF
v("VIF_PLAIN", vif(lm(mpg ~ wt + hp + disp + drat, data = d)))
# GVIF: wt, factor(cyl) (2 dummies), hp, factor(gear) (2 dummies)
g <- vif(lm(mpg ~ wt + factor(cyl) + hp + factor(gear), data = d))
v("GVIF", g[, "GVIF"]); v("GVIF_ADJ", g[, "GVIF^(1/(2*Df))"])
