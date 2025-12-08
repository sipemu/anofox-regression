//! Influence measures: Cook's distance, DFFITS, DFBETAS.

use faer::Col;

/// Compute Cook's distance for each observation.
///
/// Cook's distance measures the influence of each observation on the fitted values.
/// D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)
///
/// Observations with D_i > 4/n or D_i > 1 are typically considered influential.
pub fn cooks_distance(
    residuals: &Col<f64>,
    leverage: &Col<f64>,
    mse: f64,
    n_params: usize,
) -> Col<f64> {
    let n = residuals.nrows();

    if mse <= 0.0 || !mse.is_finite() || n_params == 0 {
        return Col::from_fn(n, |_| f64::NAN);
    }

    Col::from_fn(n, |i| {
        let e_i = residuals[i];
        let h_ii = leverage[i];
        let one_minus_h = (1.0 - h_ii).max(1e-14);

        // D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)
        let d_i = (e_i * e_i / (n_params as f64 * mse)) * (h_ii / (one_minus_h * one_minus_h));

        if d_i.is_finite() {
            d_i.max(0.0)
        } else {
            f64::NAN
        }
    })
}

/// Compute DFFITS for each observation.
///
/// DFFITS measures the difference in fitted value when observation i is deleted.
/// DFFITS_i = r*_i * sqrt(h_ii / (1 - h_ii))
///
/// where r*_i is the externally studentized residual.
///
/// Observations with |DFFITS_i| > 2*sqrt(p/n) are typically considered influential.
pub fn dffits(residuals: &Col<f64>, leverage: &Col<f64>, mse: f64, n_params: usize) -> Col<f64> {
    let n = residuals.nrows();
    let df_resid = n - n_params;

    if df_resid <= 1 || mse <= 0.0 || !mse.is_finite() {
        return Col::from_fn(n, |_| f64::NAN);
    }

    let rss = mse * df_resid as f64;

    Col::from_fn(n, |i| {
        let h_ii = leverage[i];
        let e_i = residuals[i];
        let one_minus_h = (1.0 - h_ii).max(1e-14);

        // Leave-one-out MSE
        let rss_loo = rss - e_i * e_i / one_minus_h;
        let df_loo = (df_resid - 1) as f64;

        if df_loo <= 0.0 || rss_loo <= 0.0 {
            return f64::NAN;
        }

        let mse_loo = rss_loo / df_loo;
        let s_loo = mse_loo.sqrt();

        // Externally studentized residual
        let r_star = e_i / (s_loo * one_minus_h.sqrt());

        // DFFITS
        r_star * (h_ii / one_minus_h).sqrt()
    })
}

/// Identify influential observations based on Cook's distance.
///
/// Returns indices of observations with D_i > threshold.
/// Common thresholds: 4/n or 1.
pub fn influential_cooks(cooks_d: &Col<f64>, threshold: Option<f64>) -> Vec<usize> {
    let n = cooks_d.nrows();
    let cutoff = threshold.unwrap_or(4.0 / n as f64);

    cooks_d
        .iter()
        .enumerate()
        .filter(|(_, &d)| d.is_finite() && d > cutoff)
        .map(|(i, _)| i)
        .collect()
}

/// Identify influential observations based on DFFITS.
///
/// Returns indices of observations with |DFFITS_i| > threshold.
/// Common threshold: 2*sqrt(p/n).
pub fn influential_dffits(
    dffits: &Col<f64>,
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    let n = dffits.nrows();
    let cutoff = threshold.unwrap_or(2.0 * (n_params as f64 / n as f64).sqrt());

    dffits
        .iter()
        .enumerate()
        .filter(|(_, &d)| d.is_finite() && d.abs() > cutoff)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooks_distance_non_negative() {
        let residuals = Col::from_fn(20, |i| (i as f64 - 9.5));
        let leverage = Col::from_fn(20, |i| 0.1 + 0.02 * i as f64);
        let mse = 10.0;
        let n_params = 3;

        let cooks = cooks_distance(&residuals, &leverage, mse, n_params);

        for i in 0..cooks.nrows() {
            assert!(
                cooks[i] >= 0.0 || cooks[i].is_nan(),
                "Cook's distance[{}] = {} should be >= 0",
                i,
                cooks[i]
            );
        }
    }

    #[test]
    fn test_dffits_outlier_detection() {
        // Create data with one influential point
        // Use realistic values that won't cause numerical issues
        let mut residuals = Col::from_fn(30, |_| 0.5);
        let mut leverage = Col::from_fn(30, |_| 0.1);

        // Make point 15 have moderately high residual and leverage
        // Values chosen to not cause negative leave-one-out RSS
        residuals[15] = 2.0;
        leverage[15] = 0.4;

        let mse = 1.0;
        let n_params = 2;

        let dff = dffits(&residuals, &leverage, mse, n_params);

        // Point 15 should have higher |DFFITS| than average
        let dff_15 = dff[15].abs();
        let other_mean: f64 = (0..30)
            .filter(|&i| i != 15)
            .filter(|&i| dff[i].is_finite())
            .map(|i| dff[i].abs())
            .sum::<f64>()
            / 29.0;

        assert!(
            dff_15 > other_mean,
            "Point 15 DFFITS={} should be larger than mean={}",
            dff_15,
            other_mean
        );
    }

    #[test]
    fn test_cooks_influential_detection() {
        let mut residuals = Col::from_fn(20, |_| 0.1);
        let mut leverage = Col::from_fn(20, |_| 0.1);

        // Make point 10 influential
        residuals[10] = 10.0;
        leverage[10] = 0.9;

        let mse = 1.0;
        let n_params = 2;

        let cooks = cooks_distance(&residuals, &leverage, mse, n_params);
        let influential = influential_cooks(&cooks, Some(0.5));

        assert!(
            influential.contains(&10),
            "Point 10 should be identified as influential"
        );
    }
}
