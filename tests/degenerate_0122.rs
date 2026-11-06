//! 0.12.2: data degeneracies are `RegressionError::Degenerate(..)` (typed),
//! not `InvalidInput` message text.

use anofox_regression::imputation::{areg_impute, AregImputeOptions, ImputeVariable};
use anofox_regression::solvers::path::{lambda_grid, PathOptions};
use anofox_regression::solvers::{
    ContinuationRatioRegressor, DegenerateData, GlmmRegressor, MultinomialLogisticRegressor,
    OrdinalRegressor, RegressionError,
};
use anofox_regression::transform::{areg, AregOptions, AregType};
use faer::{Col, Mat};

fn kind<T>(r: Result<T, RegressionError>) -> DegenerateData {
    match r {
        Err(RegressionError::Degenerate(d)) => d,
        Err(e) => panic!("expected a Degenerate error, got {e:?}"),
        Ok(_) => panic!("expected a Degenerate error, got Ok"),
    }
}

#[test]
fn degeneracies_are_typed() {
    let n = 30;
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { i as f64 } else { 1.0 });
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.1)
        .collect();
    let opts = AregOptions::default().x_types(Some(vec![AregType::Spline, AregType::Categorical]));
    assert_eq!(kind(areg(&x, &y, &opts)), DegenerateData::TooFewLevels);

    let xi = Mat::from_fn(n, 2, |i, j| if j == 0 { i as f64 } else { 2.0 });
    let kinds = [ImputeVariable::default(), ImputeVariable::default()];
    assert_eq!(
        kind(areg_impute(&xi, &kinds, &AregImputeOptions::default())),
        DegenerateData::ConstantColumn
    );

    let x1 = Mat::from_fn(n, 1, |i, _| i as f64);
    let one = Col::from_fn(n, |_| 1.0);
    assert_eq!(
        kind(OrdinalRegressor::builder().build().fit(&x1, &one)),
        DegenerateData::TooFewResponseLevels
    );
    assert_eq!(
        kind(
            MultinomialLogisticRegressor::builder()
                .build()
                .fit(&x1, &one)
        ),
        DegenerateData::TooFewResponseLevels
    );
    let two = Col::from_fn(n, |i| (i % 2) as f64);
    assert_eq!(
        kind(ContinuationRatioRegressor::builder().build().fit(&x1, &two)),
        DegenerateData::TooFewResponseLevels
    );
    let yg = Col::from_fn(n, |i| i as f64 * 0.3 + (i % 3) as f64);
    assert_eq!(
        kind(GlmmRegressor::gaussian().build().fit(&x1, &yg, &vec![0; n])),
        DegenerateData::TooFewGroups
    );
    assert_eq!(
        kind(lambda_grid(
            &x1,
            &one,
            &PathOptions::elastic_net(0.5),
            10,
            None
        )),
        DegenerateData::NoAssociation
    );
}
