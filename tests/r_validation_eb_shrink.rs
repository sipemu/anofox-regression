//! Issue #58: `solvers::eb_shrink` against `metafor::rma(method = "DL")` and
//! `blup()`. Reference values from `tests/r_scripts/generate_eb_shrink_validation.R`
//! (metafor 4.x).

use anofox_regression::solvers::eb_shrink::{eb_shrink, EbShrinkOptions, TauMethod};

const TOL: f64 = 1e-10;

fn close(name: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOL * want.abs().max(1.0),
        "{name}: got {got:e}, want {want:e}"
    );
}

fn close_all(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(&format!("{name}[{i}]"), g, w);
    }
}

const YI: [f64; 5] = [0.10, 0.30, 0.35, 0.65, 1.00];
const SEI: [f64; 5] = [0.30, 0.10, 0.50, 0.20, 0.40];

#[test]
fn dl_matches_metafor_rma_and_blup() {
    let r = eb_shrink(&YI, &SEI, &EbShrinkOptions::default()).unwrap();
    close("mu", r.mu, 4.223980166707431e-01);
    close("mu_se", r.mu_se, 1.260284589424375e-01);
    close("tau2", r.tau_squared, 2.394587965990844e-02);
    close("q", r.q, 5.737201081799203e+00);
    close("i2", r.i_squared, 3.027959203504876e-01);
    assert_eq!(r.n_groups, 5);
    let g = |f: fn(&anofox_regression::solvers::eb_shrink::ShrunkenGroup) -> f64| {
        r.groups.iter().map(f).collect::<Vec<_>>()
    };
    close_all(
        "shrunken (blup pred)",
        &g(|x| x.shrunken),
        &[
            3.546456404300858e-01,
            3.360568109876677e-01,
            4.160696345940865e-01,
            5.076283666502101e-01,
            4.975896306916652e-01,
        ],
    );
    close_all(
        "weight",
        &g(|x| x.weight),
        &[
            2.101513431760687e-01,
            7.054134375015052e-01,
            8.741098675999866e-02,
            3.744710337438922e-01,
            1.301789401544693e-01,
        ],
    );
    close_all(
        "shrunken_se",
        &g(|x| x.shrunken_se),
        &[
            1.375268006093583e-01,
            8.398889435523636e-02,
            1.478267455164987e-01,
            1.223880768284055e-01,
            1.443212750245614e-01,
        ],
    );
}

#[test]
fn none_method_matches_metafor_fixed_effect() {
    let opts = EbShrinkOptions {
        method: TauMethod::None,
        ..Default::default()
    };
    let r = eb_shrink(&YI, &SEI, &opts).unwrap();
    close("mu", r.mu, 3.758587967356235e-01);
    close("mu_se", r.mu_se, 8.265842980736918e-02);
    assert_eq!(r.tau_squared, 0.0);
    close("q", r.q, 5.737201081799203e+00);
    close("i2", r.i_squared, 3.027959203504876e-01);
    for g in &r.groups {
        assert_eq!(g.shrunken, r.mu);
    }
}

#[test]
fn fixed_tau_squared_matches_metafor() {
    let opts = EbShrinkOptions {
        tau_squared: Some(0.05),
        ..Default::default()
    };
    let r = eb_shrink(&YI, &SEI, &opts).unwrap();
    close("mu", r.mu, 4.385608856088561e-01);
    close("mu_se", r.mu_se, 1.524704314689400e-01);
    close("tau2", r.tau_squared, 0.05);
    close("q", r.q, 5.737201081799203e+00);
    // I^2 stays Higgins–Thompson (Q - df) / Q; metafor reports 0.4756 here because
    // it derives I^2 from the supplied tau^2 (documented difference).
    close("i2", r.i_squared, 3.027959203504876e-01);
    close_all(
        "shrunken",
        &r.groups.iter().map(|g| g.shrunken).collect::<Vec<_>>(),
        &[
            3.176462836056931e-01,
            3.230934809348094e-01,
            4.238007380073801e-01,
            5.560270602706027e-01,
            5.722368652257951e-01,
        ],
    );
}

#[test]
fn bcg_trials_match_metafor() {
    // metafor::dat.bcg, log risk ratios and their standard errors.
    let yi = [
        -8.893113339202054e-01,
        -1.585388657201431e+00,
        -1.348073148299693e+00,
        -1.441551190021305e+00,
        -2.175473222112956e-01,
        -7.861155858188640e-01,
        -1.620898223598392e+00,
        1.195233352384051e-02,
        -4.694176487381494e-01,
        -1.371344803472784e+00,
        -3.393588283383906e-01,
        4.459134005713787e-01,
        -1.731394821687982e-02,
    ];
    let sei = [
        5.706003548929507e-01,
        4.411135017182582e-01,
        6.444904695710910e-01,
        1.414568199213017e-01,
        2.262966463950159e-01,
        8.310005087789513e-02,
        4.722470196542432e-01,
        6.294107798423641e-02,
        2.375588568402554e-01,
        2.702310004663213e-01,
        1.114101161096232e-01,
        7.297299810204818e-01,
        2.672165033900157e-01,
    ];
    let r = eb_shrink(&yi, &sei, &EbShrinkOptions::default()).unwrap();
    close("mu", r.mu, -7.141172220732273e-01);
    close("mu_se", r.mu_se, 1.787420895354031e-01);
    close("tau2", r.tau_squared, 3.087602628623761e-01);
    close("q", r.q, 1.522330080823733e+02);
    close("i2", r.i_squared, 9.211734685456204e-01);
    close_all(
        "shrunken",
        &r.groups.iter().map(|g| g.shrunken).collect::<Vec<_>>(),
        &[
            -7.993909731858653e-01,
            -1.248573568693491e+00,
            -9.844290132641388e-01,
            -1.397277187359840e+00,
            -2.881904295096069e-01,
            -7.845405241511262e-01,
            -1.240611733209390e+00,
            2.754440276730498e-03,
            -5.072315485381770e-01,
            -1.245635565579696e+00,
            -3.538419540922825e-01,
            -2.883644709003100e-01,
            -1.481913587447598e-01,
        ],
    );
    close_all(
        "shrunken_se",
        &r.groups.iter().map(|g| g.shrunken_se).collect::<Vec<_>>(),
        &[
            3.980888084723142e-01,
            3.454854265071101e-01,
            4.208423739497777e-01,
            1.370844853719578e-01,
            2.095826862723999e-01,
            8.218606149691397e-02,
            3.598445616190225e-01,
            6.254113662672706e-02,
            2.184337941675840e-01,
            2.430168612481431e-01,
            1.092360952252070e-01,
            4.420854221311794e-01,
            2.408174517902191e-01,
        ],
    );
}
