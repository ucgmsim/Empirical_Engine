//! P21 throughput over a synthetic site set, timing the same path as
//! `gm_models._core.p21` minus the numpy conversion.
//!
//! `cargo bench` writes HTML reports to `target/criterion/report/index.html`;
//! `cargo bench -- --profile-time 5` additionally writes flamegraphs.

use _core::coeffs::Coeffs;
use _core::im::IM;
use _core::p21::{compute, P21Inputs};
use _core::result::GMMResult;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

const N_SITES: usize = 10_000;

fn bench_p21(c: &mut Criterion) {
    let column = |lo: f64, span: f64| -> Vec<f64> {
        (0..N_SITES)
            .map(|i| lo + span * i as f64 / N_SITES as f64)
            .collect()
    };
    let interface = P21Inputs::Interface {
        mag: column(6.0, 3.0),
        rrup: column(1.0, 500.0),
        vs30: column(150.0, 1000.0),
    };
    let slab = P21Inputs::Slab {
        mag: column(6.0, 3.0),
        rrup: column(1.0, 500.0),
        vs30: column(150.0, 1000.0),
        hypo_depth: column(10.0, 100.0),
        backarc: (0..N_SITES).map(|i| i % 2 == 0).collect(),
    };

    for (name, inputs) in [("interface", &interface), ("slab", &slab)] {
        for (im_name, im) in [("PGA", IM::PGA {}), ("SA(1.3)", IM::SA { period: 1.3 })] {
            c.bench_function(&format!("p21/{name}/{im_name}/{N_SITES}_sites"), |b| {
                b.iter(|| {
                    let coeffs = Coeffs::for_im(im).unwrap();
                    black_box(
                        inputs
                            .iter()
                            .unwrap()
                            .map(|row| compute(im, &coeffs, &row, 0.0, true))
                            .collect::<Result<GMMResult, _>>()
                            .unwrap(),
                    )
                })
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_p21
}
criterion_main!(benches);
