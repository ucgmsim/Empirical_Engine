//! P21 throughput over a synthetic site set, timing the same path as
//! `gm_models._core.p21` minus the numpy conversion.
//!
//! `cargo bench` writes HTML reports to `target/criterion/report/index.html`;
//! `cargo bench -- --profile-time 5` additionally writes flamegraphs.

use _core::im::IM;
use _core::p21::{coeffs::Coeffs, compute, P21Row};
use _core::result::GMMResult;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

const N_SITES: usize = 10_000;

fn bench_p21(c: &mut Criterion) {
    let row = |i: usize| {
        let x = i as f64 / N_SITES as f64;
        (6.0 + 3.0 * x, 1.0 + 500.0 * x, 150.0 + 1000.0 * x)
    };
    let interface: Vec<P21Row> = (0..N_SITES)
        .map(|i| {
            let (mag, rrup, vs30) = row(i);
            P21Row::Interface { mag, rrup, vs30 }
        })
        .collect();
    let slab: Vec<P21Row> = (0..N_SITES)
        .map(|i| {
            let (mag, rrup, vs30) = row(i);
            P21Row::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth: 10.0 + 100.0 * i as f64 / N_SITES as f64,
                backarc: i % 2 == 0,
            }
        })
        .collect();

    for (name, inputs) in [("interface", &interface), ("slab", &slab)] {
        for (im_name, im) in [("PGA", IM::PGA {}), ("SA(1.3)", IM::SA { period: 1.3 })] {
            c.bench_function(&format!("p21/{name}/{im_name}/{N_SITES}_sites"), |b| {
                b.iter(|| {
                    let coeffs = Coeffs::for_im(im).unwrap();
                    black_box(
                        inputs
                            .iter()
                            .map(|row| compute(im, &coeffs, row, 0.0, true))
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
