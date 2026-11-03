//! P21 throughput over a synthetic site set.
//!
//! `cargo bench` writes HTML reports to `target/criterion/report/index.html`;
//! `cargo bench -- --profile-time 5` additionally writes flamegraphs.

use _core::coeffs::Coeffs;
use _core::im::IM;
use _core::p21::{compute, Inputs, TectType};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

const N_SITES: usize = 10_000;

fn bench_p21(c: &mut Criterion) {
    let inputs: Vec<Inputs> = (0..N_SITES)
        .map(|i| {
            let f = i as f64 / N_SITES as f64;
            Inputs {
                mag: 6.0 + 3.0 * f,
                rrup: 1.0 + 500.0 * f,
                vs30: 150.0 + 1000.0 * f,
                hypo_depth: 10.0 + 100.0 * f,
                backarc: i % 2 == 0,
            }
        })
        .collect();

    for (name, tect_type) in [("interface", TectType::Interface), ("slab", TectType::Slab)] {
        for (im_name, im) in [("PGA", IM::PGA {}), ("SA(1.3)", IM::SA { period: 1.3 })] {
            c.bench_function(&format!("p21/{name}/{im_name}/{N_SITES}_sites"), |b| {
                b.iter(|| {
                    let coeffs = Coeffs::for_im(im).unwrap();
                    for inp in &inputs {
                        black_box(
                            compute(tect_type, im, &coeffs, black_box(inp), 0.0, true).unwrap(),
                        );
                    }
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
