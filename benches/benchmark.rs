use std::thread;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
// use cse::interpolation::{AitkenNeville, BarycentricLagrange, CubicSpline, Interpolation, Newton};
// use cse::linalg::solve::{Lu, Qr};
// use cse::quadrature::{adapative_trapezoidal, composite_trapezoidal, romberg, simpson};
use cse::sorting::{merge_sort, quick_sort, selection_sort};
use rand::{thread_rng, Rng};

// fn linalg_benchmark(c: &mut Criterion) {
//     let a: SMatrix<f64, 100, 100> = SMatrix::new_random();
//     c.bench_function("lu construction", |b| b.iter(|| Lu::new(black_box(a))));
//     c.bench_function("qr construction", |b| b.iter(|| Qr::new(black_box(a))));

//     let y: SVector<f64, 100> = SVector::new_random();
//     let lu = Lu::new(a).unwrap();
//     let qr = Qr::new(a);
//     c.bench_function("lu solution", |b| b.iter(|| lu.solve(&y)));
//     c.bench_function("qr solution", |b| b.iter(|| qr.solve(&y)));
// }

// fn interpolation_benchmark(c: &mut Criterion) {
//     let x: Vec<f64> = DVector::new_random(100).data.into();
//     let y: Vec<f64> = DVector::new_random(100).data.into();

//     c.bench_function("barycentric lagrange construction", |b| {
//         b.iter(|| BarycentricLagrange::new(black_box(x.clone()), black_box(y.clone())))
//     });
//     c.bench_function("newton construction", |b| {
//         b.iter(|| Newton::new(black_box(x.clone()), black_box(y.clone())))
//     });
//     c.bench_function("cubic spline construction", |b| {
//         b.iter(|| {
//             CubicSpline::new(
//                 black_box(x.clone()),
//                 black_box(y.clone()),
//                 black_box(0.),
//                 black_box(0.),
//             )
//         })
//     });

//     let x: Vec<f64> = DVector::new_random(100).data.into();
//     let bl = BarycentricLagrange::new(x.clone(), y.clone());
//     let an = AitkenNeville::new(x.clone(), y.clone());
//     let ne = Newton::new(x.clone(), y.clone());
//     let cs = CubicSpline::new(x.clone(), y.clone(), 0., 0.);

//     c.bench_function("barycentric lagrange interpolation", |b| {
//         b.iter(|| {
//             for xi in x.clone() {
//                 bl.interpolate(xi);
//             }
//         })
//     });
//     c.bench_function("aitken-neville interpolation", |b| {
//         b.iter(|| {
//             for xi in x.clone() {
//                 an.interpolate(xi);
//             }
//         })
//     });
//     c.bench_function("newton interpolation", |b| {
//         b.iter(|| {
//             for xi in x.clone() {
//                 ne.interpolate(xi);
//             }
//         })
//     });
//     c.bench_function("cubic splining", |b| {
//         b.iter(|| {
//             for xi in x.clone() {
//                 cs.interpolate(xi);
//             }
//         })
//     });
// }

// fn f(x: f64, sleep_micros: u64) -> f64 {
//     thread::sleep(Duration::from_micros(sleep_micros));
//     x.exp()
// }

// fn quadrature_benchmark(c: &mut Criterion) {
//     let err = 1e-8;
//     let f_test = |x| f(x, 0);
//     let exact = 1f64.exp() - 1.;

//     let mut n_ct = 10;
//     let mut int_t = composite_trapezoidal(f_test, 0., 1., n_ct);
//     while (int_t - exact).abs() > err {
//         n_ct *= 2;
//         int_t = composite_trapezoidal(f_test, 0., 1., n_ct);
//     }

//     let mut n_s = 10;
//     let mut int_s = simpson(f_test, 0., 1., n_s);
//     while (int_s - exact).abs() > err {
//         n_s *= 2;
//         int_s = simpson(f_test, 0., 1., n_s);
//     }

//     let mut err_at = err;
//     let mut int_at = adapative_trapezoidal(f_test, 0., 1., err_at);
//     while (int_at - exact).abs() > err {
//         err_at /= 2.;
//         int_at = adapative_trapezoidal(f_test, 0., 1., err_at);
//     }

//     let mut err_rom = err;
//     let mut int_rom = romberg(f_test, 0., 1., err_rom);
//     while (int_rom - exact).abs() > err {
//         err_rom /= 2.;
//         int_rom = romberg(f_test, 0., 1., err_rom);
//     }

//     let mut group = c.benchmark_group("quadrature");
//     for sleep in [0, 5, 10] {
//         let f = |x| f(x, sleep);

//         group.bench_function(BenchmarkId::new("composite trapezoidal", sleep), |b| {
//             b.iter(|| composite_trapezoidal(f, 0., 1., n_ct))
//         });
//         group.bench_function(BenchmarkId::new("simpson", sleep), |b| {
//             b.iter(|| simpson(f, 0., 1., n_s))
//         });
//         group.bench_function(BenchmarkId::new("adaptive trapezoidal", sleep), |b| {
//             b.iter(|| adapative_trapezoidal(f, 0., 1., err_at))
//         });
//         group.bench_function(BenchmarkId::new("romberg", sleep), |b| {
//             b.iter(|| romberg(f, 0., 1., err_at))
//         });
//     }
// }

fn sort_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("sort");
    for n in [100, 1_000, 10_000] {
        let arr: Vec<i32> = (0..n).map(|_| rng.gen_range(0..100)).collect();

        group.bench_function(BenchmarkId::new("selection", n), |b| {
            b.iter(|| selection_sort(&mut arr.clone()))
        });
        group.bench_function(BenchmarkId::new("merge", n), |b| {
            b.iter(|| merge_sort(&arr.clone()))
        });
        group.bench_function(BenchmarkId::new("quick", n), |b| {
            b.iter(|| quick_sort(&mut arr.clone()))
        });
    }
}

criterion_group!(
    benches,
    // linalg_benchmark,
    // interpolation_benchmark,
    // quadrature_benchmark,
    sort_benchmark
);
criterion_main!(benches);
