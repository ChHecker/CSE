use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cse::{
    interpolation::{AitkenNeville, BarycentricLagrange, CubicSpline, Interpolation, Newton},
    linalg::{Lu, Qr},
};
use nalgebra::{DVector, SMatrix, SVector};

fn linalg_benchmark(c: &mut Criterion) {
    let a: SMatrix<f64, 100, 100> = SMatrix::new_random();
    c.bench_function("lu construction", |b| b.iter(|| Lu::new(black_box(a))));
    c.bench_function("qr construction", |b| b.iter(|| Qr::new(black_box(a))));

    let y: SVector<f64, 100> = SVector::new_random();
    let lu = Lu::new(a).unwrap();
    let qr = Qr::new(a);
    c.bench_function("lu solution", |b| b.iter(|| lu.solve(&y)));
    c.bench_function("qr solution", |b| b.iter(|| qr.solve(&y)));
}

fn interpolation_benchmark(c: &mut Criterion) {
    let x: Vec<f64> = DVector::new_random(100).data.into();
    let y: Vec<f64> = DVector::new_random(100).data.into();

    c.bench_function("barycentric lagrange construction", |b| {
        b.iter(|| BarycentricLagrange::new(black_box(x.clone()), black_box(y.clone())))
    });
    c.bench_function("newton construction", |b| {
        b.iter(|| Newton::new(black_box(x.clone()), black_box(y.clone())))
    });
    c.bench_function("cubic spline construction", |b| {
        b.iter(|| {
            CubicSpline::new(
                black_box(x.clone()),
                black_box(y.clone()),
                black_box(0.),
                black_box(0.),
            )
        })
    });

    let x: Vec<f64> = DVector::new_random(100).data.into();
    let bl = BarycentricLagrange::new(x.clone(), y.clone());
    let an = AitkenNeville::new(x.clone(), y.clone());
    let ne = Newton::new(x.clone(), y.clone());
    let cs = CubicSpline::new(x.clone(), y.clone(), 0., 0.);

    c.bench_function("barycentric lagrange interpolation", |b| {
        b.iter(|| {
            for xi in x.clone() {
                bl.interpolate(xi);
            }
        })
    });
    c.bench_function("aitken-neville interpolation", |b| {
        b.iter(|| {
            for xi in x.clone() {
                an.interpolate(xi);
            }
        })
    });
    c.bench_function("newton interpolation", |b| {
        b.iter(|| {
            for xi in x.clone() {
                ne.interpolate(xi);
            }
        })
    });
    c.bench_function("cubic splining", |b| {
        b.iter(|| {
            for xi in x.clone() {
                cs.interpolate(xi);
            }
        })
    });
}

criterion_group!(benches, linalg_benchmark, interpolation_benchmark);
criterion_main!(benches);
