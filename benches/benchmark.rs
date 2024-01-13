use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cse::{
    barnes_hut::{barnes_hut, gravitational_acceleration, Particle},
    interpolation::{AitkenNeville, BarycentricLagrange, CubicSpline, Interpolation, Newton},
    linalg::{Lu, Qr},
};
use nalgebra::{DVector, SMatrix, SVector, Vector3};
use rand::{rngs::StdRng, Rng, SeedableRng};

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

fn barnes_hut_particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [10, 20, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    (0..n_par)
                        .map(|_| {
                            Particle::new(
                                rng.gen_range(0.0..1000.0),
                                10. * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Vec<_>>()
                },
                |p| {
                    barnes_hut(
                        black_box(p),
                        |r, m1, m2| gravitational_acceleration(r, m1, m2, 1e-4),
                        0.1,
                        100,
                        1.5,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn barnes_hut_theta(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let particles = (0..50)
        .map(|_| {
            Particle::new(
                rng.gen_range(0.0..1000.0),
                10. * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::from_parameter(theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || particles.clone(),
                |p| {
                    barnes_hut(
                        black_box(p),
                        |r, m1, m2| gravitational_acceleration(r, m1, m2, 1e-4),
                        0.1,
                        100,
                        theta,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(
    benches,
    linalg_benchmark,
    interpolation_benchmark,
    barnes_hut_particles,
    barnes_hut_theta
);
criterion_main!(benches);
