use crate::{algebraic_solvers::newton, IntoMatrix, IntoVector, IterativeResult};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimMin, DimMinimum, OMatrix, RealField, SVector,
    ToTypenum,
};

fn call<F: RealField + Copy, const D: usize, T: IntoVector<F, D>>(
    f: &impl Fn(F, T) -> T,
    t: F,
    x: SVector<F, D>,
) -> SVector<F, D> {
    f(t, T::from_vector(x)).into_vector()
}

pub fn explicit_euler<const D: usize, T: IntoVector<f64, D>>(
    f: impl Fn(f64, T) -> T,
    y0: T,
    dt: f64,
    t_end: f64,
) -> Vec<T> {
    let n_steps = (t_end / dt) as usize;

    let t: Vec<f64> = (0..=n_steps + 1).map(|i| i as f64 * dt).collect();
    let mut y = vec![y0.into_vector(); n_steps + 1];

    for i in 0..n_steps {
        y[i + 1] = y[i] + call(&f, t[i], y[i]) * dt;
    }

    y.into_iter().map(|elem| T::from_vector(elem)).collect()
}

pub fn heun<const D: usize, T: IntoVector<f64, D>>(
    f: impl Fn(f64, T) -> T,
    y0: T,
    dt: f64,
    t_end: f64,
) -> Vec<T> {
    let n_steps = (t_end / dt) as usize;

    let t: Vec<f64> = vec![dt; n_steps + 1]
        .into_iter()
        .enumerate()
        .map(|(i, dt)| i as f64 * dt)
        .collect();
    let mut y = vec![y0.into_vector(); n_steps + 1];

    for i in 0..n_steps {
        let y_prim = call(&f, t[i + 1], y[i] + call(&f, t[i], y[i])) * dt;
        y[i + 1] = y[i] + (call(&f, t[i], y[i]) + y_prim) / 2. * dt;
    }

    y.into_iter().map(|elem| T::from_vector(elem)).collect()
}

pub fn runge_kutta<const D: usize, T: IntoVector<f64, D>>(
    f: impl Fn(f64, T) -> T,
    y0: T,
    dt: f64,
    t_end: f64,
) -> Vec<T> {
    let n_steps = (t_end / dt) as usize;

    let t: Vec<f64> = vec![dt; n_steps + 1]
        .into_iter()
        .enumerate()
        .map(|(i, dt)| i as f64 * dt)
        .collect();
    let mut y = vec![y0.into_vector(); n_steps + 1];

    for i in 0..n_steps {
        let y_1 = call(&f, t[i], y[i]);
        let y_half1 = call(&f, t[i] + dt / 2., y[i] + y_1 * dt / 2.);
        let y_half2 = call(&f, t[i] + dt / 2., y[i] + y_half1 * dt / 2.);
        let y_2 = call(&f, t[i] + dt, y[i] + y_half2 * dt);
        y[i + 1] = y[i] + (y_1 + y_half1 * 2. + y_half2 * 2. + y_2) * dt / 6.;
    }

    y.into_iter().map(|elem| T::from_vector(elem)).collect()
}

pub fn implicit_euler<const D: usize, V: IntoVector<f64, D>, M: IntoMatrix<f64, D>>(
    f: impl Fn(f64, V) -> V,
    df: impl Fn(f64, V) -> M,
    y0: V,
    dt: f64,
    t_end: f64,
) -> IterativeResult<Vec<V>>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>>
        + Allocator<(usize, usize), DimMinimum<Const<D>, Const<D>>>,
{
    let mut iterations = 0;
    let n_steps = (t_end / dt) as usize;

    let t: Vec<f64> = (0..=n_steps + 1).map(|i| i as f64 * dt).collect();
    let mut y = vec![y0.into_vector(); n_steps + 1];

    for i in 0..n_steps {
        y[i + 1] = match newton(
            |x| x - y[i] - f(t[i], V::from_vector(x)).into_vector() * dt,
            |x| {
                OMatrix::<f64, Const<D>, Const<D>>::identity()
                    - df(t[i], V::from_vector(x)).into_matrix() * dt
            },
            y[i],
            1e-8,
            100,
        ) {
            IterativeResult::Converged {
                result: y,
                iterations: n,
            } => {
                iterations = n;
                y
            }
            IterativeResult::MaxIterations(_) => {
                return IterativeResult::MaxIterations(
                    y[0..i]
                        .iter()
                        .map(|elem| V::from_vector(*elem))
                        .collect::<Vec<V>>(),
                )
            }
            IterativeResult::Failed => return IterativeResult::Failed,
        };

        if (y[i + 1] - y[i] - call(&f, t[i], y[i + 1]) * dt).norm() > 1e-8 {
            return IterativeResult::MaxIterations(
                y[0..i].iter().map(|elem| V::from_vector(*elem)).collect(),
            );
        }
    }

    IterativeResult::Converged {
        result: y.into_iter().map(|elem| V::from_vector(elem)).collect(),
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2};

    const MU: f64 = 1.;

    fn dahlquist(_t: f64, x: f64) -> f64 {
        -x
    }

    fn dahlquist_derivative(_t: f64, _x: f64) -> f64 {
        -1.
    }

    fn van_der_pol(_t: f64, x: Vector2<f64>) -> Vector2<f64> {
        let (x, y) = (x[0], x[1]);
        Vector2::new(y, MU * (1. - x * x) * y - x)
    }

    fn van_der_pol_derivative(_t: f64, x: Vector2<f64>) -> Matrix2<f64> {
        let (x, y) = (x[0], x[1]);
        Matrix2::new(0., 1., -2. * MU * x * y - 1., MU * (1. - x * x))
    }

    #[test]
    fn test_explicit_euler() {
        let ee = explicit_euler(dahlquist, 1., 0.5, 100.);
        assert!(ee[0] == 1.);
        assert!(ee.last().unwrap().abs() < 1e-20);
    }

    #[test]
    fn test_implicit_euler() {
        let ie = implicit_euler(dahlquist, dahlquist_derivative, 1., 0.5, 100.);
        assert!(ie.is_converged());
        let ie = ie.unwrap();
        assert!(ie[0] == 1.);
        assert!(ie.last().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_heun() {
        let heun = heun(dahlquist, 1., 0.5, 100.);
        assert!(heun[0] == 1.);
        assert!(heun.last().unwrap().abs() < 1e-20);
    }

    #[test]
    fn test_runge_kutta() {
        let rk = runge_kutta(dahlquist, 1., 0.5, 100.);
        assert!(rk[0] == 1.);
        assert!(rk.last().unwrap().abs() < 1e-20);
    }

    #[test]
    fn test_explicit_euler_2d() {
        let ee = explicit_euler(van_der_pol, Vector2::new(1., 1.), 0.25, 100.);
        assert!(ee[0] == Vector2::new(1., 1.));
        for arr in ee {
            for elem in arr.iter() {
                assert!(elem.abs() < 10.);
            }
        }
    }

    #[test]
    fn test_heun_2d() {
        let heun = heun(van_der_pol, Vector2::new(1., 1.), 0.25, 100.);
        assert!(heun[0] == Vector2::new(1., 1.));
        for arr in heun {
            for elem in arr.iter() {
                assert!(elem.abs() < 10.);
            }
        }
    }

    #[test]
    fn test_runge_kutta_2d() {
        let rk = runge_kutta(van_der_pol, Vector2::new(1., 1.), 0.25, 100.);
        assert!(rk[0] == Vector2::new(1., 1.));
        for arr in rk {
            for elem in arr.iter() {
                assert!(elem.abs() < 10.);
            }
        }
    }

    #[test]
    fn test_implicit_euler_2d() {
        let ie = implicit_euler(
            van_der_pol,
            van_der_pol_derivative,
            Vector2::new(1., 1.),
            0.25,
            100.,
        )
        .unwrap();
        assert!(ie[0] == Vector2::new(1., 1.));
        for arr in ie {
            for elem in arr.iter() {
                assert!(elem.abs() < 10.);
            }
        }
    }
}
