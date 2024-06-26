use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector};
use num_traits::Zero;
use twofloat::TwoFloat;

/* L x = b
 *
 * 1 0 0
 * x 1 0
 * x x 1
 */
fn forward_substitution<D: Dim>(l: &OMatrix<f64, D, D>, b: OVector<f64, D>) -> OVector<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    let n = l.shape().0;

    let mut x = b;
    for i in 0..n {
        for j in 0..i {
            x[i] -= l[(i, j)] * x[j];
        }
    }

    x
}

/* U x = b
 *
 * x x x
 * 0 x x
 * 0 0 x
 */
fn backward_substitution<D: Dim>(
    u: &OMatrix<f64, D, D>,
    b: OVector<f64, D>,
) -> Option<OVector<f64, D>>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D, Const<1>>,
{
    let n = u.shape().0;

    let mut x: OVector<f64, D> = b;
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= u[(i, j)] * x[j];
        }
        if u[(i, i)].abs() < 1e-16 {
            return None;
        }
        x[i] /= u[(i, i)];
    }

    Some(x)
}

fn double_double_matrix_mul<D: Dim>(a: &OMatrix<f64, D, D>, x: &OVector<f64, D>) -> OVector<f64, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D, Const<1>>,
{
    let dim = x.shape().0;
    let mut out = OVector::zeros_generic(x.shape_generic().0, Const);

    for i in 0..dim {
        let mut sum = TwoFloat::zero();
        for j in 0..dim {
            sum += TwoFloat::new_mul(a[(i, j)], x[j]);
        }
        out[i] = sum.into();
    }

    out
}

#[derive(Clone, Debug)]
pub struct Lu<D: Dim>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    pub l: OMatrix<f64, D, D>,
    pub u: OMatrix<f64, D, D>,
    a: OMatrix<f64, D, D>,
}

impl<D: Dim> Lu<D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    pub fn new(a: OMatrix<f64, D, D>) -> Option<Self> {
        assert!(a.shape().0 == a.shape().1);
        let n = a.shape().0;
        let n_gen = a.shape_generic().0;

        let mut l = OMatrix::<f64, D, D>::identity_generic(n_gen, n_gen);
        let mut u = a.clone();

        for i in 0..n {
            let current_row = u.row(i).clone_owned();
            let pivot_element = current_row[i];

            for (j, mut row) in u.row_iter_mut().enumerate().skip(i + 1) {
                if pivot_element.abs() < 1e-16 {
                    return None;
                }
                let factor = row[i] / pivot_element;
                l[(j, i)] = factor;
                row -= factor * &current_row;
            }
        }

        Some(Self { a, l, u })
    }

    /// Solve the equation $LUx = b$.
    pub fn solve(&self, b: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        let y = forward_substitution(&self.l, b.clone());
        backward_substitution(&self.u, y.clone())
    }

    pub fn solve_refine(&self, b: &OVector<f64, D>, epsilon: f64) -> Option<OVector<f64, D>> {
        let mut x = self.solve(b)?;

        let mut cond = epsilon + 1.;
        while cond > epsilon {
            let res = b - double_double_matrix_mul(&self.a, &x);
            let z = self.solve(&res)?;
            x += &z;
            cond = z.norm() / x.norm();
        }

        Some(x)
    }
}

#[derive(Clone, Debug)]
pub struct Qr<D: Dim>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    pub q: OMatrix<f64, D, D>,
    pub r: OMatrix<f64, D, D>,
}

impl<D: Dim> Qr<D>
where
    DefaultAllocator:
        Allocator<f64, D, D> + Allocator<f64, D, Const<1>> + Allocator<f64, Const<1>, D>,
{
    pub fn new(a: OMatrix<f64, D, D>) -> Self {
        assert!(a.shape().0 == a.shape().1);
        let n = a.shape().0;
        let n_gen = a.shape_generic().0;

        let mut q = OMatrix::<f64, D, D>::identity_generic(n_gen, n_gen);
        let mut r = a;

        for i in 0..n {
            let r_view = r.view((i, i), (n - i, n - i));
            let col_view = r_view.column(0);
            let mut goal = OVector::<f64, D>::zeros_generic(n_gen, Const);
            goal[i] = col_view.norm();
            let mut v = OVector::<f64, D>::zeros_generic(n_gen, Const);
            v.view_mut((i, 0), (n - i, 1)).copy_from(
                &(col_view + col_view[0].signum() * goal.view((i, 0), (n - i, 1))).normalize(),
            );

            r -= 2. * &v * (v.transpose() * &r);
            q -= 2. * (&q * &v) * v.transpose();
        }

        Self { q, r }
    }

    /// Solve the equation $QRx = b$.
    pub fn solve(&self, b: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        backward_substitution(&self.r, self.q.transpose() * b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::{Matrix3, Matrix6, SMatrix, Vector3, Vector6};
    use rand::Rng;

    fn random_pos_def_matrix() -> Matrix3<f64> {
        let rand: Matrix3<f64> = Matrix3::new_random();
        0.5 * (rand + rand.transpose()) + 3. * Matrix3::identity()
    }

    #[test]
    fn test_double_double_matrix_mul() {
        let a: Matrix6<f64> = 1e6 * Matrix6::new_random();
        let x: Vector6<f64> = 1e6 * Vector6::new_random();
        let b_double = a * x;
        let b_double_double = double_double_matrix_mul(&a, &x);
        assert_ne!(b_double, b_double_double);
    }

    #[test]
    fn lu_structure() {
        const D: usize = 5;
        let a: SMatrix<f64, D, D> = 100. * SMatrix::new_random();
        let sol = Lu::new(a).unwrap();
        let l = sol.l;
        let u = sol.u;

        for i in 0..D {
            for j in 0..D {
                match i.cmp(&j) {
                    std::cmp::Ordering::Less => assert_abs_diff_eq!(l[(i, j)], 0., epsilon = 1e-8),
                    std::cmp::Ordering::Equal => assert_abs_diff_eq!(l[(i, j)], 1., epsilon = 1e-8),
                    std::cmp::Ordering::Greater => {
                        assert_abs_diff_eq!(u[(i, j)], 0., epsilon = 1e-8)
                    }
                };
            }
        }
    }

    #[test]
    fn lu_consistency() {
        let a: SMatrix<f64, 5, 5> = 100. * SMatrix::new_random();
        let sol = Lu::new(a).unwrap();

        let c = sol.l * sol.u;
        assert_abs_diff_eq!(c, a, epsilon = 1e-8);
    }

    #[test]
    fn lu_solve_consistency() {
        let mut rng = rand::thread_rng();
        let l = Matrix3::new(1., 0., 0., rng.gen(), 1., 0., rng.gen(), rng.gen(), 1.);
        let u = Matrix3::new(
            rng.gen(),
            rng.gen(),
            rng.gen(),
            0.,
            rng.gen(),
            rng.gen(),
            0.,
            0.,
            rng.gen(),
        );
        let b = Vector3::new_random();

        let lu = Lu {
            l,
            u,
            a: SMatrix::identity(),
        };
        let x = lu.solve(&b).unwrap();
        let c = l * u * x;
        assert_abs_diff_eq!(c, b, epsilon = 1e-8);
    }

    #[test]
    fn lu_solve() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let lu = Lu::new(a).unwrap();
        let x = lu.solve(&b).unwrap();
        assert!((a * x - b).norm() < 1e-6);
    }

    #[test]
    fn lu_solve_refine() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let lu = Lu::new(a).unwrap();
        let x = lu.solve_refine(&b, 1e-6).unwrap();
        assert!((a * x - b).norm() < 1e-6);
    }

    #[test]
    fn qr_structure() {
        const D: usize = 5;
        let a: SMatrix<f64, D, D> = 100. * SMatrix::new_random();
        let sol = Qr::new(a);
        let q = sol.q;
        let r = sol.r;

        let identity = q * q.transpose();
        assert_abs_diff_eq!(identity, SMatrix::identity(), epsilon = 1e-8);

        for j in 0..D {
            for i in j + 1..D {
                assert_abs_diff_eq!(r[(i, j)], 0., epsilon = 1e-8)
            }
        }
    }

    #[test]
    fn qr_consistency() {
        let a: SMatrix<f64, 5, 5> = 100. * SMatrix::new_random();
        let sol = Qr::new(a);

        let c = sol.q * sol.r;
        assert_abs_diff_eq!(c, a, epsilon = 1e-8);
    }

    #[test]
    fn qr_solve_consistency() {
        let mut rng = rand::thread_rng();
        let v = Vector3::<f64>::new_random().normalize();
        let q = Matrix3::identity() - 2. * v * v.transpose();
        let r = Matrix3::new(
            rng.gen(),
            rng.gen(),
            rng.gen(),
            0.,
            rng.gen(),
            rng.gen(),
            0.,
            0.,
            rng.gen(),
        );
        let b = Vector3::new_random();

        let qr = Qr { q, r };
        let x = qr.solve(&b).unwrap();
        let c = q * r * x;
        assert_abs_diff_eq!(c, b, epsilon = 1e-8);
    }

    #[test]
    fn qr_solve() {
        let a = random_pos_def_matrix();
        let b: Vector3<f64> = Vector3::new_random();
        let lu = Qr::new(a);
        let x = lu.solve(&b).unwrap();
        assert!((a * x - b).norm() < 1e-6);
    }
}
