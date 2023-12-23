use nalgebra::{allocator::Allocator, Const, DefaultAllocator, OMatrix, OVector, SVector};

/* L x = b
 *
 * 1 0 0
 * x 1 0
 * x x 1
 */
fn forward_substitution<const D: usize>(
    l: &OMatrix<f64, Const<D>, Const<D>>,
    b: OVector<f64, Const<D>>,
) -> OVector<f64, Const<D>> {
    let mut x = b;
    for i in 0..D {
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
fn backward_substitution<const D: usize>(
    u: &OMatrix<f64, Const<D>, Const<D>>,
    b: OVector<f64, Const<D>>,
) -> Option<OVector<f64, Const<D>>>
where
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>> + Allocator<f64, Const<D>, Const<1>>,
{
    let mut x: OVector<f64, Const<D>> = b;
    for i in (0..D).rev() {
        for j in i + 1..D {
            x[i] -= u[(i, j)] * x[j];
        }
        if u[(i, i)].abs() < 1e-16 {
            return None;
        }
        x[i] /= u[(i, i)];
    }

    Some(x)
}

#[derive(Clone, Debug)]
pub struct Lu<const D: usize>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, Const<D>, Const<D>>,
{
    pub l: OMatrix<f64, Const<D>, Const<D>>,
    pub u: OMatrix<f64, Const<D>, Const<D>>,
}

impl<const D: usize> Lu<D>
where
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>> + Allocator<f64, Const<D>, Const<1>>,
{
    pub fn new(a: OMatrix<f64, Const<D>, Const<D>>) -> Option<Self> {
        let mut l = OMatrix::<f64, Const<D>, Const<D>>::identity();
        let mut u = a;

        for i in 0..D {
            let current_row = u.row(i).clone_owned();
            let pivot_element = current_row[i];

            for (j, mut row) in u.row_iter_mut().enumerate().skip(i + 1) {
                if pivot_element.abs() < 1e-16 {
                    return None;
                }
                let factor = row[i] / pivot_element;
                l[(j, i)] = factor;
                row -= factor * current_row;
            }
        }

        Some(Self { l, u })
    }

    /// Solve the equation $LUx = b$.
    pub fn solve(&self, b: &OVector<f64, Const<D>>) -> Option<OVector<f64, Const<D>>> {
        let y = forward_substitution(&self.l, b.clone());
        backward_substitution(&self.u, y.clone())
    }
}

#[derive(Clone, Debug)]
pub struct Qr<const D: usize>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, Const<D>, Const<D>>,
{
    pub q: OMatrix<f64, Const<D>, Const<D>>,
    pub r: OMatrix<f64, Const<D>, Const<D>>,
}

impl<const D: usize> Qr<D>
where
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>> + Allocator<f64, Const<1>, Const<D>>,
{
    pub fn new(a: OMatrix<f64, Const<D>, Const<D>>) -> Self {
        let mut q = OMatrix::<f64, Const<D>, Const<D>>::identity();
        let mut r = a;

        for i in 0..D {
            let r_view = r.view((i, i), (D - i, D - i));
            let col_view = r_view.column(0);
            let mut goal = SVector::<f64, D>::zeros();
            goal[i] = col_view.norm();
            let mut v = SVector::<f64, D>::zeros();
            v.view_mut((i, 0), (D - i, 1)).copy_from(
                &(col_view + col_view[0].signum() * goal.view((i, 0), (D - i, 1))).normalize(),
            );

            r -= 2. * &v * (v.transpose() * &r);
            q -= 2. * (&q * v) * v.transpose();
        }

        Self { q, r }
    }

    /// Solve the equation $QRx = b$.
    pub fn solve(&self, b: &OVector<f64, Const<D>>) -> Option<OVector<f64, Const<D>>> {
        backward_substitution(&self.r, self.q.transpose() * b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::{Matrix3, SMatrix, Vector3};
    use rand::Rng;

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

        let lu = Lu { l, u };
        let x = lu.solve(&b).unwrap();
        let c = l * u * x;
        assert_abs_diff_eq!(c, b, epsilon = 1e-8);
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
}
