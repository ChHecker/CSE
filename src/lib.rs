use nalgebra::{Const, OMatrix, RealField, SVector, Vector1};

pub mod algebraic_solvers;
pub mod containers;
#[cfg(test)]
pub(crate) mod csv;
pub mod fft;
pub mod graphs;
pub mod interpolation;
pub mod least_squares;
pub mod linalg;
pub mod ode_solvers;
pub mod quadrature;
pub mod sorting;

pub trait IntoVector<T: RealField + Copy, const D: usize>: Clone {
    fn into_vector(self) -> SVector<T, D>;
    fn from_vector(value: SVector<T, D>) -> Self;
}

impl<T: RealField + Copy> IntoVector<T, 1> for T {
    fn into_vector(self) -> SVector<T, 1> {
        Vector1::new(self)
    }

    fn from_vector(value: SVector<T, 1>) -> Self {
        value.x
    }
}

impl<T: RealField + Copy, const D: usize> IntoVector<T, D> for SVector<T, D> {
    fn into_vector(self) -> SVector<T, D> {
        self
    }

    fn from_vector(value: SVector<T, D>) -> Self {
        value
    }
}

pub trait IntoMatrix<T: RealField + Copy, const D: usize>: Clone {
    fn into_matrix(self) -> OMatrix<T, Const<D>, Const<D>>;
    fn from_matrix(value: OMatrix<T, Const<D>, Const<D>>) -> Self;
}

impl<T: RealField + Copy> IntoMatrix<T, 1> for T {
    fn into_matrix(self) -> OMatrix<T, Const<1>, Const<1>> {
        OMatrix::<T, Const<1>, Const<1>>::new(self)
    }

    fn from_matrix(value: OMatrix<T, Const<1>, Const<1>>) -> Self {
        value.x
    }
}

impl<T: RealField + Copy, const D: usize> IntoMatrix<T, D> for OMatrix<T, Const<D>, Const<D>> {
    fn into_matrix(self) -> OMatrix<T, Const<D>, Const<D>> {
        self
    }

    fn from_matrix(value: OMatrix<T, Const<D>, Const<D>>) -> Self {
        value
    }
}

pub enum IterativeResult<V> {
    Converged(V),
    MaxIterations(V),
    Failed,
}

impl<V> IterativeResult<V> {
    pub fn unwrap(self) -> V {
        match self {
            IterativeResult::Converged(v) => v,
            IterativeResult::MaxIterations(v) => v,
            IterativeResult::Failed => panic!("called unwrap on Failed"),
        }
    }

    pub fn successful_or<E>(self, err: E) -> Result<V, E> {
        match self {
            IterativeResult::Converged(v) => Ok(v),
            IterativeResult::MaxIterations(v) => Ok(v),
            IterativeResult::Failed => Err(err),
        }
    }

    pub fn converged_or<E>(self, err: E) -> Result<V, E> {
        match self {
            IterativeResult::Converged(v) => Ok(v),
            IterativeResult::MaxIterations(_) => Err(err),
            IterativeResult::Failed => Err(err),
        }
    }

    pub fn is_converged(&self) -> bool {
        match &self {
            IterativeResult::Converged(_) => true,
            IterativeResult::MaxIterations(_) | IterativeResult::Failed => false,
        }
    }

    pub fn is_successful(&self) -> bool {
        match &self {
            IterativeResult::Converged(_) | IterativeResult::MaxIterations(_) => true,
            IterativeResult::Failed => false,
        }
    }
}
