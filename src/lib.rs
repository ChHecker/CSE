use nalgebra::{Const, OMatrix, RealField, SVector, Vector1};

pub mod containers;
#[cfg(test)]
pub(crate) mod csv;
pub mod fft;
pub mod graphs;
pub mod interpolation;
pub mod linalg;
pub mod newton;
pub mod ode;
pub mod parser;
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
