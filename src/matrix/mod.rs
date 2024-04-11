pub mod storage;
use storage::*;
pub use storage::{OwnedStorage, Storage, StorageMut};

#[macro_use]
pub mod vector;
pub use vector::{
    DVector, OVector, RowVector, RowVectorView, RowVectorViewMut, SVector, Vector, VectorStorage,
};

mod ops;

use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Mul},
};

use num_traits::{One, Zero};

pub trait Dim: 'static + Clone + Copy + Debug + Eq {
    fn dim(&self) -> usize;

    fn from_dim(dim: usize) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Const<const D: usize>;
impl<const D: usize> Dim for Const<D> {
    fn dim(&self) -> usize {
        D
    }

    fn from_dim(_dim: usize) -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyn(usize);
impl Dim for Dyn {
    fn dim(&self) -> usize {
        self.0
    }

    fn from_dim(dim: usize) -> Self {
        Self(dim)
    }
}

pub trait SameDim<D: Dim>: Dim {}

impl<D: Dim> SameDim<D> for D {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    pub data: S,
    phantom: PhantomData<(T, R, C)>,
}

pub type OMatrix<T, R, C> = Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Storage>;
pub type SMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
pub type DMatrix<T> = Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>;

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Create a new matrix without checking for consistency between
    /// the storage's and matrix's dimensions.
    ///
    /// # Safety
    /// Providing a storage whose dimensions is not equal to the matrix's dimension is undefined behavior.
    pub unsafe fn from_data_unchecked(data: S) -> Self {
        Self {
            data,
            phantom: PhantomData,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        let (rows, columns) = self.data.shape();
        (rows.dim(), columns.dim())
    }

    pub fn shape_generic(&self) -> (R, C) {
        self.data.shape()
    }

    pub fn row(&self, index: usize) -> RowVectorView<'_, T, C> {
        self.data.row(index)
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    pub fn row_mut(&mut self, index: usize) -> RowVectorViewMut<'_, T, C> {
        self.data.row_mut(index)
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    T: Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C, Storage = S>,
{
    pub fn zeros_generic(rows: R, columns: C) -> Self {
        let iter = (0..rows.dim() * columns.dim()).map(|_| T::zero());
        let data = DefaultAllocator::from_iter(iter, rows, columns);

        unsafe { Self::from_data_unchecked(data) }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    T: Clone,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    pub fn fill(&mut self, elem: T) {
        for i in self.iter_mut() {
            *i = elem.clone();
        }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    T: Zero + One,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C, Storage = S>,
{
    pub fn identity_generic(rows: R, columns: C) -> Self {
        let iter = (0..rows.dim() * columns.dim()).map(|k| {
            let i = k / columns.dim();
            let j = k % columns.dim();
            if i == j {
                T::one()
            } else {
                T::zero()
            }
        });
        let data = DefaultAllocator::from_iter(iter, rows, columns);

        unsafe { Self::from_data_unchecked(data) }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    T: Clone,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    pub fn transpose(&self) -> Matrix<T, C, R, <DefaultAllocator as Allocator<T, C, R>>::Storage>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        let (rows_lhs, columns_lhs) = self.shape_generic();

        let mut storage = DefaultAllocator::new_uninit(columns_lhs, rows_lhs);

        for (i, mut row) in storage.row_iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                elem.write(self[[j, i]].clone());
            }
        }

        unsafe { Matrix::from_data_unchecked(DefaultAllocator::assume_init(storage)) }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    T: AddAssign<T>,
    for<'a, 'b> &'a T: Add<&'b T, Output = T> + Mul<&'b T, Output = T>,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    pub fn dot<R2, C2, S2>(
        &self,
        rhs: &Matrix<T, R2, C2, S2>,
    ) -> Matrix<T, R, C2, <DefaultAllocator as Allocator<T, R, C2>>::Storage>
    where
        R2: SameDim<C>,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<T, R, C2>,
    {
        let (rows_lhs, columns_lhs) = self.shape_generic();
        let (rows_rhs, columns_rhs) = rhs.shape_generic();

        assert_eq!(
            columns_lhs.dim(), rows_rhs.dim(),
            "Invalid shape for matrix multiplication: Expected left-hand side to have same number of columns as rows of right-hand side\nLeft-hand side: ({}, {}), right-hand side: ({}, {})",
            rows_lhs.dim(),
            columns_lhs.dim(),
            rows_rhs.dim(),
            columns_rhs.dim()
        );

        let mut storage = DefaultAllocator::new_uninit(rows_lhs, columns_rhs);

        for (i, mut row) in storage.row_iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                let mut sum = &self[[i, 0]] * &rhs[[0, j]];
                for k in 1..columns_lhs.dim() {
                    sum += &self[[i, k]] * &rhs[[k, j]];
                }
                elem.write(sum);
            }
        }

        unsafe { Matrix::from_data_unchecked(DefaultAllocator::assume_init(storage)) }
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageIterator<T, R, C>,
{
    pub fn iter(&self) -> S::Iter<'_> {
        self.data.iter()
    }

    pub fn row_iter(&self) -> S::RowIter<'_> {
        self.data.row_iter()
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageIteratorMut<T, R, C>,
{
    pub fn iter_mut(&mut self) -> S::IterMut<'_> {
        self.data.iter_mut()
    }

    pub fn row_iter_mut(&mut self) -> S::RowIterMut<'_> {
        self.data.row_iter_mut()
    }
}

impl<T> Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>> {
    pub fn reshape(self, rows: usize, columns: usize) -> Option<Self> {
        unsafe {
            Some(Self::from_data_unchecked(
                self.data.reshape(Dyn(rows), Dyn(columns))?,
            ))
        }
    }
}

impl<T, const R: usize, const C: usize, S> Matrix<T, Const<R>, Const<C>, S>
where
    T: Zero,
    S: RawStorage<T, Const<R>, Const<C>>,
    DefaultAllocator: Allocator<T, Const<R>, Const<C>, Storage = S>,
{
    pub fn zeros() -> Self {
        let iter = (0..R * C).map(|_| T::zero());
        let data = DefaultAllocator::from_iter(iter, Const, Const);

        unsafe { Self::from_data_unchecked(data) }
    }
}

impl<T, const R: usize, const C: usize, S> Matrix<T, Const<R>, Const<C>, S>
where
    T: Zero + One,
    S: RawStorage<T, Const<R>, Const<C>>,
    DefaultAllocator: Allocator<T, Const<R>, Const<C>, Storage = S>,
{
    pub fn identity() -> Self {
        let iter = (0..R * C).map(|k| {
            let i = k / C;
            let j = k % C;
            if i == j {
                T::one()
            } else {
                T::zero()
            }
        });
        let data = DefaultAllocator::from_iter(iter, Const, Const);

        unsafe { Self::from_data_unchecked(data) }
    }
}

impl<T, S> From<T> for Matrix<T, Const<1>, Const<1>, S>
where
    S: RawStorage<T, Const<1>, Const<1>> + From<T>,
{
    fn from(value: T) -> Self {
        unsafe { Matrix::from_data_unchecked(S::from(value)) }
    }
}

impl<T, R, C, S> IntoIterator for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: OwnedStorage<T, R, C>,
{
    type Item = T;

    type IntoIter = S::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T, R, C, S> Index<[usize; 2]> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &T {
        self.data.index(index)
    }
}

impl<T, R, C, S> IndexMut<[usize; 2]> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        self.data.index_mut(index)
    }
}

impl<T, R, C, S> Index<(usize, usize)> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        self.index([index.0, index.1])
    }
}

impl<T, R, C, S> IndexMut<(usize, usize)> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        self.index_mut([index.0, index.1])
    }
}

impl<T, R, C, S> Index<usize> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.data.index(index)
    }
}

impl<T, R, C, S> IndexMut<usize> for Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.data.index_mut(index)
    }
}

#[macro_export]
macro_rules! mat {
    [$([$($elem:expr),* $(,)*]),* $(,)*] => {
        {
            let arr = $crate::matrix::storage::ArrayStorage::from([$([$($elem),*]),*]);
            unsafe { $crate::matrix::Matrix::from_data_unchecked(arr) }
        }
    };
    [$($elem:expr),* $(,)*] => {
        {
            let arr = $crate::matrix::storage::ArrayStorage::from([[$($elem),*]]);
            unsafe { $crate::matrix::Matrix::from_data_unchecked(arr) }
        }
    };
}

#[cfg(test)]
mod tests {

    #[test]
    fn mat() {
        let mat = mat![[1, 2, 3], [4, 5, 6]];
        assert_eq!(mat.data.array, [[1, 2, 3], [4, 5, 6]]);
    }

    #[test]
    fn binary_ops() {
        let mat1 = mat![[1, 2, 3], [4, 5, 6]];
        let mat2 = mat![[6, 5, 4], [3, 2, 1]];

        let add = mat1.clone() + mat2.clone();
        assert_eq!(add.data.array, [[7, 7, 7], [7, 7, 7]]);

        let sub = mat1.clone() - mat2.clone();
        assert_eq!(sub.data.array, [[-5, -3, -1], [1, 3, 5]]);

        let mul = mat1.clone() * mat2.clone();
        assert_eq!(mul.data.array, [[6, 10, 12], [12, 10, 6]]);

        let div = mat1.clone() / mat2.clone();
        assert_eq!(div.data.array, [[0, 0, 0], [1, 2, 6]]);
    }

    #[test]
    fn binary_assigment_ops() {
        let mat1 = mat![[1, 2, 3], [4, 5, 6]];
        let mat2 = mat![[6, 5, 4], [3, 2, 1]];

        let mut add = mat1.clone();
        add += mat2.clone();
        assert_eq!(add.data.array, [[7, 7, 7], [7, 7, 7]]);

        let mut sub = mat1.clone();
        sub -= mat2.clone();
        assert_eq!(sub.data.array, [[-5, -3, -1], [1, 3, 5]]);

        let mut mul = mat1.clone();
        mul *= mat2.clone();
        assert_eq!(mul.data.array, [[6, 10, 12], [12, 10, 6]]);

        let mut div = mat1.clone();
        div /= mat2.clone();
        assert_eq!(div.data.array, [[0, 0, 0], [1, 2, 6]]);
    }

    #[test]
    fn transpose() {
        let mat = mat![[1, 2, 3], [4, 5, 6]];

        let transpose = mat.transpose();
        assert_eq!(transpose.data.array, [[1, 4], [2, 5], [3, 6]]);
    }

    #[test]
    fn dot() {
        let mat1 = mat![[1, 2, 3], [4, 5, 6]];
        let mat2 = vector![1, 2, 3];
        let mat3 = mat1.dot(&mat2);

        assert_eq!(mat3.data.array, [[14], [32]]);
    }
}
