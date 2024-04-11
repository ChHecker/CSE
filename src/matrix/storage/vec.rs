use core::slice;
use std::borrow::{Borrow, BorrowMut};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecStorage<T, R: Dim, C: Dim> {
    pub vec: Vec<T>,
    rows: R,
    columns: C,
}

impl<T, R: Dim, C: Dim> VecStorage<T, R, C> {
    pub fn new(vec: Vec<T>, rows: R, columns: C) -> Self {
        assert_eq!(vec.len(), rows.dim() * columns.dim());

        Self { vec, rows, columns }
    }

    /// Create a new vec storage without checking for correct vec length.
    ///
    /// # Safety
    /// Providing a vector whose length is not equal `rows * columns` is undefined behavior.
    pub unsafe fn new_unchecked(vec: Vec<T>, rows: R, columns: C) -> Self {
        Self { vec, rows, columns }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i >= self.rows.dim() || j >= self.columns.dim() {
            return None;
        }
        Some(self.index([i, j]))
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i >= self.rows.dim() || j >= self.columns.dim() {
            return None;
        }
        Some(self.index_mut([i, j]))
    }

    pub fn reshape<R2: Dim, C2: Dim>(self, rows: R2, columns: C2) -> Option<VecStorage<T, R2, C2>> {
        if self.rows.dim() * self.columns.dim() != rows.dim() * columns.dim() {
            return None;
        }

        unsafe { Some(VecStorage::new_unchecked(self.vec, rows, columns)) }
    }
}

impl<T, R: Dim, C: Dim> RawStorage<T, R, C> for VecStorage<T, R, C> {
    fn shape(&self) -> (R, C) {
        (self.rows, self.columns)
    }

    #[inline(always)]
    fn assert_same_shape<R2, C2, S2>(&self, other: &S2)
    where
        R2: SameDim<R>,
        C2: SameDim<C>,
        S2: Storage<T, R2, C2>,
    {
        let (rows_a, columns_a) = self.shape();
        let (rows_b, columns_b) = other.shape();
        let same_shape = rows_a.dim() == rows_b.dim() && columns_a.dim() == columns_b.dim();

        assert!(
            same_shape,
            "Invalid shape: Expected same dimensions\nLeft-hand side: ({}, {}), right-hand side: ({}, {})",
            rows_a.dim(),
            columns_a.dim(),
            rows_b.dim(),
            columns_b.dim()
        )
    }

    fn row(&self, index: usize) -> RowVectorView<'_, T, C> {
        let (_, columns) = self.shape();

        unsafe {
            RowVector::from_data_unchecked(ViewStorage::from_raw_parts_generic(
                &self.vec[index * self.columns.dim()],
                Const,
                columns,
            ))
        }
    }
}

impl<T, R: Dim, C: Dim> RawStorageMut<T, R, C> for VecStorage<T, R, C> {
    fn row_mut(&mut self, index: usize) -> RowVectorViewMut<'_, T, C> {
        let (_, columns) = self.shape();

        unsafe {
            RowVector::from_data_unchecked(ViewStorage::from_raw_parts_mut_generic(
                &mut self.vec[index * self.columns.dim()],
                Const,
                columns,
            ))
        }
    }
}

impl<T, R: Dim, C: Dim> From<VecStorage<T, R, C>> for Vec<T> {
    fn from(value: VecStorage<T, R, C>) -> Self {
        value.vec
    }
}

impl<T> From<T> for VecStorage<T, Const<1>, Const<1>> {
    fn from(value: T) -> Self {
        unsafe { Self::new_unchecked(vec![value], Const, Const) }
    }
}

impl<T, R: Dim, C: Dim> IntoIterator for VecStorage<T, R, C> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<T, R: Dim, C: Dim> StorageIterator<T, R, C> for VecStorage<T, R, C> {
    type Iter<'a> = slice::Iter<'a, T>
    where
        T: 'a;

    type RowIter<'a> = RowIter<'a, T, C>
    where
        T: 'a;

    fn iter(&self) -> slice::Iter<'_, T> {
        self.vec.iter()
    }

    fn row_iter(&self) -> RowIter<'_, T, C> {
        RowIter {
            iter: self.vec.chunks(self.columns.dim()),
            columns: self.columns,
        }
    }
}

impl<T, R: Dim, C: Dim> StorageIteratorMut<T, R, C> for VecStorage<T, R, C> {
    type IterMut<'a> = slice::IterMut<'a, T>
    where
        T: 'a;

    type RowIterMut<'a> = RowIterMut<'a, T, C>
    where
        T: 'a;

    fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.vec.iter_mut()
    }

    fn row_iter_mut(&mut self) -> RowIterMut<'_, T, C> {
        RowIterMut {
            iter: self.vec.chunks_mut(self.columns.dim()),
            columns: self.columns,
        }
    }
}

pub struct RowIter<'a, T, C: Dim> {
    pub(crate) iter: slice::Chunks<'a, T>,
    pub(crate) columns: C,
}

impl<'a, T, C: Dim> Iterator for RowIter<'a, T, C> {
    type Item = RowVectorView<'a, T, C>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            Some(Matrix::from_data_unchecked(
                ViewStorage::from_raw_parts_generic(&self.iter.next()?[0], Const, self.columns),
            ))
        }
    }
}

pub struct RowIterMut<'a, T, C: Dim> {
    pub(crate) iter: slice::ChunksMut<'a, T>,
    pub(crate) columns: C,
}

impl<'a, T, C: Dim> Iterator for RowIterMut<'a, T, C> {
    type Item = RowVectorViewMut<'a, T, C>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            Some(Matrix::from_data_unchecked(
                ViewStorage::from_raw_parts_mut_generic(
                    &mut self.iter.next()?[0],
                    Const,
                    self.columns,
                ),
            ))
        }
    }
}

impl<T, R: Dim, C: Dim> Index<[usize; 2]> for VecStorage<T, R, C> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &T {
        &self.vec[index[0] * self.columns.dim() + index[1]]
    }
}

impl<T, R: Dim, C: Dim> IndexMut<[usize; 2]> for VecStorage<T, R, C> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        &mut self.vec[index[0] * self.columns.dim() + index[1]]
    }
}

impl<T, R: Dim, C: Dim> Index<usize> for VecStorage<T, R, C> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.vec[index]
    }
}

impl<T, R: Dim, C: Dim> IndexMut<usize> for VecStorage<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.vec[index]
    }
}

impl<T, R: Dim, C: Dim> Borrow<ViewStorage<T, R, C>> for VecStorage<T, R, C> {
    fn borrow(&self) -> &ViewStorage<T, R, C> {
        unsafe { ViewStorage::from_raw_parts_generic(self.vec.as_ptr(), self.rows, self.columns) }
    }
}

impl<T, R: Dim, C: Dim> BorrowMut<ViewStorage<T, R, C>> for VecStorage<T, R, C> {
    fn borrow_mut(&mut self) -> &mut ViewStorage<T, R, C> {
        unsafe {
            ViewStorage::from_raw_parts_mut_generic(self.vec.as_mut_ptr(), self.rows, self.columns)
        }
    }
}

impl<T, R: Dim, C: Dim> AsRef<ViewStorage<T, R, C>> for VecStorage<T, R, C> {
    fn as_ref(&self) -> &ViewStorage<T, R, C> {
        unsafe { ViewStorage::from_raw_parts_generic(self.vec.as_ptr(), self.rows, self.columns) }
    }
}

impl<T, R: Dim, C: Dim> AsMut<ViewStorage<T, R, C>> for VecStorage<T, R, C> {
    fn as_mut(&mut self) -> &mut ViewStorage<T, R, C> {
        unsafe {
            ViewStorage::from_raw_parts_mut_generic(self.vec.as_mut_ptr(), self.rows, self.columns)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index() {
        let vec: VecStorage<i32, Dyn, Dyn> = VecStorage::new((1..=6).collect(), Dyn(2), Dyn(3));

        assert_eq!(vec[[0, 0]], 1);
        assert_eq!(vec[[0, 1]], 2);
        assert_eq!(vec[[0, 2]], 3);
        assert_eq!(vec[[1, 0]], 4);
        assert_eq!(vec[[1, 1]], 5);
        assert_eq!(vec[[1, 2]], 6);
    }

    #[test]
    fn row_iter() {
        let vec = VecStorage::new(vec![1, 2, 3, 4, 5, 6], Dyn(2), Dyn(3));
        let mut iter = vec.row_iter();

        assert_eq!(iter.next().unwrap().data.as_slice(), &[1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn row_iter_mut() {
        let mut vec = VecStorage::new(vec![1, 2, 3, 4, 5, 6], Dyn(2), Dyn(3));
        let mut iter = vec.row_iter_mut();

        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [4, 5, 6]);
    }
}
