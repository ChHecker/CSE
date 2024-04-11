use std::{
    fmt,
    ops::{Deref, DerefMut},
    ptr::slice_from_raw_parts,
    slice,
};

use self::vec::{RowIter, RowIterMut};

use super::*;

#[cfg(all(not(target_pointer_width = "32"), not(target_pointer_width = "64")))]
compile_error!("Your pointers are too small. Please try again with a more expensive computer.");

#[repr(C)]
pub struct ViewStorage<T, R: Dim, C: Dim>(PhantomData<(R, C)>, [T]);

impl<T, R: Dim, C: Dim> ViewStorage<T, R, C> {
    #[cfg(target_pointer_width = "32")]
    pub unsafe fn from_raw_parts_generic<'a>(ptr: *const T, rows: R, columns: C) -> &'a Self {
        let (rows, columns) = (rows.dim(), columns.dim());
        let size = (rows << 16) + (columns & 0xFFFF);
        let slice = slice_from_raw_parts(ptr, size);
        &*(slice as *const Self)
    }

    #[cfg(target_pointer_width = "32")]
    pub unsafe fn from_raw_parts_mut_generic<'a>(ptr: *mut T, rows: R, columns: C) -> &'a mut Self {
        let (rows, columns) = (rows.dim(), columns.dim());
        let size = (rows << 16) + (columns & 0xFFFF);
        let slice = slice_from_raw_parts(ptr, size);
        &mut *(slice as *mut Self)
    }

    #[cfg(target_pointer_width = "64")]
    pub unsafe fn from_raw_parts_generic<'a>(ptr: *const T, rows: R, columns: C) -> &'a Self {
        let (rows, columns) = (rows.dim(), columns.dim());
        let size = (rows << 32) + (columns & 0xFFFFFFFF);
        let slice = slice_from_raw_parts(ptr, size);
        &*(slice as *const Self)
    }

    #[cfg(target_pointer_width = "64")]
    pub unsafe fn from_raw_parts_mut_generic<'a>(ptr: *mut T, rows: R, columns: C) -> &'a mut Self {
        let (rows, columns) = (rows.dim(), columns.dim());
        let size = (rows << 32) + (columns & 0xFFFFFFFF);
        let slice = slice_from_raw_parts(ptr, size);
        &mut *(slice as *mut Self)
    }

    pub fn as_slice(&self) -> &[T] {
        let (rows, columns) = self.shape();
        unsafe { slice::from_raw_parts(self.1.as_ptr(), rows.dim() * columns.dim()) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let (rows, columns) = self.shape();
        unsafe { slice::from_raw_parts_mut(self.1.as_mut_ptr(), rows.dim() * columns.dim()) }
    }
}

impl<T, R: Dim, C: Dim> Deref for ViewStorage<T, R, C> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, R: Dim, C: Dim> DerefMut for ViewStorage<T, R, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T: fmt::Debug, R: Dim, C: Dim> fmt::Debug for ViewStorage<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, columns) = self.shape();

        fmt::Formatter::debug_struct(f, "View")
            .field("data", &self.as_slice())
            .field("rows", &rows)
            .field("columns", &columns)
            .finish()
    }
}

macro_rules! impl_view_storage {
    ($name:ty) => {
        impl<'a, T, R: Dim, C: Dim> RawStorage<T, R, C> for $name {
            fn shape(&self) -> (R, C) {
                let len = self.1.len();
                let rows = len >> 32;
                let columns = len & 0xFFFFFFFF;
                (R::from_dim(rows), C::from_dim(columns))
            }

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
                        &self[index * columns.dim()],
                        Const,
                        columns,
                    ))
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim> StorageIterator<T, R, C> for $name {
            type Iter<'b> = slice::Iter<'b, T>
            where
                Self: 'b,
                T: 'b;

            type RowIter<'b> = RowIter<'b, T, C>
            where
                Self: 'b,
                T: 'b;

            fn iter(&self) -> slice::Iter<'_, T> {
                self.as_slice().iter()
            }

            fn row_iter(&self) -> RowIter<'_, T, C> {
                let columns = self.shape().1;
                RowIter {
                    iter: self.chunks(columns.dim()),
                    columns,
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim> Index<[usize; 2]> for $name {
            type Output = T;

            fn index(&self, index: [usize; 2]) -> &T {
                self.index(index[0] * self.shape().1.dim() + index[1])
            }
        }

        impl<'a, T, R: Dim, C: Dim> Index<usize> for $name {
            type Output = T;

            fn index(&self, index: usize) -> &T {
                self.as_slice().index(index)
            }
        }
    };
}

impl_view_storage!(&'a ViewStorage<T, R, C>);
impl_view_storage!(&'a mut ViewStorage<T, R, C>);

impl<T: Clone, R: Dim, C: Dim> ToOwned for ViewStorage<T, R, C> {
    type Owned = VecStorage<T, R, C>;

    fn to_owned(&self) -> Self::Owned {
        let (rows, columns) = self.shape();
        let data: Vec<T> = self.deref().to_owned();
        unsafe { VecStorage::new_unchecked(data, rows, columns) }
    }
}

impl<'a, T, R: Dim, C: Dim> RawStorageMut<T, R, C> for &'a mut ViewStorage<T, R, C> {
    fn row_mut(&mut self, index: usize) -> RowVectorViewMut<'_, T, C> {
        let (_, columns) = self.shape();

        unsafe {
            RowVector::from_data_unchecked(ViewStorage::from_raw_parts_mut_generic(
                &mut self[index * columns.dim()],
                Const,
                columns,
            ))
        }
    }
}

impl<'a, T, R: Dim, C: Dim> StorageIteratorMut<T, R, C> for &'a mut ViewStorage<T, R, C> {
    type IterMut<'b> = slice::IterMut<'b, T>
    where
        Self: 'b,
        T: 'b;

    type RowIterMut<'b> = RowIterMut<'b, T, C>
    where
        Self: 'b,
        T: 'b;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.as_slice_mut().iter_mut()
    }

    fn row_iter_mut(&mut self) -> Self::RowIterMut<'_> {
        let columns = self.shape().1;
        RowIterMut {
            iter: self.as_slice_mut().chunks_mut(columns.dim()),
            columns,
        }
    }
}

impl<'a, T, R: Dim, C: Dim> IndexMut<[usize; 2]> for &'a mut ViewStorage<T, R, C> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let columns = self.shape().1.dim();
        self.index_mut(index[0] * columns + index[1])
    }
}

impl<'a, T, R: Dim, C: Dim> IndexMut<usize> for &'a mut ViewStorage<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.as_slice_mut().index_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::{Borrow, BorrowMut};

    use super::*;

    #[test]
    fn index() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let view: &ViewStorage<_, _, _> = arr.borrow();

        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[0, 1]], 2);
        assert_eq!(view[[0, 2]], 3);
        assert_eq!(view[[1, 0]], 4);
        assert_eq!(view[[1, 1]], 5);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn linear_index() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let view: &ViewStorage<_, _, _> = arr.borrow();

        assert_eq!(view[0], 1);
        assert_eq!(view[1], 2);
        assert_eq!(view[2], 3);
        assert_eq!(view[3], 4);
        assert_eq!(view[4], 5);
        assert_eq!(view[5], 6);
    }

    #[test]
    fn iter() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let view: &ViewStorage<_, _, _> = arr.borrow();
        for (i, elem) in view.iter().enumerate() {
            assert_eq!(i + 1, *elem);
        }
    }

    #[test]
    fn iter_mut() {
        let mut arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let mut view: &mut ViewStorage<_, _, _> = arr.borrow_mut();
        for (i, elem) in view.iter_mut().enumerate() {
            assert_eq!(i + 1, *elem);
        }
    }

    #[test]
    fn row_iter() {
        let arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let view: &ViewStorage<_, _, _> = arr.borrow();
        let mut iter = view.row_iter();

        assert_eq!(iter.next().unwrap().data.as_slice(), &[1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn row_iter_mut() {
        let mut arr = ArrayStorage::from([[1, 2, 3], [4, 5, 6]]);
        let mut view: &mut ViewStorage<_, _, _> = arr.borrow_mut();
        let mut iter = view.row_iter_mut();

        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [1, 2, 3]);
        assert_eq!(iter.next().unwrap().data.as_slice(), &mut [4, 5, 6]);
    }
}
