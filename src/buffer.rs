//! Structure-of-arrays typed column buffers (U01).

use core::ops::{Deref, DerefMut, Index, IndexMut};

/// Contiguous owned column buffer for SoA state storage.
///
/// One `Buffer<T>` holds a single typed field across all entities
/// (e.g. membrane voltages). Entities are addressed by index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Buffer<T> {
    data: Vec<T>,
}

impl<T> Buffer<T> {
    /// Empty buffer.
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Empty buffer with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Take ownership of an existing column vector.
    #[inline]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Build a column by evaluating `f` at each index in `0..len`.
    #[inline]
    pub fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self {
        Self {
            data: (0..len).map(f).collect(),
        }
    }

    /// Number of entities in this column.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the column has no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the underlying contiguous storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the underlying contiguous storage.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the buffer and return the owned column vector.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Append one entity value.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Grow or shrink to `new_len`, filling new slots with `value`.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(new_len, value);
    }

    /// Overwrite every entity with `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    /// Borrow entity `i`, if in range.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Mutably borrow entity `i`, if in range.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut(i)
    }
}

impl<T: Clone> Buffer<T> {
    /// Copy a slice into a new column buffer.
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    /// Column of `len` copies of `value`.
    #[inline]
    pub fn from_elem(len: usize, value: T) -> Self {
        Self {
            data: vec![value; len],
        }
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(data: Vec<T>) -> Self {
        Self::from_vec(data)
    }
}

impl<T> From<Buffer<T>> for Vec<T> {
    #[inline]
    fn from(buf: Buffer<T>) -> Self {
        buf.into_vec()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Buffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for Buffer<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn buffer_round_trip_vec() {
        let data = vec![1.0f32, -2.5, 3.25, 0.0];
        let buf = Buffer::from_vec(data.clone());
        assert_eq!(buf.len(), data.len());
        assert_eq!(buf.as_slice(), data.as_slice());
        assert_eq!(buf.into_vec(), data);
    }

    #[test]
    fn buffer_round_trip_slice() {
        let data = [10u32, 20, 30, 40];
        let buf = Buffer::from_slice(&data);
        assert_eq!(buf.as_slice(), &data);
        let out: Vec<u32> = buf.into();
        assert_eq!(out.as_slice(), &data);
    }

    #[test]
    fn buffer_from_fn_and_index() {
        let buf = Buffer::from_fn(4, |i| (i as i32) * 2);
        assert_eq!(&*buf, &[0, 2, 4, 6]);
        assert_eq!(buf[2], 4);
    }
}
