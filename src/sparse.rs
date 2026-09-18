//! Sparse connectivity in CSR and CSC form, validated on construction.
//!
//! [`Csr`] holds the forward graph, [`Csc`] its transpose for `y += Aᵀ·x`.
//! Both validate structurally when built: monotonic row pointers, every
//! column index inside the declared column count, and non-decreasing column
//! indices within each row (duplicates are multi-edges and are allowed).
//!
//! That validation is the point rather than a courtesy. An out-of-range column
//! index reaches the GPU as an out-of-bounds read, which on a GPU is not a
//! fault but a plausible number gathered from somewhere else in the buffer.
//! Checking once at construction is cheaper than checking per edge in the
//! kernel, and it makes the failure a caller error at a line the caller wrote
//! instead of a wrong answer several dispatches later.
//!
//! [`Csc::from_csr_rect`] takes an explicit column count, because a rectangular
//! operator's transpose cannot infer it from the row count.

/// Error returned when CSR parts fail structural validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CsrError {
    /// `row_ptr` must contain at least one entry (`[0]` for an empty graph).
    EmptyRowPtr,
    /// `row_ptr[0]` must be `0`.
    NonZeroStart { start: u32 },
    /// `row_ptr` must be monotonically non-decreasing.
    NotMonotonic { index: usize },
    /// `row_ptr[last]` must equal `col.len()`.
    NnzMismatch { row_ptr_end: u32, col_len: usize },
    /// A stored column index is at or past the declared column count.
    ColumnOutOfRange { col: u32, ncols: usize },
    /// Column indices within a row must be non-decreasing.
    ///
    /// Duplicates are allowed (multi-edges / separate synapses). A decrease
    /// at `edge` inside `row` is rejected so GPU gathers see coalesced order.
    RowUnsorted { row: usize, edge: usize },
    /// A CSC build step overflowed a `u32` degree, column pointer, or edge id.
    OffsetOverflow { what: &'static str },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyRowPtr => write!(f, "CSR row_ptr must be non-empty"),
            Self::NonZeroStart { start } => {
                write!(f, "CSR row_ptr must start at 0, got {start}")
            }
            Self::NotMonotonic { index } => {
                write!(f, "CSR row_ptr not monotonic at index {index}")
            }
            Self::ColumnOutOfRange { col, ncols } => {
                write!(f, "CSR column {col} is out of range for ncols={ncols}")
            }
            Self::NnzMismatch {
                row_ptr_end,
                col_len,
            } => write!(
                f,
                "CSR row_ptr end ({row_ptr_end}) != col.len() ({col_len})"
            ),
            Self::RowUnsorted { row, edge } => write!(
                f,
                "CSR row {row} is unsorted at edge {edge} (columns must be non-decreasing)"
            ),
            Self::OffsetOverflow { what } => {
                write!(f, "CSR/CSC {what} overflowed the u32 index range")
            }
        }
    }
}

impl std::error::Error for CsrError {}

/// Compressed-sparse-row connectivity graph.
///
/// `row_ptr` has length `nrows + 1`. Neighbors of row `r` are the column
/// indices `col[row_ptr[r] as usize .. row_ptr[r + 1] as usize]`. The stored
/// [`Csr::ncols`] is the declared operator width and may exceed
/// `max(col) + 1` when trailing columns are empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Csr {
    row_ptr: Vec<u32>,
    col: Vec<u32>,
    ncols: usize,
}

impl Default for Csr {
    /// Return the structurally valid zero-row CSR (`row_ptr == [0]`, `ncols == 0`).
    #[inline]
    fn default() -> Self {
        Self::empty(0, 0)
    }
}

impl Csr {
    /// Build from explicit arrays after validating CSR invariants.
    ///
    /// `ncols` is the declared operator width. It is stored on the CSR so
    /// trailing empty columns are not silently truncated by [`Csr::ncols`].
    pub fn from_parts(row_ptr: Vec<u32>, col: Vec<u32>, ncols: usize) -> Result<Self, CsrError> {
        validate(&row_ptr, &col, ncols)?;
        Ok(Self {
            row_ptr,
            col,
            ncols,
        })
    }

    /// Build from explicit arrays without validation (caller guarantees shape).
    ///
    /// An unchecked CSR still cannot reach a device kernel: [`crate::Device::prepare`]
    /// re-validates structure, row order, and `csr.ncols() == ncols`.
    #[inline]
    pub fn from_parts_unchecked(row_ptr: Vec<u32>, col: Vec<u32>, ncols: usize) -> Self {
        Self {
            row_ptr,
            col,
            ncols,
        }
    }

    /// Build from per-row adjacency lists.
    ///
    /// Each row is **stable-sorted** so column indices are non-decreasing.
    /// Duplicate column indices are preserved (multi-edges). The stored
    /// column count is `max(nrows, max_col + 1)`, or `nrows` when there are
    /// no edges.
    ///
    /// # Panics
    ///
    /// If the row-pointer table cannot be represented by `usize`, or if the
    /// total non-zero count cannot be represented by the CSR format's `u32`
    /// offsets. These cases cannot be returned from this historically
    /// infallible constructor; rejecting them before allocation preserves its
    /// signature without allowing a wrapped, valid-looking CSR.
    pub fn from_adjacency(rows: &[Vec<u32>]) -> Self {
        let nrows = rows.len();
        let row_ptr_len = nrows
            .checked_add(1)
            .expect("CSR row-pointer length overflowed usize");
        let nnz = rows
            .iter()
            .try_fold(0usize, |total, row| total.checked_add(row.len()))
            .expect("CSR adjacency non-zero count overflowed usize");
        u32::try_from(nnz)
            .expect("CSR adjacency has more non-zeros than u32 offsets can represent");

        let mut row_ptr = Vec::with_capacity(row_ptr_len);
        let mut col = Vec::with_capacity(nnz);
        let mut max_col: Option<u32> = None;
        row_ptr.push(0);
        for row in rows {
            let start = col.len();
            col.extend_from_slice(row);
            col[start..].sort(); // stable: preserves equal multi-edges' relative order
            if let Some(&last) = col[start..].last() {
                max_col = Some(max_col.map_or(last, |m| m.max(last)));
            }
            row_ptr.push(
                u32::try_from(col.len())
                    .expect("CSR adjacency offset exceeded the prevalidated u32 range"),
            );
        }
        let ncols = match max_col {
            Some(m) => nrows.max(m as usize + 1),
            None => nrows,
        };
        Self {
            row_ptr,
            col,
            ncols,
        }
    }

    /// Empty graph with `nrows` rows, `ncols` columns, and no edges.
    ///
    /// # Panics
    ///
    /// If `nrows + 1` cannot be represented by `usize`. The pointer-table
    /// length is checked explicitly so release builds cannot wrap it to zero
    /// and return an invalid, apparently empty CSR.
    pub fn empty(nrows: usize, ncols: usize) -> Self {
        let row_ptr_len = nrows
            .checked_add(1)
            .expect("CSR row-pointer length overflowed usize");
        Self {
            row_ptr: vec![0; row_ptr_len],
            col: Vec::new(),
            ncols,
        }
    }

    /// Row-pointer table (`nrows + 1` entries).
    #[inline]
    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    /// Stored column indices (`nnz` entries), row-major and non-decreasing per row
    /// when the CSR was built through a validating constructor.
    #[inline]
    pub fn col(&self) -> &[u32] {
        &self.col
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Number of stored non-zeros (edges).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    /// Declared number of columns (operator width).
    ///
    /// O(1). May exceed `max(col) + 1` when trailing columns are empty.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Column indices of neighbors for `row`.
    ///
    /// Panics if `row >= nrows()`.
    #[inline]
    pub fn row_cols(&self, row: usize) -> &[u32] {
        let start = self.row_ptr[row] as usize;
        let end = self.row_ptr[row + 1] as usize;
        &self.col[start..end]
    }

    /// Iterate neighbor column indices for `row`.
    ///
    /// Panics if `row >= nrows()`.
    #[inline]
    pub fn neighbors(&self, row: usize) -> impl Iterator<Item = u32> + '_ {
        self.row_cols(row).iter().copied()
    }

    /// Iterate `(row, col)` pairs over all edges in row-major order.
    pub fn edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.nrows()).flat_map(move |r| {
            let row = r as u32;
            self.neighbors(r).map(move |c| (row, c))
        })
    }

    /// Build a CSC reverse index over this CSR, assuming a **square** graph
    /// (`ncols == nrows`), which is what a recurrent cell graph is.
    ///
    /// Panics if any stored column is at or past `nrows`. For a rectangular
    /// operator — anything where the input and output dimensions differ — use
    /// [`Csr::to_csc_rect`], which takes the column count and returns an error
    /// instead.
    #[inline]
    pub fn to_csc(&self) -> Csc {
        Csc::from_csr(self)
    }

    /// Build a CSC reverse index over this CSR with an explicit column count.
    ///
    /// The general form of [`Csr::to_csc`]. A sparse layer whose input and
    /// output widths differ has no square assumption to fall back on, and the
    /// column count cannot be recovered from the CSR: `col` records which
    /// columns are *used*, not how many exist — except that this crate now
    /// stores [`Csr::ncols`] for that purpose; `ncols` here must still match
    /// the conversion width you intend.
    #[inline]
    pub fn to_csc_rect(&self, ncols: usize) -> Result<Csc, CsrError> {
        Csc::from_csr_rect(self, ncols)
    }
}

/// Compressed-sparse-column reverse index over CSR edge storage.
///
/// Columns are postsynaptic cells. Each CSC entry stores the presynaptic row and
/// the CSR edge index so synapse / weight tables stay CSR-ordered while
/// postsynaptic fan-in is `O(degree_in)` instead of `O(nnz)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Csc {
    /// Column pointers; length `ncols + 1`.
    pub col_ptr: Vec<u32>,
    /// Presynaptic (CSR row) index for each CSC entry.
    pub row: Vec<u32>,
    /// CSR edge index (synapse / weight table index) for each CSC entry.
    pub edge_idx: Vec<u32>,
}

impl Default for Csc {
    /// Return the structurally valid zero-column CSC (`col_ptr == [0]`).
    #[inline]
    fn default() -> Self {
        Self::empty(0)
    }
}

impl Csc {
    /// Empty reverse index with `ncols` columns and no edges.
    ///
    /// # Panics
    ///
    /// If `ncols + 1` cannot be represented by `usize`. The pointer-table
    /// length is checked explicitly so release builds cannot wrap it to zero
    /// and return an invalid, apparently empty CSC.
    pub fn empty(ncols: usize) -> Self {
        let col_ptr_len = ncols
            .checked_add(1)
            .expect("CSC column-pointer length overflowed usize");
        Self {
            col_ptr: vec![0; col_ptr_len],
            row: Vec::new(),
            edge_idx: Vec::new(),
        }
    }

    /// Build CSC fan-in from a **square** CSR graph (`ncols = csr.ncols()` when
    /// the graph was built as square; historically `csr.nrows()`).
    ///
    /// # Panics
    ///
    /// If any stored column is at or past the square width, or if a `u32`
    /// degree / pointer / edge id would wrap. That is a caller error — this
    /// constructor's whole premise is that the graph is square — but it is a
    /// panic only because it predates a rectangular caller. For anything that
    /// is not a recurrent cell graph, call [`Csc::from_csr_rect`], which takes
    /// the column count and returns [`CsrError`].
    pub fn from_csr(csr: &Csr) -> Self {
        // Prefer the stored width so trailing empty columns survive the
        // transpose. Fall back is unnecessary: `ncols` is always set.
        let width = csr.ncols();
        Self::from_csr_rect(csr, width).unwrap_or_else(|e| {
            panic!("CSC build over a square graph: {e}");
        })
    }

    /// Build CSC fan-in from a CSR with an explicit column count.
    ///
    /// The general form. `ncols` is the declared width of the operator, which
    /// is not recoverable from the stored column indices alone — `col` records
    /// which columns are used, not how many exist, so a matrix whose last
    /// columns are all empty is indistinguishable from a narrower one without
    /// the stored [`Csr::ncols`].
    ///
    /// Degree tallies, column pointers, and edge ids use `checked_add` /
    /// `u32::try_from` so a hostile unchecked CSR cannot wrap `u32` counters.
    pub fn from_csr_rect(csr: &Csr, ncols: usize) -> Result<Self, CsrError> {
        // `Csr` has an explicitly unchecked constructor, so a public API that
        // accepts `&Csr` cannot assume these invariants still hold. Besides
        // contradicting this module's contract, trusting `row_ptr` here used
        // to turn an nnz mismatch into an indexing panic below.
        validate(csr.row_ptr(), csr.col(), ncols)?;

        if ncols == 0 {
            // An operator with no columns can still have rows; it just has
            // nowhere to store an edge. A non-empty `col` here means the CSR
            // and the declared width disagree.
            return match csr.col().first() {
                Some(&col) => Err(CsrError::ColumnOutOfRange { col, ncols }),
                None => Ok(Self::empty(0)),
            };
        }
        let nnz = csr.nnz();
        let mut degrees = vec![0u32; ncols];
        for &c in csr.col() {
            let col = c as usize;
            if col >= ncols {
                return Err(CsrError::ColumnOutOfRange { col: c, ncols });
            }
            degrees[col] = degrees[col]
                .checked_add(1)
                .ok_or(CsrError::OffsetOverflow { what: "column degree" })?;
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        let mut acc = 0u32;
        for &d in &degrees {
            acc = acc
                .checked_add(d)
                .ok_or(CsrError::OffsetOverflow { what: "column pointer" })?;
            col_ptr.push(acc);
        }

        let mut row = vec![0u32; nnz];
        let mut edge_idx = vec![0u32; nnz];
        let mut next = col_ptr[..ncols].to_vec();
        for r in 0..csr.nrows() {
            let start = csr.row_ptr()[r] as usize;
            let end = csr.row_ptr()[r + 1] as usize;
            let row_u32 = u32::try_from(r).map_err(|_| CsrError::OffsetOverflow {
                what: "row index",
            })?;
            for e in start..end {
                let c = csr.col()[e] as usize;
                let slot = next[c] as usize;
                row[slot] = row_u32;
                edge_idx[slot] = u32::try_from(e).map_err(|_| CsrError::OffsetOverflow {
                    what: "edge index",
                })?;
                next[c] = next[c]
                    .checked_add(1)
                    .ok_or(CsrError::OffsetOverflow { what: "column cursor" })?;
            }
        }

        Ok(Self {
            col_ptr,
            row,
            edge_idx,
        })
    }

    /// Number of columns (postsynaptic cells).
    #[inline]
    pub fn ncols(&self) -> usize {
        self.col_ptr.len().saturating_sub(1)
    }

    /// Number of stored non-zeros (edges).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.edge_idx.len()
    }

    /// CSR edge indices of incoming synapses for postsynaptic `col`.
    ///
    /// Panics if `col >= ncols()`.
    #[inline]
    pub fn col_edge_indices(&self, col: usize) -> &[u32] {
        let start = self.col_ptr[col] as usize;
        let end = self.col_ptr[col + 1] as usize;
        &self.edge_idx[start..end]
    }

    /// Iterate `(pre, csr_edge_idx)` pairs for postsynaptic `col`.
    ///
    /// Panics if `col >= ncols()`.
    #[inline]
    pub fn incoming(&self, col: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let start = self.col_ptr[col] as usize;
        let end = self.col_ptr[col + 1] as usize;
        (start..end).map(move |i| (self.row[i], self.edge_idx[i]))
    }
}

fn validate(row_ptr: &[u32], col: &[u32], ncols: usize) -> Result<(), CsrError> {
    if row_ptr.is_empty() {
        return Err(CsrError::EmptyRowPtr);
    }
    if row_ptr[0] != 0 {
        return Err(CsrError::NonZeroStart { start: row_ptr[0] });
    }
    for i in 1..row_ptr.len() {
        if row_ptr[i] < row_ptr[i - 1] {
            return Err(CsrError::NotMonotonic { index: i });
        }
    }
    let end = *row_ptr.last().expect("row_ptr non-empty");
    if end as usize != col.len() {
        return Err(CsrError::NnzMismatch {
            row_ptr_end: end,
            col_len: col.len(),
        });
    }
    for &c in col {
        if c as usize >= ncols {
            return Err(CsrError::ColumnOutOfRange { col: c, ncols });
        }
    }
    // Non-decreasing within each row. Duplicates (multi-edges) are allowed.
    for row in 0..row_ptr.len().saturating_sub(1) {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for edge in start.saturating_add(1)..end {
            if col[edge] < col[edge - 1] {
                return Err(CsrError::RowUnsorted { row, edge });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Csc, Csr, CsrError};
    use proptest::prelude::*;

    #[test]
    fn empty_constructors_reject_dimension_overflow() {
        let csr = std::panic::catch_unwind(|| Csr::empty(usize::MAX, 0));
        assert!(
            csr.is_err(),
            "CSR empty constructor must reject n + 1 overflow instead of returning an invalid empty pointer table"
        );
        let csc = std::panic::catch_unwind(|| Csc::empty(usize::MAX));
        assert!(
            csc.is_err(),
            "CSC empty constructor must reject n + 1 overflow instead of returning an invalid empty pointer table"
        );
    }

    #[test]
    fn sparse_defaults_are_structurally_valid_zero_dimension_empties() {
        let csr = Csr::default();
        assert_eq!(csr, Csr::empty(0, 0));
        assert_eq!(
            Csr::from_parts(csr.row_ptr().to_vec(), csr.col().to_vec(), csr.ncols()),
            Ok(csr)
        );

        assert_eq!(Csc::default(), Csc::empty(0));
    }

    #[test]
    fn default_csr_converts_through_square_and_rectangular_apis() {
        let csr = Csr::default();
        let expected = Csc::empty(0);

        assert_eq!(csr.to_csc(), expected);
        assert_eq!(csr.to_csc_rect(0), Ok(expected));
    }

    /// A material fixture would require more than four billion `u32`s. Pin the
    /// narrowing boundary in source instead: a plain `as u32` silently wraps in
    /// release, while `try_from` is checked in every build profile.
    #[test]
    fn adjacency_offsets_are_checked_before_narrowing() {
        let source = include_str!("sparse.rs");
        let body = source
            .split_once("pub fn from_adjacency")
            .expect("from_adjacency source")
            .1
            .split_once("/// Empty graph")
            .expect("end of from_adjacency source")
            .0;
        assert!(
            body.contains("u32::try_from(col.len())"),
            "from_adjacency must validate each usize offset before storing it as u32"
        );
        assert!(
            !body.contains("row_ptr.push(col.len() as u32)"),
            "from_adjacency must never silently truncate an offset"
        );
    }

    /// Same style as the adjacency offset pin: wrapping `+= 1` / `as u32` in
    /// `from_csr_rect` must not return. Hostile unchecked CSRs with huge nnz
    /// would otherwise produce a valid-looking CSC with wrapped pointers.
    #[test]
    fn from_csr_rect_degrees_and_edge_ids_are_checked() {
        let source = include_str!("sparse.rs");
        let body = source
            .split_once("pub fn from_csr_rect")
            .expect("from_csr_rect source")
            .1
            .split_once("/// Number of columns")
            .expect("end of from_csr_rect source")
            .0;
        assert!(
            body.contains("checked_add(1)"),
            "from_csr_rect must checked_add column degrees"
        );
        assert!(
            body.contains("u32::try_from(e)"),
            "from_csr_rect must try_from edge indices instead of `as u32`"
        );
        assert!(
            !body.contains("degrees[col] += 1"),
            "from_csr_rect must never wrap column degrees"
        );
        assert!(
            !body.contains("edge_idx[slot] = e as u32"),
            "from_csr_rect must never silently truncate an edge index"
        );
    }

    #[test]
    fn stored_ncols_preserves_trailing_empty_columns() {
        let csr = Csr::from_parts(vec![0, 1], vec![0], 4).expect("valid");
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.col(), &[0]);
        assert_eq!(csr.nrows(), 1);
        // Accessors only — fields are private.
        let _ = csr.row_ptr();
        let _ = csr.col();
    }

    #[test]
    fn from_parts_rejects_column_out_of_range() {
        assert_eq!(
            Csr::from_parts(vec![0, 1], vec![4], 4),
            Err(CsrError::ColumnOutOfRange { col: 4, ncols: 4 })
        );
    }

    #[test]
    fn from_parts_rejects_unsorted_row_but_allows_duplicates() {
        assert_eq!(
            Csr::from_parts(vec![0, 2], vec![2, 0], 3),
            Err(CsrError::RowUnsorted { row: 0, edge: 1 })
        );
        let dup = Csr::from_parts(vec![0, 3], vec![1, 1, 2], 3).expect("duplicates ok");
        assert_eq!(dup.row_cols(0), &[1, 1, 2]);
    }

    #[test]
    fn from_adjacency_stable_sorts_and_sets_ncols() {
        let csr = Csr::from_adjacency(&[vec![2, 0, 0], vec![1]]);
        assert_eq!(csr.row_cols(0), &[0, 0, 2]);
        assert_eq!(csr.ncols(), 3); // max(2 rows, max_col 2 + 1)
        let tall = Csr::from_adjacency(&[vec![0], vec![], vec![], vec![]]);
        assert_eq!(tall.ncols(), 4); // max(4, 1)
    }

    #[test]
    fn csr_from_adjacency_neighbors() {
        let csr = Csr::from_adjacency(&[vec![1, 2], vec![0], vec![]]);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.row_cols(0), &[1, 2]);
        assert_eq!(csr.neighbors(1).collect::<Vec<_>>(), vec![0]);
        assert_eq!(csr.neighbors(2).count(), 0);
        assert_eq!(csr.ncols(), 3);
    }

    #[test]
    fn csr_from_parts_rejects_bad_shape() {
        assert_eq!(
            Csr::from_parts(vec![], vec![], 0),
            Err(CsrError::EmptyRowPtr)
        );
        assert_eq!(
            Csr::from_parts(vec![1, 1], vec![], 0),
            Err(CsrError::NonZeroStart { start: 1 })
        );
        assert_eq!(
            Csr::from_parts(vec![0, 2, 1], vec![0, 1], 2),
            Err(CsrError::NotMonotonic { index: 2 })
        );
        assert_eq!(
            Csr::from_parts(vec![0, 1], vec![0, 1], 2),
            Err(CsrError::NnzMismatch {
                row_ptr_end: 1,
                col_len: 2
            })
        );
    }

    #[test]
    fn csc_from_csr_rect_rejects_structurally_invalid_csr() {
        let cases = [
            (
                "empty row_ptr",
                Csr::from_parts_unchecked(vec![], vec![], 0),
                0,
                CsrError::EmptyRowPtr,
            ),
            (
                "non-zero row_ptr start",
                Csr::from_parts_unchecked(vec![1, 1], vec![], 0),
                0,
                CsrError::NonZeroStart { start: 1 },
            ),
            (
                "non-monotonic row_ptr",
                Csr::from_parts_unchecked(vec![0, 2, 1], vec![0], 3),
                3,
                CsrError::NotMonotonic { index: 2 },
            ),
            (
                "row_ptr/nnz mismatch",
                Csr::from_parts_unchecked(vec![0, 2], vec![0], 3),
                3,
                CsrError::NnzMismatch {
                    row_ptr_end: 2,
                    col_len: 1,
                },
            ),
            (
                "unsorted row",
                Csr::from_parts_unchecked(vec![0, 2], vec![2, 0], 3),
                3,
                CsrError::RowUnsorted { row: 0, edge: 1 },
            ),
        ];

        for (name, csr, ncols, expected) in cases {
            let result = std::panic::catch_unwind(|| Csc::from_csr_rect(&csr, ncols));
            let actual = result.unwrap_or_else(|_| {
                panic!("{name}: conversion panicked instead of returning {expected}")
            });
            assert_eq!(actual, Err(expected), "{name}");
        }
    }

    #[test]
    fn csr_edges_row_major() {
        // from_adjacency stable-sorts each row.
        let csr = Csr::from_adjacency(&[vec![2, 0], vec![1]]);
        let edges: Vec<_> = csr.edges().collect();
        assert_eq!(edges, vec![(0, 0), (0, 2), (1, 1)]);
    }

    #[test]
    fn csc_fan_in_matches_csr_edges() {
        // Edges: 0→2, 1→2, 3→2 (coincidence wiring).
        let csr = Csr::from_parts(vec![0, 1, 2, 2, 3], vec![2, 2, 2], 4).expect("csr");
        let csc = Csc::from_csr(&csr);
        assert_eq!(csc.ncols(), 4);
        assert_eq!(csc.nnz(), 3);
        assert!(csc.incoming(0).next().is_none());
        assert!(csc.incoming(1).next().is_none());
        let into_2: Vec<_> = csc.incoming(2).collect();
        assert_eq!(into_2, vec![(0, 0), (1, 1), (3, 2)]);
        assert!(csc.incoming(3).next().is_none());
    }

    #[test]
    fn csc_round_trip_edge_ids_cover_csr() {
        let csr = Csr::from_adjacency(&[vec![1, 2], vec![0, 2], vec![0]]);
        let csc = csr.to_csc();
        let mut seen = vec![false; csr.nnz()];
        for (pre, post) in csr.edges() {
            let hit = csc
                .incoming(post as usize)
                .any(|(p, e)| p == pre && csr.col()[e as usize] == post);
            assert!(hit, "missing reverse entry for ({pre},{post})");
        }
        for e in csc.edge_idx.iter().map(|&e| e as usize) {
            seen[e] = true;
        }
        assert!(seen.iter().all(|&v| v));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Neighbor iteration must recover each adjacency list after stable sort.
        #[test]
        fn csr_neighbor_iteration_matches_adjacency(
            rows in proptest::collection::vec(
                proptest::collection::vec(0u32..64, 0..8),
                0..16,
            )
        ) {
            let csr = Csr::from_adjacency(&rows);
            prop_assert_eq!(csr.nrows(), rows.len());
            prop_assert_eq!(
                csr.nnz(),
                rows.iter().map(Vec::len).sum::<usize>()
            );

            for (r, raw) in rows.iter().enumerate() {
                let mut expected = raw.clone();
                expected.sort();
                let via_iter: Vec<u32> = csr.neighbors(r).collect();
                prop_assert_eq!(&via_iter, &expected);
                prop_assert_eq!(csr.row_cols(r), expected.as_slice());

                let start = csr.row_ptr()[r] as usize;
                let end = csr.row_ptr()[r + 1] as usize;
                prop_assert_eq!(&csr.col()[start..end], expected.as_slice());
            }

            let flat_expected: Vec<u32> = rows
                .iter()
                .flat_map(|row| {
                    let mut s = row.clone();
                    s.sort();
                    s
                })
                .collect();
            let flat_got: Vec<u32> = csr.edges().map(|(_, c)| c).collect();
            prop_assert_eq!(flat_got, flat_expected);
        }

        /// `from_parts` must accept adjacency-built CSR and preserve neighbors.
        #[test]
        fn csr_from_parts_preserves_neighbors(
            rows in proptest::collection::vec(
                proptest::collection::vec(0u32..32, 0..6),
                0..12,
            )
        ) {
            let built = Csr::from_adjacency(&rows);
            let csr = Csr::from_parts(
                built.row_ptr().to_vec(),
                built.col().to_vec(),
                built.ncols(),
            )
            .expect("adjacency CSR must be valid");
            for (r, raw) in rows.iter().enumerate() {
                let mut expected = raw.clone();
                expected.sort();
                let got: Vec<u32> = csr.neighbors(r).collect();
                prop_assert_eq!(&got, &expected);
            }
        }
    }
}
