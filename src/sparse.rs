//! Sparse CSR connectivity storage (U01).

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
            Self::NnzMismatch {
                row_ptr_end,
                col_len,
            } => write!(
                f,
                "CSR row_ptr end ({row_ptr_end}) != col.len() ({col_len})"
            ),
        }
    }
}

impl std::error::Error for CsrError {}

/// Compressed-sparse-row connectivity graph.
///
/// `row_ptr` has length `nrows + 1`. Neighbors of row `r` are the column
/// indices `col[row_ptr[r] as usize .. row_ptr[r + 1] as usize]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    pub row_ptr: Vec<u32>,
    pub col: Vec<u32>,
}

impl Csr {
    /// Build from explicit arrays after validating CSR invariants.
    pub fn from_parts(row_ptr: Vec<u32>, col: Vec<u32>) -> Result<Self, CsrError> {
        validate(&row_ptr, &col)?;
        Ok(Self { row_ptr, col })
    }

    /// Build from explicit arrays without validation (caller guarantees shape).
    #[inline]
    pub fn from_parts_unchecked(row_ptr: Vec<u32>, col: Vec<u32>) -> Self {
        Self { row_ptr, col }
    }

    /// Build from per-row adjacency lists.
    pub fn from_adjacency(rows: &[Vec<u32>]) -> Self {
        let nrows = rows.len();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let nnz: usize = rows.iter().map(Vec::len).sum();
        let mut col = Vec::with_capacity(nnz);
        row_ptr.push(0);
        for row in rows {
            col.extend_from_slice(row);
            row_ptr.push(col.len() as u32);
        }
        Self { row_ptr, col }
    }

    /// Empty graph with `nrows` rows and no edges.
    pub fn empty(nrows: usize) -> Self {
        Self {
            row_ptr: vec![0; nrows + 1],
            col: Vec::new(),
        }
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

    /// Inferred number of columns (max column index + 1, or 0 if empty).
    ///
    /// # Cost
    ///
    /// **This is O(nnz), not O(1).** It scans every stored column index; the
    /// `#[inline]` below refers to call overhead and does not make it cheap.
    /// Unlike [`Csc::ncols`], which is a genuine O(1) `col_ptr.len() - 1`, this
    /// value is not cached because `Csr` does not store a column count.
    ///
    /// Never call it in a loop condition. `for c in 0..csr.ncols()` is O(nnz²).
    /// Hoist it into a `let` before the loop.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.col
            .iter()
            .copied()
            .max()
            .map_or(0, |m| (m + 1) as usize)
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

    /// Build a CSC reverse index over this CSR (square cell graph: `ncols = nrows`).
    #[inline]
    pub fn to_csc(&self) -> Csc {
        Csc::from_csr(self)
    }
}

/// Compressed-sparse-column reverse index over CSR edge storage.
///
/// Columns are postsynaptic cells. Each CSC entry stores the presynaptic row and
/// the CSR edge index so synapse / weight tables stay CSR-ordered while
/// postsynaptic fan-in is `O(degree_in)` instead of `O(nnz)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csc {
    /// Column pointers; length `ncols + 1`.
    pub col_ptr: Vec<u32>,
    /// Presynaptic (CSR row) index for each CSC entry.
    pub row: Vec<u32>,
    /// CSR edge index (synapse / weight table index) for each CSC entry.
    pub edge_idx: Vec<u32>,
}

impl Csc {
    /// Empty reverse index with `ncols` columns and no edges.
    pub fn empty(ncols: usize) -> Self {
        Self {
            col_ptr: vec![0; ncols + 1],
            row: Vec::new(),
            edge_idx: Vec::new(),
        }
    }

    /// Build CSC fan-in from a CSR graph.
    ///
    /// Uses `ncols = csr.nrows()` (directed cell graph). Column indices in
    /// `csr.col` must be `< ncols`.
    pub fn from_csr(csr: &Csr) -> Self {
        let ncols = csr.nrows();
        if ncols == 0 {
            return Self::empty(0);
        }
        let nnz = csr.nnz();
        let mut degrees = vec![0u32; ncols];
        for &c in &csr.col {
            let col = c as usize;
            assert!(
                col < ncols,
                "CSC build: column {c} out of range (ncols={ncols})"
            );
            degrees[col] += 1;
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        let mut acc = 0u32;
        for &d in &degrees {
            acc += d;
            col_ptr.push(acc);
        }

        let mut row = vec![0u32; nnz];
        let mut edge_idx = vec![0u32; nnz];
        let mut next = col_ptr[..ncols].to_vec();
        for r in 0..csr.nrows() {
            let start = csr.row_ptr[r] as usize;
            let end = csr.row_ptr[r + 1] as usize;
            for e in start..end {
                let c = csr.col[e] as usize;
                let slot = next[c] as usize;
                row[slot] = r as u32;
                edge_idx[slot] = e as u32;
                next[c] += 1;
            }
        }

        Self {
            col_ptr,
            row,
            edge_idx,
        }
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

fn validate(row_ptr: &[u32], col: &[u32]) -> Result<(), CsrError> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Csc, Csr, CsrError};
    use proptest::prelude::*;

    #[test]
    fn csr_from_adjacency_neighbors() {
        let csr = Csr::from_adjacency(&[vec![1, 2], vec![0], vec![]]);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.row_cols(0), &[1, 2]);
        assert_eq!(csr.neighbors(1).collect::<Vec<_>>(), vec![0]);
        assert_eq!(csr.neighbors(2).count(), 0);
    }

    #[test]
    fn csr_from_parts_rejects_bad_shape() {
        assert_eq!(Csr::from_parts(vec![], vec![]), Err(CsrError::EmptyRowPtr));
        assert_eq!(
            Csr::from_parts(vec![1, 1], vec![]),
            Err(CsrError::NonZeroStart { start: 1 })
        );
        assert_eq!(
            Csr::from_parts(vec![0, 2, 1], vec![0, 1]),
            Err(CsrError::NotMonotonic { index: 2 })
        );
        assert_eq!(
            Csr::from_parts(vec![0, 1], vec![0, 1]),
            Err(CsrError::NnzMismatch {
                row_ptr_end: 1,
                col_len: 2
            })
        );
    }

    #[test]
    fn csr_edges_row_major() {
        let csr = Csr::from_adjacency(&[vec![2, 0], vec![1]]);
        let edges: Vec<_> = csr.edges().collect();
        assert_eq!(edges, vec![(0, 2), (0, 0), (1, 1)]);
    }

    #[test]
    fn csc_fan_in_matches_csr_edges() {
        // Edges: 0→2, 1→2, 3→2 (coincidence wiring).
        let csr = Csr::from_parts(vec![0, 1, 2, 2, 3], vec![2, 2, 2]).expect("csr");
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
                .any(|(p, e)| p == pre && csr.col[e as usize] == post);
            assert!(hit, "missing reverse entry for ({pre},{post})");
        }
        for e in csc.edge_idx.iter().map(|&e| e as usize) {
            seen[e] = true;
        }
        assert!(seen.iter().all(|&v| v));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Neighbor iteration must recover each adjacency list exactly.
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

            for (r, expected) in rows.iter().enumerate() {
                let via_iter: Vec<u32> = csr.neighbors(r).collect();
                prop_assert_eq!(&via_iter, expected);
                prop_assert_eq!(csr.row_cols(r), expected.as_slice());

                let start = csr.row_ptr[r] as usize;
                let end = csr.row_ptr[r + 1] as usize;
                prop_assert_eq!(&csr.col[start..end], expected.as_slice());
            }

            let flat_expected: Vec<u32> = rows.iter().flatten().copied().collect();
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
            let csr = Csr::from_parts(built.row_ptr.clone(), built.col.clone())
                .expect("adjacency CSR must be valid");
            for (r, expected) in rows.iter().enumerate() {
                let got: Vec<u32> = csr.neighbors(r).collect();
                prop_assert_eq!(&got, expected);
            }
        }
    }
}
