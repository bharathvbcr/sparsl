//! Make the Metal kernel source an explicit build input.
//!
//! `src/backend/metal.rs` embeds `kernels/spmv.metal` with `include_str!`, and
//! Rust does track that — but a mutation run in this workspace edited the
//! `.metal` file, rebuilt nothing, and reported that the mutant survived. The
//! kernel under test was the unmutated one.
//!
//! That failure mode is worse than a slow build: it makes a check that never
//! ran look identical to one that ran and passed, which is the exact confusion
//! this crate is otherwise built to prevent. An explicit `rerun-if-changed`
//! costs nothing and removes the doubt.
fn main() {
    println!("cargo:rerun-if-changed=src/kernels/spmv.metal");
    println!("cargo:rerun-if-changed=src/kernels");
}
