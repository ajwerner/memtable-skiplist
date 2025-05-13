// Copyright (c) 2024-present, Andrew Werner
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

//! This crate is a purpose-built concurrent skiplist intended for use
//! in the fjall-rs/lsm-tree crate.
//!
//! Due to the specific requirements of that crate, this data structure
//! is notable in the features it lacks:
//!     * Updates
//!     * Deletes
//!     * Overwrites
//!
//! It supports two primary operations:
//!     * Insert
//!     * Double-Ended Iteration over a bounded range
//!
//! The data structure supports concurrent reads and writes.
//! Also, it uses arena-based memory allocation.
//!
//!

#![deny(clippy::all, missing_docs, clippy::cargo)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::indexing_slicing)]
#![warn(clippy::pedantic, clippy::nursery)]
#![warn(clippy::expect_used)]
#![allow(clippy::missing_const_for_fn)]
#![warn(clippy::multiple_crate_versions)]
#![allow(clippy::option_if_let_else)]
#![warn(clippy::needless_lifetimes)]

mod arena;
mod skipmap;

pub use skipmap::SkipMap;

#[cfg(test)]
mod test;
