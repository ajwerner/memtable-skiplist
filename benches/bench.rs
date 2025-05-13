// Copyright (c) 2024-present, Andrew Werner
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rng, seq::SliceRandom, RngCore};

use crossbeam_skiplist::SkipMap as CrossbeamSkipMap;
use memtable_skiplist::SkipMap;

const COUNTS: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];

fn bench_skipmap_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    for &n in &COUNTS {
        group.bench_with_input(BenchmarkId::new("CrossbeamSkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            b.iter(|| {
                let map = CrossbeamSkipMap::new();
                for &k in &keys {
                    map.insert(k, k);
                }
            });
        });
        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            b.iter(|| {
                let map = SkipMap::new(rng().next_u32());
                for &k in &keys {
                    map.insert(k, k);
                }
            });
        });
    }
    group.finish();
}

fn bench_skipmap_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    for &n in &COUNTS {
        group.bench_with_input(BenchmarkId::new("CrossbeamSkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            let m = CrossbeamSkipMap::new();
            for k in keys {
                m.insert(k, k);
            }
            b.iter(|| {
                m.iter().for_each(|v| {
                    black_box(v);
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            let m = SkipMap::new(rng().next_u32());
            for k in keys {
                m.insert(k, k);
            }
            b.iter(|| {
                m.iter().for_each(|v| {
                    black_box(v);
                })
            });
        });
    }
    group.finish();
}

fn bench_skipmap_iter_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_range");
    const COUNT: u64 = 1_000_000;
    for n in [1, 10, 100, 10000] {
        group.bench_with_input(BenchmarkId::new("CrossbeamSkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..COUNT).collect();
            keys.shuffle(&mut rng());
            let m = CrossbeamSkipMap::new();
            for k in keys {
                m.insert(k, k);
            }
            let mut r = rng();
            b.iter(|| {
                let start = r.next_u64() % COUNT;
                m.range(start..start + n).for_each(|v| {
                    black_box(v);
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..1000000 as u64).collect();
            keys.shuffle(&mut rng());
            let m = SkipMap::new(rng().next_u32());
            for k in keys {
                m.insert(k, k);
            }
            let mut r = rng();
            b.iter(|| {
                let start = r.next_u64() % COUNT;
                m.range(start..start + n).for_each(|v| {
                    black_box(v);
                })
            });
        });
    }
    group.finish();
}

fn bench_skipmap_iter_range_rev(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_range_rev");
    const COUNT: u64 = 1_000_000;
    for n in [1, 10, 100, 10000] {
        group.bench_with_input(BenchmarkId::new("CrossbeamSkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..COUNT).collect();
            keys.shuffle(&mut rng());
            let m = CrossbeamSkipMap::new();
            for k in keys {
                m.insert(k, k);
            }
            let mut r = rng();
            b.iter(|| {
                let start = r.next_u64() % COUNT;
                m.range(start..start + n).rev().for_each(|v| {
                    black_box(v);
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..1000000 as u64).collect();
            keys.shuffle(&mut rng());
            let m = SkipMap::new(rng().next_u32());
            for k in keys {
                m.insert(k, k);
            }
            let mut r = rng();
            b.iter(|| {
                let start = r.next_u64() % COUNT;
                m.range(start..start + n).rev().for_each(|v| {
                    black_box(v);
                })
            });
        });
    }
    group.finish();
}

fn bench_skipmap_iter_rev(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_rev");
    for &n in &COUNTS {
        group.bench_with_input(BenchmarkId::new("CrossbeamSkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            let m = CrossbeamSkipMap::new();
            for k in keys {
                m.insert(k, k);
            }
            b.iter(|| {
                m.iter().rev().for_each(|v| {
                    black_box(v);
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("SkipMap", n), &n, |b, &n| {
            let mut keys: Vec<u64> = (0..n as u64).collect();
            keys.shuffle(&mut rng());
            let m = SkipMap::new(rng().next_u32());
            for k in keys {
                m.insert(k, k);
            }
            b.iter(|| {
                m.iter().rev().for_each(|v| {
                    black_box(v);
                })
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_skipmap_insert, bench_skipmap_iter,
              bench_skipmap_iter_rev, bench_skipmap_iter_range,
              bench_skipmap_iter_range_rev
}
criterion_main!(benches);
