// Copyright (c) 2024-present, Andrew Werner
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use std::{
    collections::BTreeMap,
    num::NonZero,
    ops::RangeBounds,
    sync::Barrier,
};

use super::*;
use quickcheck::{Arbitrary, Gen};
use rand::{rng, RngCore};

#[test]
fn test_basic() {
    let v = SkipMap::<usize, usize>::new(rng().next_u32());
    assert_eq!(v.insert(1, 1), true);
    assert_eq!(v.len(), 1);
    assert_eq!(v.insert(1, 2), false);
    assert_eq!(v.len(), 1);
    assert_eq!(v.insert(2, 2), true);
    assert_eq!(v.len(), 2);
    assert_eq!(v.insert(2, 1), false);
    let got: Vec<_> = v.iter().map(|e| (*e.key(), *e.value())).collect();
    assert_eq!(got, vec![(1, 1), (2, 2)]);
    let got_rev: Vec<_> = v.iter().rev().map(|e| (*e.key(), *e.value())).collect();
    assert_eq!(got_rev, vec![(2, 2), (1, 1)]);
}

#[derive(Clone, Debug)]
struct TestOperation {
    key: i32,
    value: i32,
}

impl Arbitrary for TestOperation {
    fn arbitrary(g: &mut Gen) -> Self {
        TestOperation {
            key: i32::arbitrary(g),
            value: i32::arbitrary(g),
        }
    }
}

#[derive(Debug, Clone)]
struct TestOperations {
    seed: u32,
    threads: usize,
    ops: Vec<TestOperation>,
}

impl Arbitrary for TestOperations {
    fn arbitrary(g: &mut Gen) -> Self {
        let max_threads = std::thread::available_parallelism()
            .map(NonZero::get)
            .unwrap_or(64)
            * 16;
        Self {
            seed: u32::arbitrary(g),
            threads: usize::arbitrary(g) % max_threads,
            ops: <Vec<TestOperation> as Arbitrary>::arbitrary(g),
        }
    }
}

#[test]
fn test_quickcheck() {
    fn prop(operations: TestOperations) -> bool {
        let mut skipmap = SkipMap::new(operations.seed);
        let barrier = Barrier::new(operations.threads);
        let outcomes = std::thread::scope(|scope| {
            let (mut ops, mut threads_to_launch) = (operations.ops.as_slice(), operations.threads);
            let mut thread_outcomes = Vec::new();
            while threads_to_launch > 0 {
                let items = ops.len() / threads_to_launch;
                let (subslice, remaining) = ops.split_at(items);
                ops = remaining;
                threads_to_launch -= 1;
                let skipmap = &skipmap;
                let barrier = &barrier;
                let spawned = scope.spawn(move || {
                    barrier.wait();
                    let mut outcomes = Vec::new();
                    for op in subslice {
                        outcomes.push(skipmap.insert(op.key, op.value));
                    }
                    outcomes
                });
                thread_outcomes.push(spawned);
            }
            thread_outcomes
                .into_iter()
                .flat_map(|v| v.join().unwrap())
                .collect::<Vec<_>>()
        });
        let successful_ops = operations
            .ops
            .into_iter()
            .zip(outcomes.into_iter())
            .filter_map(|(op, outcome)| outcome.then_some(op))
            .collect::<Vec<_>>();
        skipmap.check_integrity();

        verify_ranges(&skipmap, &successful_ops);

        let skipmap_items: Vec<_> = skipmap.iter().map(|e| (*e.key(), *e.value())).collect();
        let skipmap_items_rev: Vec<_> = skipmap
            .iter()
            .rev()
            .map(|e| (*e.key(), *e.value()))
            .collect();
        let mut skipmap_items_rev_rev = skipmap_items_rev.clone();
        skipmap_items_rev_rev.reverse();
        assert_eq!(successful_ops.len(), skipmap.len(), "len");
        assert_eq!(skipmap_items.len(), skipmap.len(), "items");
        assert_eq!(skipmap_items.len(), skipmap_items_rev.len(), "rev items");
        assert_eq!(
            skipmap_items, skipmap_items_rev_rev,
            "Forward iteration should match\n{:#?}\n{:#?}",
            skipmap_items, skipmap_items_rev_rev
        );

        true
    }

    quickcheck::quickcheck(prop as fn(TestOperations) -> bool);
}

fn verify_ranges(skipmap: &SkipMap<i32, i32>, successful_ops: &Vec<TestOperation>) {
    let mut successful_keys_sorted = successful_ops.iter().map(|op| op.key).collect::<Vec<_>>();
    successful_keys_sorted.sort();
    let btree = successful_ops
        .iter()
        .map(|&TestOperation { key, value }| (key, value))
        .collect::<BTreeMap<_, _>>();

    for _ in 0..10 {
        if successful_ops.len() == 0 {
            break;
        }
        let (a, b) = (
            rng().next_u32() as usize % successful_ops.len(),
            rng().next_u32() as usize % successful_ops.len(),
        );
        let (start, end) = (a.min(b), a.max(b));
        fn assert_range_eq<B: RangeBounds<i32> + Clone + std::fmt::Debug>(
            a: &BTreeMap<i32, i32>,
            b: &SkipMap<i32, i32>,
            bounds: B,
        ) {
            {
                let ra = a
                    .range(bounds.clone())
                    .map(|(&a, &b)| (a, b))
                    .collect::<Vec<_>>();
                let rb = b
                    .range(bounds.clone())
                    .map(|entry| (*entry.key(), *entry.value()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    ra,
                    rb,
                    "{} {:?} forward: {:#?} != {:#?}",
                    std::any::type_name::<B>(),
                    bounds,
                    ra,
                    rb
                );
            }
            {
                let ra = a
                    .range(bounds.clone())
                    .rev()
                    .map(|(&a, &b)| (a, b))
                    .collect::<Vec<_>>();
                let rb = b
                    .range(bounds.clone())
                    .rev()
                    .map(|entry| (*entry.key(), *entry.value()))
                    .collect::<Vec<_>>();

                assert_eq!(
                    ra,
                    rb,
                    "{} {:?} backwards: {:#?} != {:#?}",
                    std::any::type_name::<B>(),
                    bounds,
                    ra,
                    rb
                );
            }
        }
        let (start, end) = (successful_keys_sorted[start], successful_keys_sorted[end]);
        assert_range_eq(&btree, skipmap, ..);
        assert_range_eq(&btree, skipmap, ..end);
        assert_range_eq(&btree, skipmap, ..=end);
        assert_range_eq(&btree, skipmap, start..);
        assert_range_eq(&btree, skipmap, start..end);
        assert_range_eq(&btree, skipmap, start..=end);
    }
}
