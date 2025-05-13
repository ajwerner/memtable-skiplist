// Copyright (c) 2024-present, Andrew Werner
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

#![allow(unsafe_code)]

use std::{
    alloc::Layout,
    borrow::Borrow,
    marker::PhantomData,
    mem::offset_of,
    ops::{Bound, RangeBounds, RangeFull},
    ptr::NonNull,
    sync::{
        atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering},
        LazyLock,
    },
};

use crate::arena::Arenas;

const MAX_HEIGHT: usize = 20;

static PROBABILITIES: LazyLock<[u32; MAX_HEIGHT - 1]> = LazyLock::new(|| {
    let mut probabilities = [0u32; MAX_HEIGHT - 1];
    const P_VALUE: f64 = 1f64 / std::f64::consts::E;
    let mut p = 1f64;
    for i in 0..MAX_HEIGHT {
        if i > 0 {
            probabilities[i - 1] = ((u32::MAX as f64) * p) as u32;
        }
        p *= P_VALUE;
    }
    probabilities
});

#[repr(C)]
pub(crate) struct Links<K, V> {
    next: AtomicPtr<Node<K, V>>,
    prev: AtomicPtr<Node<K, V>>,
}

#[repr(C)]
pub(crate) struct Node<K, V> {
    data: (K, V),
    // Note that this is a lie! Sometimes this array is shorter than MAX_HEIGHT.
    // and will instead point to garbage. That's okay because we'll use other
    // bookkeeping invariants to ensure that we never actually access the garbage.
    pub(crate) tower: [Links<K, V>; MAX_HEIGHT],
}

/// A SkipMap is a concurrent mapping structure like a BTreeMao
/// but it allows for concurrent reads and writes. A tradeoff
/// is that it does not allow for updates or deletions.
pub struct SkipMap<K, V> {
    // We need a vec of buffers.
    arena: ArenasAllocator<K, V>,
    // Accessed with relaxed ordering because they never change.
    head: AtomicPtr<Node<K, V>>,
    // Accessed with relaxed ordering because they never change.
    tail: AtomicPtr<Node<K, V>>,

    hot: crossbeam_utils::CachePadded<HotData>,
}

impl<K, V> Default for SkipMap<K, V> {
    fn default() -> Self {
        Self::new(1)
    }
}

unsafe impl<K, V> Send for SkipMap<K, V> {}
unsafe impl<K, V> Sync for SkipMap<K, V> {}

#[derive(Default)]
struct HotData {
    seed: AtomicU32,
    height: AtomicUsize,
    len: AtomicUsize,
}
impl<K, V> SkipMap<K, V> {
    /// New constructs a new `[SkipMap]`.
    pub fn new(seed: u32) -> Self {
        let arena = ArenasAllocator::default();
        let head = arena.alloc(MAX_HEIGHT);
        let tail = arena.alloc(MAX_HEIGHT);
        for i in 0..MAX_HEIGHT {
            unsafe { &(*head).tower[i].next }.store(tail, Ordering::Relaxed);
            unsafe { &(*tail).tower[i].prev }.store(head, Ordering::Relaxed);
        }
        Self {
            arena,
            head: AtomicPtr::new(head),
            tail: AtomicPtr::new(tail),
            hot: crossbeam_utils::CachePadded::new(HotData {
                seed: AtomicU32::new(seed),
                height: AtomicUsize::new(1),
                len: AtomicUsize::new(0),
            }),
        }
    }
}

impl<K, V> SkipMap<K, V>
where
    K: Ord,
{
    /// Iter constructs an iterator over the complete
    /// range.
    pub fn iter(&self) -> Iter<'_, K, V, K, RangeFull> {
        Iter::new(self, RangeFull)
    }

    fn head(&self) -> *mut Node<K, V> {
        // Relaxed is fine here because this never changes.
        self.head.load(Ordering::Relaxed)
    }

    fn tail(&self) -> *mut Node<K, V> {
        // Relaxed is fine here because this never changes.
        self.tail.load(Ordering::Relaxed)
    }

    fn find_from_node<Q>(&self, bounds: Bound<&Q>) -> NonNull<Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = match bounds {
            std::ops::Bound::Included(v) => match self.seek_for_base_splice(v) {
                SpliceOrMatch::Splice(splice) => splice.prev,
                SpliceOrMatch::Match(node) => {
                    unsafe { &(*node).tower[0] }.prev.load(Ordering::Acquire)
                }
            },
            std::ops::Bound::Excluded(v) => match self.seek_for_base_splice(v) {
                SpliceOrMatch::Splice(splice) => splice.prev,
                SpliceOrMatch::Match(node) => node,
            },
            std::ops::Bound::Unbounded => self.head(),
        };
        unsafe { NonNull::new_unchecked(node) }
    }

    fn find_to_node<Q>(&self, bounds: Bound<&Q>) -> NonNull<Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = match bounds {
            std::ops::Bound::Included(v) => match self.seek_for_base_splice(v) {
                SpliceOrMatch::Splice(splice) => splice.next,
                SpliceOrMatch::Match(node) => {
                    unsafe { &(*node).tower[0] }.next.load(Ordering::Acquire)
                }
            },
            std::ops::Bound::Excluded(v) => match self.seek_for_base_splice(v) {
                SpliceOrMatch::Splice(splice) => splice.next,
                SpliceOrMatch::Match(node) => node,
            },
            std::ops::Bound::Unbounded => self.tail(),
        };
        unsafe { NonNull::new_unchecked(node) }
    }

    /// Range constructs an iterator over a range of the
    /// SkipMap.
    pub fn range<Q, R>(&self, range: R) -> Iter<'_, K, V, Q, R>
    where
        K: Borrow<Q>,
        R: RangeBounds<Q>,
        Q: Ord + ?Sized,
    {
        Iter {
            map: self,
            range,
            exhaused: false,
            next: None,
            next_back: None,
            _phantom: Default::default(),
        }
    }

    /// The SkipMap is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The current number of entries in the SkipMap.
    pub fn len(&self) -> usize {
        self.hot.len.load(Ordering::Relaxed)
    }

    /// The current height of the SkipMap.
    pub fn height(&self) -> usize {
        self.hot.height.load(Ordering::Relaxed)
    }

    fn new_node(&self, key: K, value: V) -> (*mut Node<K, V>, usize) {
        let height = self.random_height();
        let node = self.arena.alloc(height);
        unsafe { (*node).data = (key, value) };
        (node, height)
    }

    fn random_height(&self) -> usize {
        // Pseudorandom number generation from "Xorshift RNGs" by George Marsaglia.
        //
        // This particular set of operations generates 32-bit integers. See:
        // https://en.wikipedia.org/wiki/Xorshift#Example_implementation
        let mut num = self.hot.seed.load(Ordering::Relaxed);
        num ^= num << 13;
        num ^= num >> 17;
        num ^= num << 5;
        self.hot.seed.store(num, Ordering::Relaxed);
        let val = num as u32;

        let mut height = 1;
        for &p in PROBABILITIES.iter() {
            if val > p {
                break;
            }
            height += 1;
        }
        // Keep decreasing the height while it's much larger than all towers currently in the
        // skip list.
        let head = self.head();
        let tail = self.tail();
        let head_tower = &unsafe { &*head }.tower;
        while height >= 4 && head_tower[height - 2].next.load(Ordering::Acquire) == tail {
            height -= 1;
        }

        // Track the max height to speed up lookups
        let mut max_height = self.hot.height.load(Ordering::Relaxed);
        while height > max_height {
            match self.hot.height.compare_exchange(
                max_height,
                height,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(h) => max_height = h,
            }
        }
        height
    }

    fn find_splice_for_level<Q>(
        &self,
        key: &Q,
        level: usize,
        start: *mut Node<K, V>,
    ) -> SpliceOrMatch<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut prev = start;
        let mut next = unsafe { (&*prev).tower[level].next.load(Ordering::Acquire) };
        loop {
            // Assume prev.key < key.
            let after_next = unsafe { (&*next).tower[level].next.load(Ordering::Acquire) };
            if after_next.is_null() {
                // we've reached the end of the list.
                return Splice { prev, next }.into();
            };
            let next_key = unsafe { &(*next).data.0 };
            match key.cmp(next_key.borrow()) {
                std::cmp::Ordering::Less => return Splice { next, prev }.into(),
                std::cmp::Ordering::Equal => return SpliceOrMatch::Match(next),
                std::cmp::Ordering::Greater => {
                    prev = next;
                    next = after_next;
                }
            }
        }
    }

    // Returns true if k was found in the map.
    fn seek_splices(&self, key: &K) -> Option<Splices<K, V>> {
        let mut splices = Splices::default();
        let mut level = self.hot.height.load(Ordering::Relaxed) - 1;
        let mut prev = self.head.load(Ordering::Relaxed);
        loop {
            match self.find_splice_for_level(key.borrow(), level, prev) {
                SpliceOrMatch::Splice(splice) => {
                    prev = splice.prev;
                    splices[level] = Some(splice)
                }
                SpliceOrMatch::Match(_match) => break None,
            }
            if level == 0 {
                break Some(splices);
            }
            level -= 1;
        }
    }

    fn seek_for_base_splice<Q>(&self, key: &Q) -> SpliceOrMatch<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut level = self.height() - 1;
        let mut prev = self.head();
        loop {
            match self.find_splice_for_level(key, level, prev) {
                n @ SpliceOrMatch::Match(_) => return n,
                s @ SpliceOrMatch::Splice(_) if level == 0 => return s,
                SpliceOrMatch::Splice(s) => {
                    prev = s.prev;
                    level -= 1;
                }
            }
        }
    }

    /// Insert a key-value pair into the SkipMap. Returns true
    /// if the entry was inserted.
    pub fn insert(&self, k: K, v: V) -> bool {
        let Some(splices) = self.seek_splices(&k) else {
            return false;
        };
        let (node, height) = self.new_node(k, v);
        for level in 0..height {
            let mut splice = match splices[level].clone() {
                Some(splice) => splice,
                // This node increased the height.
                None => Splice {
                    prev: self.head.load(Ordering::Relaxed),
                    next: self.tail.load(Ordering::Relaxed),
                },
            };

            loop {
                let Splice { next, prev } = splice;
                // +----------------+     +------------+     +----------------+
                // |      prev      |     |     nd     |     |      next      |
                // | prevNextOffset |---->|            |     |                |
                // |                |<----| prevOffset |     |                |
                // |                |     | nextOffset |---->|                |
                // |                |     |            |<----| nextPrevOffset |
                // +----------------+     +------------+     +----------------+
                //
                // 1. Initialize prevOffset and nextOffset to point to prev and next.
                // 2. CAS prevNextOffset to repoint from next to nd.
                // 3. CAS nextPrevOffset to repoint from prev to nd.
                unsafe { &(*node).tower[level].prev }.store(prev, Ordering::Release);
                unsafe { &(*node).tower[level].next }.store(next, Ordering::Release);

                // Check whether next has an updated link to prev. If it does not,
                // that can mean one of two things:
                //   1. The thread that added the next node hasn't yet had a chance
                //      to add the prev link (but will shortly).
                //   2. Another thread has added a new node between prev and next.
                let next_prev = unsafe { &(*next).tower[level].prev }.load(Ordering::Acquire);
                if next_prev != prev {
                    // Determine whether #1 or #2 is true by checking whether prev
                    // is still pointing to next. As long as the atomic operations
                    // have at least acquire/release semantics (no need for
                    // sequential consistency), this works, as it is equivalent to
                    // the "publication safety" pattern.
                    let prev_next = unsafe { &(*prev).tower[level].next }.load(Ordering::Acquire);
                    if prev_next == next {
                        let _ = unsafe { &(*next).tower[level].prev }.compare_exchange(
                            next_prev,
                            prev,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                    }
                }

                if unsafe { &(*prev).tower[level].next }
                    .compare_exchange(next, node, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // We inserted ourselves after prev, now put the back pointer.
                    let _ = unsafe { &(*next).tower[level].prev }.compare_exchange(
                        prev,
                        node,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    break;
                };
                splice = match self.find_splice_for_level(unsafe { &(*node).data.0 }, level, prev) {
                    SpliceOrMatch::Splice(splice) => splice,
                    SpliceOrMatch::Match(_non_null) => return false,
                }
            }
        }
        self.hot.len.fetch_add(1, Ordering::Relaxed);
        true
    }
}

enum SpliceOrMatch<K, V> {
    Splice(Splice<K, V>),
    Match(*mut Node<K, V>),
}

impl<K, V> From<Splice<K, V>> for SpliceOrMatch<K, V> {
    fn from(value: Splice<K, V>) -> Self {
        SpliceOrMatch::Splice(value)
    }
}

type Splices<K, V> = [Option<Splice<K, V>>; MAX_HEIGHT];

struct Splice<K, V> {
    prev: *mut Node<K, V>,
    next: *mut Node<K, V>,
}

impl<K, V> Clone for Splice<K, V> {
    fn clone(&self) -> Self {
        let &Self { prev, next } = self;
        Self { prev, next }
    }
}

pub struct Iter<'m, K, V, Q: ?Sized, R> {
    map: &'m SkipMap<K, V>,
    range: R,
    exhaused: bool,
    next: Option<NonNull<Node<K, V>>>,
    next_back: Option<NonNull<Node<K, V>>>,
    _phantom: PhantomData<fn(Q)>,
}

impl<'m, K, V, Q, R> Iter<'m, K, V, Q, R>
where
    K: Borrow<Q> + Ord,
    R: RangeBounds<Q>,
    Q: Ord + ?Sized,
{
    fn new(map: &'m SkipMap<K, V>, range: R) -> Self {
        Self {
            map: map,
            range,
            exhaused: false,
            next: None,
            next_back: None,
            _phantom: Default::default(),
        }
    }
}

pub struct Entry<'m, K, V>(&'m Node<K, V>);

impl<'m, K, V> Entry<'m, K, V> {
    pub fn key(&self) -> &'m K {
        &self.0.data.0
    }

    pub fn value(&self) -> &'m V {
        &self.0.data.1
    }
}

impl<'m, K, V, Q: ?Sized, R> Iter<'m, K, V, Q, R> {
    fn exhaust(&mut self) {
        self.exhaused = true;
        self.next = None;
        self.next_back = None;
    }
}

impl<'m, K, V, Q, R> Iterator for Iter<'m, K, V, Q, R>
where
    K: Borrow<Q> + Ord,
    R: RangeBounds<Q>,
    Q: Ord + ?Sized,
{
    type Item = Entry<'m, K, V>;

    #[allow(unsafe_code)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhaused {
            return None;
        }
        let next = match self.next {
            Some(next) => next,
            None => {
                let before = self.map.find_from_node(self.range.start_bound());
                let next = unsafe { before.as_ref() }.tower[0]
                    .next
                    .load(Ordering::Acquire);
                match NonNull::new(next) {
                    Some(next) => next,
                    None => {
                        self.exhaused = true;
                        self.next_back = None;
                        return None;
                    }
                }
            }
        };
        if match self.range.end_bound() {
            Bound::Included(bound) => unsafe { next.as_ref() }.data.0.borrow() > bound,
            Bound::Excluded(bound) => unsafe { next.as_ref() }.data.0.borrow() >= bound,
            Bound::Unbounded => false,
        } {
            self.exhaust();
            return None;
        }
        let after_next = unsafe { next.as_ref() }.tower[0]
            .next
            .load(Ordering::Acquire);
        let Some(after_next) = NonNull::new(after_next) else {
            self.exhaust();
            return None;
        };
        if self.next_back.is_none_or(|next_back| next_back != next) {
            self.next = Some(after_next);
        } else {
            self.exhaust();
        };
        Some(Entry(unsafe { next.as_ref() }))
    }
}

impl<'m, K, V, Q, R> DoubleEndedIterator for Iter<'m, K, V, Q, R>
where
    K: Borrow<Q> + Ord,
    R: RangeBounds<Q>,
    Q: Ord + ?Sized,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.exhaused {
            return None;
        }
        let next_back = if let Some(next_back) = self.next_back {
            next_back
        } else {
            let before = self.map.find_to_node(self.range.end_bound());
            let next_back = unsafe { before.as_ref() }.tower[0]
                .prev
                .load(Ordering::Acquire);
            if let Some(next_back) = NonNull::new(next_back) {
                next_back
            } else {
                self.exhaust();
                return None;
            }
        };
        if match self.range.start_bound() {
            Bound::Included(bound) => unsafe { next_back.as_ref() }.data.0.borrow() < bound,
            Bound::Excluded(bound) => unsafe { next_back.as_ref() }.data.0.borrow() <= bound,
            Bound::Unbounded => false,
        } {
            self.exhaust();
            return None;
        }
        let before_next_back = unsafe { next_back.as_ref() }.tower[0]
            .prev
            .load(Ordering::Acquire);
        let Some(before_next_back) = NonNull::new(before_next_back) else {
            self.exhaust();
            return None;
        };
        if self.next.is_none_or(|next| next_back != next) {
            self.next_back = Some(before_next_back);
        } else {
            self.exhaust();
        };
        Some(Entry(unsafe { next_back.as_ref() }))
    }
}

#[cfg(test)]
impl<K, V> SkipMap<K, V>
where
    K: Ord,
{
    pub(crate) fn check_integrity(&mut self) {
        use std::collections::HashSet;
        // We want to check that there are no cycles, that the forward and backwards
        // directions have the same chains at all levels, and that the values are
        // ordered.
        let head_nodes = {
            let mut head = self.head();
            let mut head_forward_nodes = HashSet::new();
            let mut head_nodes = Vec::new();
            while !head.is_null() {
                head_nodes.push(head);
                assert!(head_forward_nodes.insert(head), "head");
                head = unsafe { &(*head).tower[0].next }.load(Ordering::Acquire);
            }
            head_nodes
        };

        let mut tail_nodes = {
            let mut tail = self.tail();
            let mut tail_backward_nodes = HashSet::new();
            let mut tail_nodes = Vec::new();
            while !tail.is_null() {
                tail_nodes.push(tail);
                assert!(tail_backward_nodes.insert(tail), "tail");
                tail = unsafe { &(*tail).tower[0].prev }.load(Ordering::Acquire);
            }
            tail_nodes
        };
        tail_nodes.reverse();
        assert_eq!(head_nodes, tail_nodes);
    }
}

struct ArenasAllocator<K, V> {
    arenas: Arenas,
    _phantom: PhantomData<fn(K, V)>,
}

impl<K, V> Default for ArenasAllocator<K, V> {
    fn default() -> Self {
        Self {
            arenas: Default::default(),
            _phantom: Default::default(),
        }
    }
}

impl<K, V> ArenasAllocator<K, V> {
    const ALIGNMENT: usize = align_of::<Node<K, V>>();
    const TOWER_OFFSET: usize = offset_of!(Node<K, V>, tower);

    fn alloc(&self, height: usize) -> *mut Node<K, V> {
        let layout = unsafe {
            Layout::from_size_align_unchecked(
                Self::TOWER_OFFSET + (height * size_of::<Links<K, V>>()),
                Self::ALIGNMENT,
            )
        };
        self.arenas.alloc(layout) as *mut Node<K, V>
    }
}
