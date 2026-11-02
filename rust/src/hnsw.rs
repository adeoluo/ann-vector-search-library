//! HNSW (Malkov and Yashunin, 2016), mirroring `ann_vector_search_library.hnsw.HNSW`.
//!
//! `Hnsw<f64>` reproduces the Python index exactly. `Hnsw<f32>` runs the same algorithm
//! on 32-bit floats with lane-parallel distances (see `distance.rs`): faster and half
//! the memory, but its sums round differently, so it is judged on recall, not parity.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use rayon::prelude::*;

use crate::distance::{Element, Metric};

/// `(distance, id)` ordered like a Python tuple: by distance, then by id.
#[derive(Clone, Copy, Debug)]
struct Key(f64, i64);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

/// A heap entry: ordered by `key` only, carrying the node's slot so a pop needs no id lookup.
#[derive(Clone, Copy)]
struct Item {
    key: Key,
    slot: usize,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// A min-heap with Python `heapq` semantics over `Key`.
#[derive(Default)]
struct MinHeap(BinaryHeap<std::cmp::Reverse<Item>>);

impl MinHeap {
    fn push(&mut self, key: Key, slot: usize) {
        self.0.push(std::cmp::Reverse(Item { key, slot }));
    }
    fn pop(&mut self) -> Option<Item> {
        self.0.pop().map(|r| r.0)
    }
    fn peek(&self) -> Option<Item> {
        self.0.peek().map(|r| r.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Visited marks reused across searches: a slot is visited when its mark equals the
/// current epoch, so starting a search is one increment instead of clearing a set.
#[derive(Default)]
struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// True when `slot` was not yet visited (and marks it).
    #[inline]
    fn insert(&mut self, slot: usize) -> bool {
        let mark = &mut self.marks[slot];
        if *mark == self.epoch {
            return false;
        }
        *mark = self.epoch;
        true
    }
}

thread_local! {
    static VISITED: RefCell<Visited> = RefCell::new(Visited::default());
}

/// An HNSW index over vectors of `T` (`f64` or `f32`) with integer ids.
pub struct Hnsw<T: Element = f64> {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    metric: Metric,
    /// Vectors stored contiguously, `dim` values per slot.
    data: Vec<T>,
    ids: Vec<i64>,
    neighbors: Vec<Vec<Vec<usize>>>,
    index: HashMap<i64, usize>,
    entry: Option<usize>,
    max_level: usize,
    count: AtomicU64,
}

impl<T: Element> Hnsw<T> {
    pub fn new(dim: usize, m: usize, ef_construction: usize, metric: Metric) -> Self {
        Self {
            dim,
            m,
            ef_construction,
            metric,
            data: Vec::new(),
            ids: Vec::new(),
            neighbors: Vec::new(),
            index: HashMap::new(),
            entry: None,
            max_level: 0,
            count: AtomicU64::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Distance computations so far (same accounting as the Python `DistanceCounter`).
    pub fn distance_count(&self) -> u64 {
        self.count.load(AtomicOrdering::Relaxed)
    }

    pub fn reset_count(&self) {
        self.count.store(0, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn vector(&self, slot: usize) -> &[T] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    #[inline]
    fn dist(&self, query: &[T], slot: usize, count: &mut u64) -> f64 {
        *count += 1;
        T::distance(self.metric, query, self.vector(slot))
    }

    /// Insert `vector` under `id` with the given top `level`.
    pub fn add(&mut self, id: i64, vector: &[f64], level: usize) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "expected dimension {}, got {}",
                self.dim,
                vector.len()
            ));
        }
        if self.index.contains_key(&id) {
            return Err(format!("duplicate node id: {id}"));
        }
        let query: Vec<T> = vector.iter().map(|&x| T::from_f64(x)).collect();
        let slot = self.ids.len();
        self.data.extend_from_slice(&query);
        self.ids.push(id);
        self.neighbors.push(vec![Vec::new(); level + 1]);
        self.index.insert(id, slot);

        let Some(mut entry) = self.entry else {
            self.entry = Some(slot);
            self.max_level = level;
            return Ok(());
        };
        let mut count = 0u64;
        for lc in ((level + 1)..=self.max_level).rev() {
            entry = self.greedy(&query, entry, lc, &mut count);
        }
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates =
                self.search_layer(&query, &[entry], lc, self.ef_construction, &mut count);
            // Python sorts the candidates by a fresh distance call each; the values are
            // the ones search_layer already computed, so they are reused and counted.
            count += candidates.len() as u64;
            let scored = candidates
                .iter()
                .map(|&(s, d)| (d, self.ids[s], s))
                .collect();
            let neighbors = self.select_scored(scored, self.m, &mut count);
            for &n in &neighbors {
                self.connect(n, slot, lc, &mut count);
            }
            self.neighbors[slot][lc] = neighbors;
            // Python: min(candidates, key=dist), first minimum wins.
            let mut best: Option<(f64, usize)> = None;
            for &(c, d) in &candidates {
                count += 1;
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, c));
                }
            }
            if let Some((_, c)) = best {
                entry = c;
            }
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(slot);
        }
        *self.count.get_mut() += count;
        Ok(())
    }

    fn connect(&mut self, source: usize, target: usize, level: usize, count: &mut u64) {
        self.neighbors[source][level].push(target);
        let limit = if level == 0 { 2 * self.m } else { self.m };
        if self.neighbors[source][level].len() > limit {
            let pruned = self.select_neighbors(
                self.vector(source),
                &self.neighbors[source][level],
                limit,
                count,
            );
            self.neighbors[source][level] = pruned;
        }
    }

    fn greedy(&self, query: &[T], entry: usize, level: usize, count: &mut u64) -> usize {
        let mut current = entry;
        let mut current_dist = self.dist(query, current, count);
        let mut improved = true;
        while improved {
            improved = false;
            // Python iterates the list of the node the pass started from while
            // `current` changes inside the loop.
            for &n in &self.neighbors[current][level] {
                let d = self.dist(query, n, count);
                if d < current_dist {
                    current_dist = d;
                    current = n;
                    improved = true;
                }
            }
        }
        current
    }

    /// The `ef` closest nodes found from `entries`, as `(slot, distance)` in the order
    /// the results heap pops them (farthest first).
    fn search_layer(
        &self,
        query: &[T],
        entries: &[usize],
        level: usize,
        ef: usize,
        count: &mut u64,
    ) -> Vec<(usize, f64)> {
        VISITED.with(|cell| {
            let mut visited = cell.borrow_mut();
            visited.begin(self.ids.len());
            let mut candidates = MinHeap::default();
            let mut results = MinHeap::default(); // keyed (-d, id): pops the farthest
            for &e in entries {
                visited.insert(e);
                let d = self.dist(query, e, count);
                candidates.push(Key(d, self.ids[e]), e);
                results.push(Key(-d, self.ids[e]), e);
            }
            while let Some(Item {
                key: Key(dist, _),
                slot: current,
            }) = candidates.pop()
            {
                let worst = -results.peek().map_or(f64::INFINITY, |i| i.key.0);
                if dist > worst && results.len() >= ef {
                    break;
                }
                for &n in &self.neighbors[current][level] {
                    if !visited.insert(n) {
                        continue;
                    }
                    let d = self.dist(query, n, count);
                    let farthest = -results.peek().map_or(f64::INFINITY, |i| i.key.0);
                    if results.len() < ef || d < farthest {
                        candidates.push(Key(d, self.ids[n]), n);
                        results.push(Key(-d, self.ids[n]), n);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
            let mut out = Vec::with_capacity(results.len());
            while let Some(Item {
                key: Key(neg, _),
                slot,
            }) = results.pop()
            {
                out.push((slot, -neg));
            }
            out
        })
    }

    /// The distance heuristic over `candidates` (deduplicated), measured from `query`.
    fn select_neighbors(
        &self,
        query: &[T],
        candidates: &[usize],
        m: usize,
        count: &mut u64,
    ) -> Vec<usize> {
        let mut unique: Vec<usize> = candidates.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let scored = unique
            .iter()
            .map(|&c| (self.dist(query, c, count), self.ids[c], c))
            .collect();
        self.select_scored(scored, m, count)
    }

    /// Keep a candidate only if it is closer to the query than to every neighbor
    /// already selected. `ordered` holds `(distance to query, id, slot)`.
    fn select_scored(
        &self,
        mut ordered: Vec<(f64, i64, usize)>,
        m: usize,
        count: &mut u64,
    ) -> Vec<usize> {
        ordered.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        for &(cand_dist, _, c) in &ordered {
            if selected.len() >= m {
                break;
            }
            // Python recomputes the candidate's distance here; reuse it, count it.
            *count += 1;
            let cand = self.vector(c);
            let keep = selected
                .iter()
                .all(|&chosen| self.dist(cand, chosen, count) >= cand_dist);
            if keep {
                selected.push(c);
            }
        }
        selected
    }

    /// The `k` approximate nearest `(id, distance)` pairs.
    pub fn search(
        &self,
        query: &[f64],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(i64, f64)>, String> {
        let Some(mut entry) = self.entry else {
            return Ok(Vec::new());
        };
        if query.len() != self.dim {
            return Err(format!(
                "expected dimension {}, got {}",
                self.dim,
                query.len()
            ));
        }
        let query: Vec<T> = query.iter().map(|&x| T::from_f64(x)).collect();
        let ef = ef.unwrap_or(self.ef_construction.max(k));
        let mut count = 0u64;
        for lc in (1..=self.max_level).rev() {
            entry = self.greedy(&query, entry, lc, &mut count);
        }
        let found = self.search_layer(&query, &[entry], 0, ef, &mut count);
        self.count.fetch_add(count, AtomicOrdering::Relaxed);
        let mut scored: Vec<(i64, f64)> = found.iter().map(|&(n, d)| (self.ids[n], d)).collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Search many queries in parallel.
    pub fn search_batch(
        &self,
        queries: &[Vec<f64>],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<Vec<(i64, f64)>>, String> {
        queries.par_iter().map(|q| self.search(q, k, ef)).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    fn recall_on_small_data<T: Element>(metric: Metric) -> f64 {
        let mut s = 42;
        let points: Vec<Vec<f64>> = (0..300)
            .map(|_| (0..8).map(|_| lcg(&mut s)).collect())
            .collect();
        let mut index = Hnsw::<T>::new(8, 8, 64, metric);
        for (i, p) in points.iter().enumerate() {
            let level = if i % 16 == 0 { 1 } else { 0 };
            index.add(i as i64, p, level).unwrap();
        }
        let mut hits = 0;
        for q in points.iter().take(50) {
            let mut exact: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, metric.distance(q, p)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let got = index.search(q, 10, None).unwrap();
            let truth: HashSet<i64> = exact.iter().take(10).map(|e| e.0 as i64).collect();
            hits += got.iter().filter(|g| truth.contains(&g.0)).count();
        }
        hits as f64 / 500.0
    }

    #[test]
    fn finds_exact_neighbors_on_small_data() {
        for metric in [Metric::Euclidean, Metric::Cosine] {
            assert!(recall_on_small_data::<f64>(metric) > 0.95, "{metric:?}");
            assert!(recall_on_small_data::<f32>(metric) > 0.95, "{metric:?} f32");
        }
    }

    #[test]
    fn errors_and_empty() {
        let mut index = Hnsw::<f64>::new(2, 4, 16, Metric::Cosine);
        assert!(index.search(&[1.0, 0.0], 3, None).unwrap().is_empty());
        index.add(1, &[1.0, 0.0], 0).unwrap();
        assert!(index.add(1, &[0.0, 1.0], 0).is_err());
        assert!(index.add(2, &[0.0], 0).is_err());
        assert!(index.search(&[1.0], 1, None).is_err());
        assert_eq!(index.search(&[1.0, 0.0], 5, None).unwrap(), vec![(1, 0.0)]);
        let batch = index
            .search_batch(&[vec![1.0, 0.0], vec![0.0, 1.0]], 1, Some(4))
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(index.len(), 1);
    }
}
