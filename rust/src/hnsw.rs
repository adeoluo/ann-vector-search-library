//! HNSW (Malkov and Yashunin, 2016), mirroring `ann_vector_search_library.hnsw.HNSW`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use rayon::prelude::*;

use crate::distance::Metric;

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

/// A min-heap with Python `heapq` semantics over `Key`.
#[derive(Default)]
struct MinHeap(BinaryHeap<std::cmp::Reverse<Key>>);

impl MinHeap {
    fn push(&mut self, k: Key) {
        self.0.push(std::cmp::Reverse(k));
    }
    fn pop(&mut self) -> Option<Key> {
        self.0.pop().map(|r| r.0)
    }
    fn peek(&self) -> Option<Key> {
        self.0.peek().map(|r| r.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

struct Node {
    id: i64,
    vector: Vec<f64>,
    neighbors: Vec<Vec<usize>>,
}

/// An HNSW index over `f64` vectors with integer ids.
pub struct Hnsw {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    metric: Metric,
    nodes: Vec<Node>,
    index: HashMap<i64, usize>,
    entry: Option<usize>,
    max_level: usize,
    count: AtomicU64,
}

impl Hnsw {
    pub fn new(dim: usize, m: usize, ef_construction: usize, metric: Metric) -> Self {
        Self {
            dim,
            m,
            ef_construction,
            metric,
            nodes: Vec::new(),
            index: HashMap::new(),
            entry: None,
            max_level: 0,
            count: AtomicU64::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Distance computations so far (same accounting as the Python `DistanceCounter`).
    pub fn distance_count(&self) -> u64 {
        self.count.load(AtomicOrdering::Relaxed)
    }

    pub fn reset_count(&self) {
        self.count.store(0, AtomicOrdering::Relaxed);
    }

    #[inline]
    fn dist_to(&self, query: &[f64], node: usize) -> f64 {
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
        self.metric.distance(query, &self.nodes[node].vector)
    }

    #[inline]
    fn dist_nodes(&self, a: usize, b: usize) -> f64 {
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
        self.metric
            .distance(&self.nodes[a].vector, &self.nodes[b].vector)
    }

    /// Insert `vector` under `id` with the given top `level`.
    pub fn add(&mut self, id: i64, vector: Vec<f64>, level: usize) -> Result<(), String> {
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
        let slot = self.nodes.len();
        self.nodes.push(Node {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.index.insert(id, slot);

        let Some(mut entry) = self.entry else {
            self.entry = Some(slot);
            self.max_level = level;
            return Ok(());
        };
        let query = self.nodes[slot].vector.clone();
        for lc in ((level + 1)..=self.max_level).rev() {
            entry = self.greedy(&query, entry, lc);
        }
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(&query, &[entry], lc, self.ef_construction);
            let neighbors = self.select_neighbors(&query, &candidates, self.m);
            self.nodes[slot].neighbors[lc] = neighbors.clone();
            for &n in &neighbors {
                self.connect(n, slot, lc);
            }
            // Python: min(candidates, key=dist), first minimum wins.
            let mut best: Option<(f64, usize)> = None;
            for &c in &candidates {
                let d = self.dist_to(&query, c);
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
        Ok(())
    }

    fn connect(&mut self, source: usize, target: usize, level: usize) {
        self.nodes[source].neighbors[level].push(target);
        let limit = if level == 0 { 2 * self.m } else { self.m };
        if self.nodes[source].neighbors[level].len() > limit {
            let query = self.nodes[source].vector.clone();
            let current = self.nodes[source].neighbors[level].clone();
            self.nodes[source].neighbors[level] = self.select_neighbors(&query, &current, limit);
        }
    }

    fn greedy(&self, query: &[f64], entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.dist_to(query, current);
        let mut improved = true;
        while improved {
            improved = false;
            // Python iterates the live list while `current` changes inside the loop.
            let mut i = 0;
            let start = current;
            while i < self.nodes[start].neighbors[level].len() {
                let n = self.nodes[start].neighbors[level][i];
                let d = self.dist_to(query, n);
                if d < current_dist {
                    current_dist = d;
                    current = n;
                    improved = true;
                }
                i += 1;
            }
        }
        current
    }

    /// Returns ids in Python's heap-array order is not needed; callers sort or take min.
    fn search_layer(
        &self,
        query: &[f64],
        entries: &[usize],
        level: usize,
        ef: usize,
    ) -> Vec<usize> {
        let mut visited: HashSet<usize> = entries.iter().copied().collect();
        let mut candidates = MinHeap::default();
        let mut results = MinHeap::default(); // keyed (-d, id): pops the farthest
        for &e in entries {
            let d = self.dist_to(query, e);
            candidates.push(Key(d, self.nodes[e].id));
            results.push(Key(-d, self.nodes[e].id));
        }
        while let Some(Key(dist, current_id)) = candidates.pop() {
            let worst = -results.peek().map_or(f64::INFINITY, |k| k.0);
            if dist > worst && results.len() >= ef {
                break;
            }
            let current = self.index[&current_id];
            for &n in &self.nodes[current].neighbors[level] {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.dist_to(query, n);
                let farthest = -results.peek().map_or(f64::INFINITY, |k| k.0);
                if results.len() < ef || d < farthest {
                    candidates.push(Key(d, self.nodes[n].id));
                    results.push(Key(-d, self.nodes[n].id));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<usize> = Vec::with_capacity(results.len());
        while let Some(Key(_, id)) = results.pop() {
            out.push(self.index[&id]);
        }
        out
    }

    /// The distance heuristic: keep a candidate only if it is closer to the query than
    /// to every neighbor already selected.
    fn select_neighbors(&self, query: &[f64], candidates: &[usize], m: usize) -> Vec<usize> {
        let mut unique: Vec<usize> = candidates.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let mut ordered: Vec<(f64, i64, usize)> = unique
            .iter()
            .map(|&c| (self.dist_to(query, c), self.nodes[c].id, c))
            .collect();
        ordered.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut selected: Vec<usize> = Vec::new();
        for &(_, _, c) in &ordered {
            if selected.len() >= m {
                break;
            }
            let cand_dist = self.dist_to(query, c);
            let mut keep = true;
            for &chosen in &selected {
                if self.dist_nodes(c, chosen) < cand_dist {
                    keep = false;
                    break;
                }
            }
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
        let ef = ef.unwrap_or(self.ef_construction.max(k));
        for lc in (1..=self.max_level).rev() {
            entry = self.greedy(query, entry, lc);
        }
        let found = self.search_layer(query, &[entry], 0, ef);
        let mut scored: Vec<(i64, f64)> = found
            .iter()
            .map(|&n| {
                (
                    self.nodes[n].id,
                    self.metric.distance(query, &self.nodes[n].vector),
                )
            })
            .collect();
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
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn finds_exact_neighbors_on_small_data() {
        let mut s = 42;
        let points: Vec<Vec<f64>> = (0..300)
            .map(|_| (0..8).map(|_| lcg(&mut s)).collect())
            .collect();
        let mut index = Hnsw::new(8, 8, 64, Metric::Euclidean);
        for (i, p) in points.iter().enumerate() {
            let level = if i % 16 == 0 { 1 } else { 0 };
            index.add(i as i64, p.clone(), level).unwrap();
        }
        let mut hits = 0;
        for q in points.iter().take(50) {
            let mut exact: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, Metric::Euclidean.distance(q, p)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let got = index.search(q, 10, None).unwrap();
            let truth: HashSet<i64> = exact.iter().take(10).map(|e| e.0 as i64).collect();
            hits += got.iter().filter(|g| truth.contains(&g.0)).count();
        }
        assert!(hits as f64 / 500.0 > 0.95, "recall too low: {hits}/500");
    }

    #[test]
    fn errors_and_empty() {
        let mut index = Hnsw::new(2, 4, 16, Metric::Cosine);
        assert!(index.search(&[1.0, 0.0], 3, None).unwrap().is_empty());
        index.add(1, vec![1.0, 0.0], 0).unwrap();
        assert!(index.add(1, vec![0.0, 1.0], 0).is_err());
        assert!(index.add(2, vec![0.0], 0).is_err());
        assert!(index.search(&[1.0], 1, None).is_err());
        assert_eq!(index.search(&[1.0, 0.0], 5, None).unwrap(), vec![(1, 0.0)]);
        let batch = index
            .search_batch(&[vec![1.0, 0.0], vec![0.0, 1.0]], 1, Some(4))
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(index.len(), 1);
    }
}
