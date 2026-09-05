//! Multi-layer query search for HNSW graphs.
//!
//! Implements Algorithm 5 (K-NN-SEARCH) from Malkov & Yashunin (2018).
//! Connects entry point descent to layer-0 greedy beam search,
//! translating internal graph indices back to external user IDs.

use crate::core::batch::VectorBatch;
use crate::core::flat_index::Metric;
use crate::core::hnsw::graph::HnswGraph;
use crate::core::hnsw::layer::HnswConfig;
use crate::core::hnsw::search_layer::greedy_search_layer;
use crate::core::topk::{top_k_largest, top_k_smallest, ScoredId};
use crate::core::vector::{cosine_similarity, dot_product};

impl HnswGraph {
    /// Search for the top-`k` nearest neighbors to `query` in the HNSW graph.
    ///
    /// # Algorithm (Malkov & Yashunin 2018, Algorithm 5):
    /// 1. Validate query dimension; return empty `Vec` if graph is empty, `k == 0`, or `ef_search == 0`.
    /// 2. Begin at `self.entry_point` at the top layer.
    /// 3. Greedily descend from `top_layer` down to layer 1 using `ef = 1`.
    /// 4. At layer 0, run [`greedy_search_layer`] with `ef = ef_search` to find candidate nodes.
    /// 5. Map internal node indices back to user-facing external `u64` IDs.
    /// 6. Apply final top-`k` selection and sorting according to `self.metric`.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim`.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<ScoredId> {
        assert_eq!(
            query.len(),
            self.dim,
            "HnswGraph::search: query dimension {} != graph dimension {}",
            query.len(),
            self.dim
        );

        if self.is_empty() || k == 0 || ef_search == 0 {
            return Vec::new();
        }

        let ep = self
            .entry_point
            .expect("HnswGraph must have entry_point when non-empty");
        let top_layer = self.nodes[ep].max_layer;

        let mut eps = vec![ep];

        // 1. Greedy descent through upper layers (top_layer down to 1) with ef=1
        for l in (1..=top_layer).rev() {
            let best = greedy_search_layer(self, query, &eps, l, 1);
            if let Some(first) = best.first() {
                eps = vec![first.id as usize];
            }
        }

        // 2. Wide search at layer 0 with ef=ef_search
        let layer0_candidates = greedy_search_layer(self, query, &eps, 0, ef_search);

        // 3. Map internal node indices back to external IDs and evaluate metric
        let external_candidates: Vec<ScoredId> = layer0_candidates
            .into_iter()
            .filter(|c| !self.tombstones.contains(&(c.id as usize)))
            .map(|c| {
                let internal_idx = c.id as usize;
                let external_id = self.nodes[internal_idx].id;
                let score = match self.metric {
                    Metric::Euclidean => c.score,
                    Metric::Cosine => cosine_similarity(query, self.get_vector(internal_idx)),
                    Metric::DotProduct => dot_product(query, self.get_vector(internal_idx)),
                };
                ScoredId {
                    id: external_id,
                    score,
                }
            })
            .collect();

        // 4. Apply final top-k selection (since ef_search is generally > k)
        match self.metric {
            Metric::Euclidean => top_k_smallest(&external_candidates, k),
            Metric::Cosine | Metric::DotProduct => top_k_largest(&external_candidates, k),
        }
    }

    /// Search for top-`k` neighbors using `config.ef_search`.
    #[inline]
    pub fn search_with_config(
        &self,
        query: &[f32],
        k: usize,
        config: &HnswConfig,
    ) -> Vec<ScoredId> {
        self.search(query, k, config.ef_search)
    }

    /// Bulk search: executes [`search`](Self::search) for every query in `queries`.
    ///
    /// # Panics
    /// Panics if `queries.dim != self.dim`.
    pub fn search_batch(
        &self,
        queries: &VectorBatch,
        k: usize,
        ef_search: usize,
    ) -> Vec<Vec<ScoredId>> {
        assert_eq!(
            queries.dim, self.dim,
            "HnswGraph::search_batch: queries dimension {} != graph dimension {}",
            queries.dim, self.dim
        );

        let mut results = Vec::with_capacity(queries.len());
        for i in 0..queries.len() {
            results.push(self.search(queries.get(i), k, ef_search));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::FlatIndex;
    use crate::core::hnsw::insert::insert;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    /// Test 1: Empty graph search returns empty Vec, no panic.
    #[test]
    fn test_search_empty_graph() {
        let graph = HnswGraph::new(3, Metric::Euclidean);
        let res = graph.search(&[1.0, 2.0, 3.0], 5, 50);
        assert!(res.is_empty());
    }

    /// Test 2: Wrong query dimension panics with clear message.
    #[test]
    #[should_panic(expected = "query dimension 3 != graph dimension 2")]
    fn test_search_wrong_dimension_panics() {
        let graph = HnswGraph::new(2, Metric::Euclidean);
        let _ = graph.search(&[1.0, 2.0, 3.0], 5, 50);
    }

    /// Test 3: Hand-verified small test against FlatIndex oracle:
    /// Build same 6-point dataset from Phase 17, search with high ef_search (exhaustive),
    /// confirm results EXACTLY match FlatIndex results.
    #[test]
    fn test_hand_verified_against_flat_index_oracle() {
        let dim = 2;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);

        let config = HnswConfig {
            m: 2,
            ef_construction: 50,
            ef_search: 50,
        };
        let mut rng = StdRng::seed_from_u64(1234);

        let points: Vec<(u64, [f32; 2])> = vec![
            (1, [1.0, 1.0]),
            (2, [1.0, 1.5]),
            (3, [1.5, 1.0]),
            (4, [9.0, 9.0]),
            (5, [9.0, 8.5]),
            (6, [8.5, 9.0]),
        ];

        for (id, pt) in &points {
            insert(&mut graph, *id, pt, &config, &mut rng).unwrap();
            flat.add(*id, pt);
        }

        let query = [1.1, 1.2];
        let k = points.len();

        let flat_results = flat.search(&query, k);
        let hnsw_results = graph.search(&query, k, 50);

        println!("\nPhase 18 Test 3: HNSW vs FlatIndex hand-verified comparison:");
        for rank in 0..k {
            println!(
                "  Rank {}: Flat (id={}, dist={:.4}) | HNSW (id={}, dist={:.4})",
                rank + 1,
                flat_results[rank].id,
                flat_results[rank].score,
                hnsw_results[rank].id,
                hnsw_results[rank].score
            );
        }

        assert_eq!(hnsw_results.len(), flat_results.len());
        for i in 0..k {
            assert_eq!(
                hnsw_results[i].id, flat_results[i].id,
                "ID mismatch at rank {}",
                i
            );
            assert!(
                (hnsw_results[i].score - flat_results[i].score).abs() < 1e-5,
                "Distance mismatch at rank {}",
                i
            );
        }
    }

    /// Test 4: Single-result sanity test (k=1) on known point.
    #[test]
    fn test_search_k1_closest_point() {
        let dim = 2;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let points = [(10, [0.0, 0.0]), (20, [10.0, 10.0]), (30, [100.0, 100.0])];

        for (id, pt) in &points {
            insert(&mut graph, *id, pt, &config, &mut rng).unwrap();
        }

        let query = [0.1, -0.1]; // clearly closest to id 10
        let res = graph.search(&query, 1, 50);

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 10);
    }

    /// Tests 5 & 6: Recall@10 vs ef_search on 1,000 vectors against FlatIndex oracle.
    /// Confirms recall increases monotonically with ef_search.
    #[test]
    fn test_recall_vs_ef_search_curve() {
        let n = 1000;
        let dim = 32;
        let top_k = 10;
        let num_queries = 20;

        let mut rng = StdRng::seed_from_u64(12345);

        // 1. Generate data
        let mut data = VectorBatch::new(dim);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-10.0..10.0));
            }
            data.push(&v);
        }

        // 2. Build FlatIndex (Ground Truth Oracle)
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);
        let ids: Vec<u64> = (0..n as u64).collect();
        flat.add_batch(&ids, &data);

        // 3. Build HnswGraph
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };

        for (i, &id) in ids.iter().enumerate() {
            insert(&mut graph, id, data.get(i), &config, &mut rng).unwrap();
        }

        // 4. Generate queries and compute ground truth
        let mut queries = Vec::new();
        let mut ground_truth: Vec<HashSet<u64>> = Vec::new();
        for _ in 0..num_queries {
            let mut q = Vec::with_capacity(dim);
            for _ in 0..dim {
                q.push(rng.gen_range(-10.0..10.0));
            }
            let gt_results = flat.search(&q, top_k);
            ground_truth.push(gt_results.iter().map(|s| s.id).collect());
            queries.push(q);
        }

        // 5. Test ef_search progression: [5, 10, 20, 50, 100]
        let ef_values = [5, 10, 20, 50, 100];
        let mut recall_curve = Vec::new();

        println!(
            "\nPhase 18 Test 5: Recall@{} vs ef_search (N={}, M=16, ef_c=200):",
            top_k, n
        );
        for &ef in &ef_values {
            let mut total_overlap = 0;
            for (q_idx, q) in queries.iter().enumerate() {
                let hnsw_res = graph.search(q, top_k, ef);
                let hnsw_ids: HashSet<u64> = hnsw_res.iter().map(|s| s.id).collect();
                total_overlap += hnsw_ids.intersection(&ground_truth[q_idx]).count();
            }

            let recall = (total_overlap as f64) / ((num_queries * top_k) as f64);
            recall_curve.push(recall);
            println!(
                "  ef_search={:>3}: Recall@{:<2} = {:>5.1}%",
                ef,
                top_k,
                recall * 100.0
            );
        }

        // Check monotonic non-decreasing progression
        for i in 0..(recall_curve.len() - 1) {
            assert!(
                recall_curve[i] <= recall_curve[i + 1] + 1e-6,
                "Recall dropped when increasing ef_search: {} > {}",
                recall_curve[i],
                recall_curve[i + 1]
            );
        }

        // High ef_search should achieve very high recall (>= 90%)
        assert!(
            *recall_curve.last().unwrap() >= 0.90,
            "High ef_search (100) should reach at least 90% recall on 1k vectors"
        );
    }

    /// Test 7: search_batch returns results preserving query order.
    #[test]
    fn test_search_batch_preserves_order() {
        let dim = 2;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let points = [
            (1, [1.0, 1.0]),
            (2, [2.0, 2.0]),
            (3, [9.0, 9.0]),
            (4, [10.0, 10.0]),
        ];

        for (id, pt) in &points {
            insert(&mut graph, *id, pt, &config, &mut rng).unwrap();
        }

        let mut queries = VectorBatch::new(dim);
        queries.push(&[1.1, 1.1]);
        queries.push(&[9.5, 9.5]);

        let batch_results = graph.search_batch(&queries, 2, 50);
        assert_eq!(batch_results.len(), 2);

        for (q_idx, single_q) in [queries.get(0), queries.get(1)].iter().enumerate() {
            let single_res = graph.search(single_q, 2, 50);
            assert_eq!(batch_results[q_idx].len(), single_res.len());
            for r in 0..single_res.len() {
                assert_eq!(batch_results[q_idx][r].id, single_res[r].id);
                assert!((batch_results[q_idx][r].score - single_res[r].score).abs() < 1e-5);
            }
        }
    }
}
