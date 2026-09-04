//! Metadata storage and filtered vector search.
//!
//! This module provides an index-agnostic metadata layer supporting post-filtered
//! vector search ("over-fetch then filter" strategy):
//! - [`MetaValue`]: Primitive metadata types ([`MetaValue::Int`], [`MetaValue::Float`],
//!   [`MetaValue::Str`], [`MetaValue::Bool`]).
//! - [`MetadataStore`]: An in-memory store mapping vector IDs to key-value metadata attributes.
//! - [`Filter`]: Composable filter expressions ([`Filter::Eq`], [`Filter::Gt`], [`Filter::Lt`],
//!   [`Filter::And`], [`Filter::Or`], [`Filter::Not`]).
//! - [`matches`]: Recursive boolean evaluation of filter expressions against an ID's metadata.
//! - [`filtered_top_k`]: Post-filtering pipeline operating on `Vec<ScoredId>` from any index search.
//!
//! # Separation of Concerns
//! Metadata storage is completely decoupled from index internals (`FlatIndex`, `IVFIndex`,
//! `HnswGraph`, `IVFPQIndex`). Metadata is updated and queried independently, keyed purely by
//! external `u64` vector IDs.

use std::collections::HashMap;

use crate::core::topk::ScoredId;

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point number.
    Float(f64),
    /// UTF-8 text string.
    Str(String),
    /// Boolean flag.
    Bool(bool),
}

/// In-memory store for vector metadata attributes.
///
/// Maps vector external IDs (`u64`) to a dictionary of field names and [`MetaValue`] values.
#[derive(Debug, Clone, Default)]
pub struct MetadataStore {
    data: HashMap<u64, HashMap<String, MetaValue>>,
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Set a metadata attribute on a given vector ID.
    pub fn set(&mut self, id: u64, field: &str, value: MetaValue) {
        self.data
            .entry(id)
            .or_default()
            .insert(field.to_string(), value);
    }

    /// Retrieve a reference to a specific metadata attribute of an ID.
    ///
    /// Returns `None` if the ID does not exist in the store or if the field is not present.
    pub fn get(&self, id: u64, field: &str) -> Option<&MetaValue> {
        self.data.get(&id).and_then(|fields| fields.get(field))
    }

    /// Retrieve all metadata attributes for a given vector ID.
    ///
    /// Returns `None` if the ID has no registered metadata.
    pub fn get_all(&self, id: u64) -> Option<&HashMap<String, MetaValue>> {
        self.data.get(&id)
    }

    /// Remove all metadata associated with a given vector ID.
    pub fn remove(&mut self, id: u64) {
        self.data.remove(&id);
    }

    /// Return the number of unique vector IDs currently tracked in the store.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the store is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A filter predicate for querying vector metadata.
///
/// Supports equality, numeric comparisons with automatic coercion, and boolean combinations.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Exact equality match (`field == value`).
    Eq(String, MetaValue),
    /// Greater-than comparison (`field > value`).
    ///
    /// Only valid for numeric fields ([`MetaValue::Int`] and [`MetaValue::Float`]).
    /// Attempting `Gt` on [`MetaValue::Str`] or [`MetaValue::Bool`] is a configuration error
    /// and evaluates to `false` without panicking.
    Gt(String, MetaValue),
    /// Less-than comparison (`field < value`).
    ///
    /// Only valid for numeric fields ([`MetaValue::Int`] and [`MetaValue::Float`]).
    /// Attempting `Lt` on [`MetaValue::Str`] or [`MetaValue::Bool`] is a configuration error
    /// and evaluates to `false` without panicking.
    Lt(String, MetaValue),
    /// Logical AND of two filter sub-expressions.
    And(Box<Filter>, Box<Filter>),
    /// Logical OR of two filter sub-expressions.
    Or(Box<Filter>, Box<Filter>),
    /// Logical NOT of a filter sub-expression.
    Not(Box<Filter>),
}

/// Helper to convert numeric [`MetaValue`] variants to `f64` for cross-type comparison.
#[inline]
fn to_f64(val: &MetaValue) -> Option<f64> {
    match val {
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Float(f) => Some(*f),
        MetaValue::Str(_) | MetaValue::Bool(_) => None,
    }
}

/// Compare two [`MetaValue`] instances for equality, supporting numeric coercion between
/// [`MetaValue::Int`] and [`MetaValue::Float`].
fn values_equal(stored: &MetaValue, target: &MetaValue) -> bool {
    match (stored, target) {
        (MetaValue::Int(a), MetaValue::Int(b)) => a == b,
        (MetaValue::Float(a), MetaValue::Float(b)) => a == b,
        (MetaValue::Str(a), MetaValue::Str(b)) => a == b,
        (MetaValue::Bool(a), MetaValue::Bool(b)) => a == b,
        // Numeric coercion across Int and Float
        (MetaValue::Int(a), MetaValue::Float(b)) => (*a as f64) == *b,
        (MetaValue::Float(a), MetaValue::Int(b)) => *a == (*b as f64),
        _ => false,
    }
}

/// Perform a numeric comparison (`>`, `<`) between two [`MetaValue`] instances.
///
/// Returns `false` if either operand is non-numeric ([`MetaValue::Str`] or [`MetaValue::Bool`]).
fn numeric_compare<F>(stored: &MetaValue, target: &MetaValue, cmp: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    match (to_f64(stored), to_f64(target)) {
        (Some(a), Some(b)) => cmp(a, b),
        _ => false,
    }
}

/// Internal recursive evaluator against an ID's metadata dictionary.
fn eval_filter(fields: &HashMap<String, MetaValue>, filter: &Filter) -> bool {
    match filter {
        Filter::Eq(field, target_val) => match fields.get(field) {
            Some(stored_val) => values_equal(stored_val, target_val),
            None => false, // missing field evaluates to false
        },
        Filter::Gt(field, target_val) => match fields.get(field) {
            Some(stored_val) => numeric_compare(stored_val, target_val, |a, b| a > b),
            None => false, // missing field evaluates to false
        },
        Filter::Lt(field, target_val) => match fields.get(field) {
            Some(stored_val) => numeric_compare(stored_val, target_val, |a, b| a < b),
            None => false, // missing field evaluates to false
        },
        Filter::And(left, right) => eval_filter(fields, left) && eval_filter(fields, right),
        Filter::Or(left, right) => eval_filter(fields, left) || eval_filter(fields, right),
        Filter::Not(sub_filter) => !eval_filter(fields, sub_filter),
    }
}

/// Evaluate whether a vector ID satisfies the specified [`Filter`] condition.
///
/// # "Missing Means Non-Match" Design Convention
/// - If `id` has **no metadata at all** in `store`, this function returns `false`.
/// - If `id` has metadata but is missing the specific field referenced in a comparison clause,
///   that clause evaluates to `false`.
/// - If a comparison is semantically invalid (e.g. `Gt` on string or boolean fields), it
///   evaluates to `false` without crashing or panicking.
///
/// # Numeric Coercion
/// Comparisons between integer and floating-point fields automatically coerce to `f64`
/// (e.g. stored `Int(5)` compared against filter `Float(3.0)` via `Gt` correctly evaluates to `true`).
pub fn matches(store: &MetadataStore, id: u64, filter: &Filter) -> bool {
    let fields = match store.get_all(id) {
        Some(f) => f,
        None => return false,
    };
    eval_filter(fields, filter)
}

/// Filter candidate search results by metadata, retaining the top `k` matching survivors.
///
/// Implements the "over-fetch then filter" pattern:
/// 1. The caller queries an index with `overfetch_k > k` candidates.
/// 2. Candidates are filtered against `store` using [`matches`].
/// 3. The first `k` surviving candidates are returned, maintaining the index's original similarity ranking.
///
/// # Honest Limitation: Fewer Than `k` Results
/// If the metadata filter is highly selective, fewer than `k` candidates may survive even from
/// a large candidate pool. In this case, `filtered_top_k` returns however many candidates survived
/// without padding or erroring. Callers requiring exactly `k` results may need to re-query the index
/// with a larger `overfetch_k`.
pub fn filtered_top_k(
    candidates: Vec<ScoredId>,
    store: &MetadataStore,
    filter: &Filter,
    k: usize,
) -> Vec<ScoredId> {
    candidates
        .into_iter()
        .filter(|c| matches(store, c.id, filter))
        .take(k)
        .collect()
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::{FlatIndex, Metric};

    /// Test 1: MetadataStore basic get/set/remove work correctly for a single id with multiple fields.
    #[test]
    fn test_metadata_store_basic_crud() {
        let mut store = MetadataStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        let id = 42u64;
        store.set(id, "category", MetaValue::Str("electronics".into()));
        store.set(id, "price", MetaValue::Float(299.99));
        store.set(id, "in_stock", MetaValue::Bool(true));
        store.set(id, "rating", MetaValue::Int(5));

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        assert_eq!(
            store.get(id, "category"),
            Some(&MetaValue::Str("electronics".into()))
        );
        assert_eq!(store.get(id, "price"), Some(&MetaValue::Float(299.99)));
        assert_eq!(store.get(id, "in_stock"), Some(&MetaValue::Bool(true)));
        assert_eq!(store.get(id, "rating"), Some(&MetaValue::Int(5)));

        let all = store.get_all(id).unwrap();
        assert_eq!(all.len(), 4);

        store.remove(id);
        assert_eq!(store.len(), 0);
        assert!(store.get(id, "category").is_none());
        assert!(store.get_all(id).is_none());
    }

    /// Test 2: get() on a nonexistent id or nonexistent field returns None, not a panic.
    #[test]
    fn test_get_nonexistent_id_and_field() {
        let mut store = MetadataStore::new();
        store.set(1, "title", MetaValue::Str("vecta book".into()));

        // Nonexistent field on existing ID
        assert!(store.get(1, "author").is_none());

        // Nonexistent ID
        assert!(store.get(999, "title").is_none());
        assert!(store.get_all(999).is_none());
    }

    /// Test 3: Filter::Eq matches correctly for Int, Str, Bool, Float field types.
    #[test]
    fn test_filter_eq_all_types() {
        let mut store = MetadataStore::new();
        let id = 10u64;
        store.set(id, "count", MetaValue::Int(100));
        store.set(id, "tag", MetaValue::Str("rust".into()));
        store.set(id, "active", MetaValue::Bool(true));
        store.set(id, "ratio", MetaValue::Float(0.75));

        assert!(matches(
            &store,
            id,
            &Filter::Eq("count".into(), MetaValue::Int(100))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Eq("count".into(), MetaValue::Int(200))
        ));

        assert!(matches(
            &store,
            id,
            &Filter::Eq("tag".into(), MetaValue::Str("rust".into()))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Eq("tag".into(), MetaValue::Str("python".into()))
        ));

        assert!(matches(
            &store,
            id,
            &Filter::Eq("active".into(), MetaValue::Bool(true))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Eq("active".into(), MetaValue::Bool(false))
        ));

        assert!(matches(
            &store,
            id,
            &Filter::Eq("ratio".into(), MetaValue::Float(0.75))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Eq("ratio".into(), MetaValue::Float(0.50))
        ));
    }

    /// Test 4: Filter::Gt/Lt work correctly for numeric fields, including the Int-vs-Float
    /// coercion case explicitly (e.g. field stored as Int(5), filtered with Gt("field", Float(3.0)) should match).
    #[test]
    fn test_filter_gt_lt_numeric_and_coercion() {
        let mut store = MetadataStore::new();
        let id = 20u64;
        store.set(id, "int_val", MetaValue::Int(5));
        store.set(id, "float_val", MetaValue::Float(10.5));

        // Int stored, Int filter
        assert!(matches(
            &store,
            id,
            &Filter::Gt("int_val".into(), MetaValue::Int(4))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Gt("int_val".into(), MetaValue::Int(5))
        ));
        assert!(matches(
            &store,
            id,
            &Filter::Lt("int_val".into(), MetaValue::Int(6))
        ));

        // Float stored, Float filter
        assert!(matches(
            &store,
            id,
            &Filter::Gt("float_val".into(), MetaValue::Float(10.0))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Lt("float_val".into(), MetaValue::Float(10.0))
        ));

        // Coercion: stored Int(5) vs filter Float(3.0)
        assert!(matches(
            &store,
            id,
            &Filter::Gt("int_val".into(), MetaValue::Float(3.0))
        ));
        assert!(matches(
            &store,
            id,
            &Filter::Lt("int_val".into(), MetaValue::Float(7.5))
        ));

        // Coercion: stored Float(10.5) vs filter Int(10)
        assert!(matches(
            &store,
            id,
            &Filter::Gt("float_val".into(), MetaValue::Int(10))
        ));
        assert!(matches(
            &store,
            id,
            &Filter::Lt("float_val".into(), MetaValue::Int(11))
        ));

        // Coercion: stored Int(5) vs filter Float(5.0) Eq
        assert!(matches(
            &store,
            id,
            &Filter::Eq("int_val".into(), MetaValue::Float(5.0))
        ));
    }

    /// Test 5: Filter::Gt/Lt on a Str or Bool field returns false (not a panic) — test this
    /// explicitly as the documented "invalid comparison = no match" behavior.
    #[test]
    fn test_filter_gt_lt_invalid_types_return_false() {
        let mut store = MetadataStore::new();
        let id = 30u64;
        store.set(id, "name", MetaValue::Str("vecta".into()));
        store.set(id, "flag", MetaValue::Bool(true));

        // Gt/Lt on Str field
        assert!(!matches(
            &store,
            id,
            &Filter::Gt("name".into(), MetaValue::Int(100))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Lt("name".into(), MetaValue::Str("other".into()))
        ));

        // Gt/Lt on Bool field
        assert!(!matches(
            &store,
            id,
            &Filter::Gt("flag".into(), MetaValue::Bool(false))
        ));
        assert!(!matches(
            &store,
            id,
            &Filter::Lt("flag".into(), MetaValue::Float(1.0))
        ));
    }

    /// Test 6: And/Or/Not combinations work correctly — test a realistic compound filter
    /// like the original learning goal: "id > 100 AND category = 'electronics'" against a small
    /// hand-crafted MetadataStore with known ids/categories, confirm exactly the expected ids match.
    #[test]
    fn test_compound_filter_id_gt_100_and_category_electronics() {
        let mut store = MetadataStore::new();

        let dataset = vec![
            (50u64, 50i64, "electronics"),
            (75u64, 75i64, "books"),
            (100u64, 100i64, "electronics"), // id == 100, not > 100
            (105u64, 105i64, "books"),
            (150u64, 150i64, "electronics"), // MATCH
            (200u64, 200i64, "electronics"), // MATCH
            (250u64, 250i64, "clothing"),
        ];

        for (vec_id, id_val, cat) in dataset {
            store.set(vec_id, "id", MetaValue::Int(id_val));
            store.set(vec_id, "category", MetaValue::Str(cat.into()));
        }

        // Filter: id > 100 AND category = 'electronics'
        let filter = Filter::And(
            Box::new(Filter::Gt("id".into(), MetaValue::Int(100))),
            Box::new(Filter::Eq(
                "category".into(),
                MetaValue::Str("electronics".into()),
            )),
        );

        let test_ids = [50, 75, 100, 105, 150, 200, 250];
        let mut matching_ids = Vec::new();

        for &id in &test_ids {
            if matches(&store, id, &filter) {
                matching_ids.push(id);
            }
        }

        println!("\nPhase 29 Test 6: Compound Filter Evaluation (id > 100 AND category = 'electronics'):");
        println!("  Candidate IDs evaluated: {:?}", test_ids);
        println!("  Matching IDs returned:   {:?}", matching_ids);

        assert_eq!(matching_ids, vec![150, 200]);

        // Also test OR and NOT combinations
        let or_filter = Filter::Or(
            Box::new(Filter::Eq(
                "category".into(),
                MetaValue::Str("books".into()),
            )),
            Box::new(Filter::Eq(
                "category".into(),
                MetaValue::Str("clothing".into()),
            )),
        );
        assert!(matches(&store, 75, &or_filter));
        assert!(matches(&store, 250, &or_filter));
        assert!(!matches(&store, 50, &or_filter));

        let not_filter = Filter::Not(Box::new(Filter::Eq(
            "category".into(),
            MetaValue::Str("electronics".into()),
        )));
        assert!(matches(&store, 75, &not_filter)); // books != electronics
        assert!(!matches(&store, 150, &not_filter)); // electronics == electronics -> NOT is false
    }

    /// Test 7: matches() on an id with NO metadata returns false for any filter (not a panic, not a crash).
    #[test]
    fn test_matches_id_with_no_metadata_returns_false() {
        let store = MetadataStore::new();
        let ghost_id = 9999u64;

        assert!(!matches(
            &store,
            ghost_id,
            &Filter::Eq("field".into(), MetaValue::Int(1))
        ));
        assert!(!matches(
            &store,
            ghost_id,
            &Filter::Gt("field".into(), MetaValue::Int(1))
        ));
        assert!(!matches(
            &store,
            ghost_id,
            &Filter::Lt("field".into(), MetaValue::Int(1))
        ));
        assert!(!matches(
            &store,
            ghost_id,
            &Filter::Not(Box::new(Filter::Eq("field".into(), MetaValue::Int(1))))
        ));
    }

    /// Test 8: filtered_top_k() end-to-end integration test: build a small FlatIndex with 10 known vectors,
    /// attach metadata to each (some matching a test filter, some not), call index.search() with an overfetch_k
    /// larger than the target k, apply filtered_top_k(), confirm the result contains ONLY vectors matching
    /// the filter, correctly ranked, and confirm the count is <= the requested k.
    #[test]
    fn test_filtered_top_k_end_to_end_with_flat_index() {
        let dim = 3;
        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        let mut store = MetadataStore::new();

        // 10 vectors along the x-axis with increasing distance from origin [0, 0, 0]
        for id in 1..=10u64 {
            let coord = id as f32;
            index.add(id, &[coord, 0.0, 0.0]);

            // Alternate category: odd = "audio", even = "video"
            let cat = if id % 2 == 1 { "audio" } else { "video" };
            store.set(id, "category", MetaValue::Str(cat.into()));
            store.set(id, "price", MetaValue::Int((id * 10) as i64));
        }

        // Query point at origin [0, 0, 0]
        let query = [0.0f32, 0.0, 0.0];
        let target_k = 3;
        let overfetch_k = 8;

        // Step 1: Overfetch candidates from index
        let candidates = index.search(&query, overfetch_k);
        assert_eq!(candidates.len(), overfetch_k);

        // Filter: category == "video" (matches even IDs: 2, 4, 6, 8, 10)
        let filter = Filter::Eq("category".into(), MetaValue::Str("video".into()));

        // Step 2: Apply filtered_top_k
        let filtered_results = filtered_top_k(candidates.clone(), &store, &filter, target_k);

        println!("\nPhase 29 Test 8: End-to-End Filtered Search Pipeline:");
        println!("  Target k:       {}", target_k);
        println!("  Over-fetch k:   {}", overfetch_k);
        println!(
            "  Candidates count from index.search(): {}",
            candidates.len()
        );
        println!(
            "  Survivors after filtered_top_k():     {}",
            filtered_results.len()
        );
        for (rank, res) in filtered_results.iter().enumerate() {
            let cat = store.get(res.id, "category").unwrap();
            println!(
                "    Rank #{}: id={} (score={:.4}, category={:?})",
                rank + 1,
                res.id,
                res.score,
                cat
            );
        }

        // Validate top k <= target_k
        assert_eq!(filtered_results.len(), target_k);

        // Validate all returned results match filter
        for res in &filtered_results {
            assert!(matches(&store, res.id, &filter));
        }

        // Validate ranking preserved: nearest even vectors are id=2, id=4, id=6
        assert_eq!(filtered_results[0].id, 2);
        assert_eq!(filtered_results[1].id, 4);
        assert_eq!(filtered_results[2].id, 6);
        assert!(filtered_results[0].score < filtered_results[1].score);
        assert!(filtered_results[1].score < filtered_results[2].score);
    }

    /// Test 9: Insufficient-candidates test: construct a scenario where the filter is restrictive
    /// enough that fewer than k candidates survive even with a generous overfetch_k, confirm
    /// filtered_top_k() returns however many DID survive (not an error, not padded with anything fake).
    #[test]
    fn test_filtered_top_k_insufficient_candidates_honest_count() {
        let dim = 2;
        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        let mut store = MetadataStore::new();

        for id in 1..=5u64 {
            index.add(id, &[id as f32, 0.0]);
            let available = id == 3; // only id=3 matches!
            store.set(id, "available", MetaValue::Bool(available));
        }

        let query = [0.0f32, 0.0];
        let requested_k = 5;
        let overfetch_k = 5;

        let candidates = index.search(&query, overfetch_k);
        let filter = Filter::Eq("available".into(), MetaValue::Bool(true));

        let filtered = filtered_top_k(candidates, &store, &filter, requested_k);

        // Only 1 item matches, so exactly 1 result returned (not 5, no padding, no error)
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 3);
    }
}
