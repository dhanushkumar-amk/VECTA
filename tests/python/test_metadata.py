import pytest
import vecta


class TestMetadataStoreCRUD:
    """Test 1, 2, 3: Basic CRUD, type round-tripping, and error handling for MetadataStore."""

    def test_metadata_store_round_trip_types(self):
        """Test 1: MetadataStore set/get/remove for int, float, str, bool."""
        store = vecta.MetadataStore()
        assert len(store) == 0

        # Set attributes
        store.set(100, "age", 30)
        store.set(100, "rating", 4.75)
        store.set(100, "category", "electronics")
        store.set(100, "in_stock", True)

        assert len(store) == 1

        # Verify exact round-trip value and type
        val_int = store.get(100, "age")
        assert val_int == 30
        assert isinstance(val_int, int)

        val_float = store.get(100, "rating")
        assert abs(val_float - 4.75) < 1e-6
        assert isinstance(val_float, float)

        val_str = store.get(100, "category")
        assert val_str == "electronics"
        assert isinstance(val_str, str)

        val_bool = store.get(100, "in_stock")
        assert val_bool is True
        assert isinstance(val_bool, bool)

        # Remove
        store.remove(100)
        assert len(store) == 0
        assert store.get(100, "age") is None

    def test_set_unsupported_value_type_raises_value_error(self):
        """Test 2: set() with an unsupported value type (e.g. list, dict) raises ValueError."""
        store = vecta.MetadataStore()

        with pytest.raises(ValueError, match="unsupported metadata value type"):
            store.set(1, "tags", ["tag1", "tag2"])

        with pytest.raises(ValueError, match="unsupported metadata value type"):
            store.set(1, "meta", {"k": "v"})

    def test_get_nonexistent_returns_none(self):
        """Test 3: get() on a nonexistent id or field returns None, not an exception."""
        store = vecta.MetadataStore()
        store.set(1, "title", "vecta")

        assert store.get(1, "nonexistent_field") is None
        assert store.get(999, "title") is None


class TestFilterParsingAndEvaluation:
    """Test 4, 5, 6: Filter expression parsing, compound evaluation, and syntax errors."""

    def test_simple_filter_parsing_and_filtered_search(self):
        """Test 4: Simple filter parsing ("eq", "category", "electronics")."""
        store = vecta.MetadataStore()
        store.set(1, "category", "electronics")
        store.set(2, "category", "books")

        candidates = [(1, 0.1), (2, 0.2)]
        survivors = vecta.filtered_search(
            candidates, store, ("eq", "category", "electronics"), k=10
        )
        assert len(survivors) == 1
        assert survivors[0][0] == 1

    def test_compound_filter_id_gt_100_and_category_electronics(self):
        """Test 5: Compound ("and", ("gt", "id", 100), ("eq", "category", "electronics"))."""
        store = vecta.MetadataStore()
        # Setup test items
        dataset = [
            (50, 50, "electronics"),
            (75, 75, "books"),
            (100, 100, "electronics"),
            (105, 105, "books"),
            (150, 150, "electronics"),  # MATCH
            (200, 200, "electronics"),  # MATCH
            (250, 250, "clothing"),
        ]
        for vec_id, id_val, cat in dataset:
            store.set(vec_id, "id", id_val)
            store.set(vec_id, "category", cat)

        candidates = [
            (50, 0.1),
            (75, 0.2),
            (100, 0.3),
            (105, 0.4),
            (150, 0.5),
            (200, 0.6),
            (250, 0.7),
        ]

        filter_expr = (
            "and",
            ("gt", "id", 100),
            ("eq", "category", "electronics"),
        )
        filtered = vecta.filtered_search(candidates, store, filter_expr, k=10)

        matching_ids = [res[0] for res in filtered]
        assert matching_ids == [150, 200]

    def test_malformed_filter_syntax_raises_value_error(self):
        """Test 6: Malformed filter syntax raises ValueError with informative messages."""
        store = vecta.MetadataStore()
        candidates = [(1, 0.5)]

        # Unknown operator
        with pytest.raises(ValueError, match="unknown filter operator 'xyz'"):
            vecta.filtered_search(candidates, store, ("xyz", "field", 1), k=5)

        # Non-tuple input
        with pytest.raises(ValueError, match="filter must be a tuple"):
            vecta.filtered_search(candidates, store, "eq", k=5)

        # Empty tuple
        with pytest.raises(ValueError, match="filter tuple cannot be empty"):
            vecta.filtered_search(candidates, store, (), k=5)

        # Wrong element count for 'eq'
        with pytest.raises(ValueError, match="'eq' filter expects 3 elements"):
            vecta.filtered_search(candidates, store, ("eq", "field"), k=5)

        # Wrong element count for 'not'
        with pytest.raises(ValueError, match="'not' filter expects 2 elements"):
            vecta.filtered_search(
                candidates, store, ("not", ("eq", "a", 1), ("eq", "b", 2)), k=5
            )


class TestFilteredSearchIntegration:
    """Test 7, 8: End-to-end integration tests with FlatIndex and IVFIndex."""

    def test_filtered_search_end_to_end_flat_index(self):
        """Test 7: Build FlatIndex with 10 vectors, attach metadata, over-fetch, and filter."""
        dim = 3
        index = vecta.FlatIndex(dim, "euclidean")
        store = vecta.MetadataStore()

        for id_val in range(1, 11):
            coord = float(id_val)
            index.add(id_val, [coord, 0.0, 0.0])
            cat = "audio" if id_val % 2 == 1 else "video"
            store.set(id_val, "category", cat)
            store.set(id_val, "price", id_val * 10)

        query = [0.0, 0.0, 0.0]
        overfetch_k = 8
        target_k = 3

        # Step 1: Over-fetch
        candidates = index.search(query, k=overfetch_k)
        assert len(candidates) == overfetch_k

        # Step 2: Filter by category == "video"
        filter_expr = ("eq", "category", "video")
        survivors = vecta.filtered_search(candidates, store, filter_expr, k=target_k)

        # Step 3: Verify survivors
        assert len(survivors) == target_k
        survivor_ids = [s[0] for s in survivors]
        assert survivor_ids == [2, 4, 6]

        # Verify ranking order preserved
        assert survivors[0][1] < survivors[1][1]
        assert survivors[1][1] < survivors[2][1]

    def test_filtered_search_index_agnostic_ivf_index(self):
        """Test 8: filtered_search with IVFIndex confirming index-agnostic behavior."""
        dim = 4
        num_clusters = 2
        ivf = vecta.IVFIndex(dim, num_clusters, "euclidean")
        store = vecta.MetadataStore()

        # Training vectors
        train_vecs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ]
        ivf.train(train_vecs, k=num_clusters)

        # Add vectors with metadata
        vectors = [
            (10, [1.0, 0.0, 0.0, 0.0], "premium"),
            (20, [0.85, 0.0, 0.0, 0.0], "basic"),
            (30, [0.95, 0.05, 0.0, 0.0], "premium"),
            (40, [0.0, 0.0, 1.0, 0.0], "premium"),
        ]

        for vid, vec, tier in vectors:
            ivf.add(vid, vec)
            store.set(vid, "tier", tier)

        query = [1.0, 0.0, 0.0, 0.0]
        # Search all clusters (nprobe=2) with overfetch
        candidates = ivf.search(query, k=4, nprobe=2)

        # Filter for tier == "premium"
        filter_expr = ("eq", "tier", "premium")
        filtered = vecta.filtered_search(candidates, store, filter_expr, k=2)

        assert len(filtered) == 2
        for vid, _score in filtered:
            assert store.get(vid, "tier") == "premium"

        # Nearest vector to query [1,0,0,0] is id 10
        assert filtered[0][0] == 10
