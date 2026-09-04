"""Tests for Phase 27: Index persistence (save/load) and generic loader dispatch."""

import os
import tempfile
import pytest
import vecta


@pytest.fixture
def temp_index_path():
    """Create a temporary path and ensure cleanup after test."""
    fd, path = tempfile.mkstemp(suffix=".vct")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestFlatIndexPersistence:
    """Test save and vecta.load() for FlatIndex."""

    def test_save_and_load_round_trip(self, temp_index_path):
        index = vecta.FlatIndex(dim=3, metric="euclidean")
        vectors = [
            (1, [1.0, 2.0, 3.0]),
            (2, [4.0, 5.0, 6.0]),
            (3, [7.0, 8.0, 9.0]),
        ]
        for vid, v in vectors:
            index.add(vid, v)

        index.save(temp_index_path)

        loaded = vecta.load(temp_index_path)
        assert isinstance(loaded, vecta.FlatIndex)
        assert type(loaded) is vecta.FlatIndex
        assert len(loaded) == 3
        assert loaded.dim() == 3

        # Confirm search results match pre-save
        query = [1.1, 2.1, 3.1]
        orig_res = index.search(query, k=2)
        loaded_res = loaded.search(query, k=2)
        assert orig_res == loaded_res


class TestIVFIndexPersistence:
    """Test save and vecta.load() for IVFIndex."""

    def test_save_and_load_round_trip(self, temp_index_path):
        dim = 4
        index = vecta.IVFIndex(dim=dim, num_clusters=2, metric="euclidean")
        train_data = [
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 1.1, 0.0, 0.0],
        ]
        index.train(train_data, k=2, max_iterations=10, seed=42)
        index.add(10, [1.05, 0.05, 0.0, 0.0])
        index.add(20, [0.05, 1.05, 0.0, 0.0])

        index.save(temp_index_path)

        loaded = vecta.load(temp_index_path)
        assert isinstance(loaded, vecta.IVFIndex)
        assert type(loaded) is vecta.IVFIndex
        assert len(loaded) == 2
        assert loaded.dim() == dim
        assert loaded.is_trained()

        query = [1.0, 0.0, 0.0, 0.0]
        orig_res = index.search(query, k=2, nprobe=2)
        loaded_res = loaded.search(query, k=2, nprobe=2)
        assert orig_res == loaded_res


class TestHnswIndexPersistence:
    """Test save and vecta.load() for HnswIndex."""

    def test_save_and_load_round_trip(self, temp_index_path):
        dim = 4
        index = vecta.HnswIndex(dim=dim, metric="euclidean", m=4, ef_construction=32, ef_search=16)
        vectors = [
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.0, 0.0, 0.0, 1.0]),
        ]
        for vid, v in vectors:
            index.add(vid, v)

        index.save(temp_index_path)

        loaded = vecta.load(temp_index_path)
        assert isinstance(loaded, vecta.HnswIndex)
        assert type(loaded) is vecta.HnswIndex
        assert len(loaded) == 4
        assert loaded.dim() == dim

        query = [0.1, 0.9, 0.0, 0.0]
        orig_res = index.search(query, k=2)
        loaded_res = loaded.search(query, k=2)
        assert orig_res == loaded_res


class TestIVFPQIndexPersistence:
    """Test save and vecta.load() for IVFPQIndex."""

    def test_save_and_load_round_trip(self, temp_index_path):
        dim = 4
        index = vecta.IVFPQIndex(dim=dim, num_clusters=2, m=2, k_per_subvector=4)
        train_data = [
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 1.1, 0.0, 0.0],
        ]
        index.train(train_data, ivf_seed=1, pq_seed=1)
        index.add(100, [1.05, 0.05, 0.0, 0.0])
        index.add(200, [0.05, 1.05, 0.0, 0.0])

        index.save(temp_index_path)

        loaded = vecta.load(temp_index_path)
        assert isinstance(loaded, vecta.IVFPQIndex)
        assert type(loaded) is vecta.IVFPQIndex
        assert len(loaded) == 2
        assert loaded.dim() == dim
        assert loaded.is_trained()

        query = [1.0, 0.0, 0.0, 0.0]
        orig_res = index.search(query, k=2, nprobe=2)
        loaded_res = loaded.search(query, k=2, nprobe=2)
        assert orig_res == loaded_res


class TestGenericLoaderDispatchAndErrors:
    """Test vecta.load() error conditions and generic class dispatch."""

    def test_load_nonexistent_path_raises_io_error(self):
        with pytest.raises(IOError):
            vecta.load("nonexistent_path_12345.vct")

    def test_load_corrupted_garbage_file_raises_error(self, temp_index_path):
        with open(temp_index_path, "wb") as f:
            f.write(b"NOT_A_VALID_VCTA_FILE_GARBAGE")

        with pytest.raises((ValueError, IOError)):
            vecta.load(temp_index_path)

    def test_load_truncated_header_raises_error(self, temp_index_path):
        with open(temp_index_path, "wb") as f:
            f.write(b"VCTA")  # only 4 bytes instead of 22

        with pytest.raises((ValueError, IOError)):
            vecta.load(temp_index_path)
