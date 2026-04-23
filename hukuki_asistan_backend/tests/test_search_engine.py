"""
tests/test_search_engine.py
============================
Unit tests for the pure-logic functions in search_engine.py and etl_pipeline.py.

What is tested here (no database or model connection required):
  - normalize_scores: edge cases and normal cases
  - build_hybrid_scores: score fusion logic with mock Karar objects
  - search_hybrid: end-to-end flow with a mocked database session
  - parse_date_string: valid, empty, and malformed date strings

Run from the hukuki_asistan_backend/ directory:
  pytest tests/

pytest-mock (pip install pytest-mock) is used to replace the database session
with a fake in-memory object, so tests run without any PostgreSQL connection.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

# ── Modules under test ────────────────────────────────────────────────────────
from etl_pipeline import parse_date_string
from search_engine import (
    build_hybrid_scores,
    normalize_scores,
    search_hybrid,
)


# =============================================================================
# FIXTURES: Reusable test data
# =============================================================================

@pytest.fixture
def sample_karar_objects() -> list:
    """
    Creates lightweight mock Karar objects without a real database.
    MagicMock lets us set any attribute we want on a fake object.
    """
    karar_a = MagicMock()
    karar_a.id = "karar_001"
    karar_a.mahkeme = "İstanbul Bölge Adliye Mahkemesi"
    karar_a.konu = "Tazminat"
    karar_a.olay_ozeti = "İşçi işten çıkarılmasına itiraz etti."
    karar_a.hukum = "Tazminata hükmedildi."
    karar_a.esas_no = "2020/100"
    karar_a.karar_no = "2021/200"
    karar_a.dava_tarihi = date(2020, 1, 15)
    karar_a.karar_tarihi = date(2021, 3, 10)
    karar_a.tam_olay = ""
    karar_a.gerekce = ""
    karar_a.summary_for_human = ""
    karar_a.summary_for_model = ""
    karar_a.rrl_facts = ""
    karar_a.rrl_reasoning = ""
    karar_a.rrl_verdict = ""
    karar_a.mentioned_laws = []

    karar_b = MagicMock()
    karar_b.id = "karar_002"
    karar_b.mahkeme = "Ankara Bölge Adliye Mahkemesi"
    karar_b.konu = "Sözleşme İhlali"
    karar_b.olay_ozeti = "Taraflar arasında sözleşme uyuşmazlığı yaşandı."
    karar_b.hukum = "Dava reddedildi."
    karar_b.esas_no = "2019/500"
    karar_b.karar_no = "2020/900"
    karar_b.dava_tarihi = date(2019, 5, 20)
    karar_b.karar_tarihi = date(2020, 7, 15)
    karar_b.tam_olay = ""
    karar_b.gerekce = ""
    karar_b.summary_for_human = ""
    karar_b.summary_for_model = ""
    karar_b.rrl_facts = ""
    karar_b.rrl_reasoning = ""
    karar_b.rrl_verdict = ""
    karar_b.mentioned_laws = []

    return [karar_a, karar_b]


@pytest.fixture
def sample_bm25_index(sample_karar_objects) -> tuple[BM25Okapi, list[str]]:
    """
    Builds a small BM25 index from the sample karar objects.
    Returns the index and the id list in the same order.
    """
    id_list = [k.id for k in sample_karar_objects]
    corpus_texts = [
        (k.olay_ozeti or "") + " " + (k.hukum or "")
        for k in sample_karar_objects
    ]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, id_list


# =============================================================================
# TESTS: normalize_scores
# =============================================================================

class TestNormalizeScores:

    def test_normal_case_returns_values_between_zero_and_one(self):
        """Normalized scores must always fall within [0.0, 1.0]."""
        scores = [10.0, 20.0, 5.0, 15.0]
        normalized = normalize_scores(scores)

        assert len(normalized) == 4
        for value in normalized:
            assert 0.0 <= value <= 1.0

    def test_minimum_score_maps_to_zero(self):
        """The lowest input score must map to exactly 0.0."""
        scores = [1.0, 5.0, 10.0]
        normalized = normalize_scores(scores)
        assert normalized[0] == pytest.approx(0.0)

    def test_maximum_score_maps_to_one(self):
        """The highest input score must map to exactly 1.0."""
        scores = [1.0, 5.0, 10.0]
        normalized = normalize_scores(scores)
        assert normalized[2] == pytest.approx(1.0)

    def test_all_equal_scores_returns_half(self):
        """When all scores are the same, return 0.5 to avoid division by zero."""
        scores = [7.0, 7.0, 7.0]
        normalized = normalize_scores(scores)
        assert all(v == pytest.approx(0.5) for v in normalized)

    def test_empty_input_returns_empty_list(self):
        """An empty input must return an empty list without raising an error."""
        normalized = normalize_scores([])
        assert normalized == []

    def test_single_element_returns_half(self):
        """A single element has min == max, so it should return [0.5]."""
        normalized = normalize_scores([42.0])
        assert normalized == [pytest.approx(0.5)]


# =============================================================================
# TESTS: parse_date_string (from etl_pipeline.py)
# =============================================================================

class TestParseDateString:

    def test_valid_date_string_is_parsed_correctly(self):
        """A well-formed DD/MM/YYYY string must return the correct date object."""
        result = parse_date_string("27/02/2018")
        assert result == date(2018, 2, 27)

    def test_empty_string_returns_none(self):
        """An empty string must return None, not raise an exception."""
        result = parse_date_string("")
        assert result is None

    def test_none_input_returns_none(self):
        """None input must return None."""
        result = parse_date_string(None)
        assert result is None

    def test_wrong_format_returns_none(self):
        """A date string in a different format must return None gracefully."""
        result = parse_date_string("2018-02-27")  # ISO format, not DD/MM/YYYY
        assert result is None

    def test_completely_invalid_string_returns_none(self):
        """Garbage input must return None without crashing."""
        result = parse_date_string("not a date at all")
        assert result is None


# =============================================================================
# TESTS: build_hybrid_scores
# =============================================================================

class TestBuildHybridScores:

    def test_returns_a_list_of_score_karar_tuples(self, sample_karar_objects, sample_bm25_index):
        """build_hybrid_scores must return a list of (float, Karar) tuples."""
        bm25_index, id_list = sample_bm25_index
        # Cosine distance values (0.0 = identical, 2.0 = opposite)
        candidates = [(sample_karar_objects[0], 0.1), (sample_karar_objects[1], 0.5)]
        tokenized_query = ["işçi", "tazminat"]

        scored = build_hybrid_scores(
            candidates=candidates,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            tokenized_query=tokenized_query,
            alpha=0.5,
        )

        assert len(scored) == 2
        for score, karar in scored:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_results_are_sorted_descending_by_score(self, sample_karar_objects, sample_bm25_index):
        """The result list must be sorted from highest to lowest score."""
        bm25_index, id_list = sample_bm25_index
        candidates = [(sample_karar_objects[0], 0.1), (sample_karar_objects[1], 0.5)]
        tokenized_query = ["tazminat"]

        scored = build_hybrid_scores(
            candidates=candidates,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            tokenized_query=tokenized_query,
            alpha=0.5,
        )

        scores_only = [score for score, _ in scored]
        # Check that each score is >= the next score
        assert all(scores_only[i] >= scores_only[i + 1] for i in range(len(scores_only) - 1))

    def test_empty_candidates_returns_empty_list(self, sample_bm25_index):
        """No candidates must result in an empty list without errors."""
        bm25_index, id_list = sample_bm25_index
        scored = build_hybrid_scores(
            candidates=[],
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            tokenized_query=["test"],
            alpha=0.5,
        )
        assert scored == []

    def test_pure_semantic_alpha_zero_uses_only_pgvector(self, sample_karar_objects, sample_bm25_index):
        """With alpha=0.0, BM25 contributes nothing; the ranking should be driven
        entirely by pgvector scores (lower distance = higher rank)."""
        bm25_index, id_list = sample_bm25_index
        # karar_001 is much closer (distance 0.05) than karar_002 (distance 0.9)
        candidates = [(sample_karar_objects[0], 0.05), (sample_karar_objects[1], 0.9)]
        tokenized_query = ["irrelevant", "tokens"]

        scored = build_hybrid_scores(
            candidates=candidates,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            tokenized_query=tokenized_query,
            alpha=0.0,  # Pure semantic
        )

        # karar_001 (closer to query) must rank first
        assert scored[0][1].id == "karar_001"


# =============================================================================
# TESTS: search_hybrid (with mocked database session)
# =============================================================================

class TestSearchHybrid:

    def test_returns_list_of_dicts_with_expected_keys(
        self, sample_karar_objects, sample_bm25_index
    ):
        """
        search_hybrid must return plain dicts with all required fields
        including the 'hibrit_skor' field.
        """
        bm25_index, id_list = sample_bm25_index

        # Mock the DB session: session.execute().all() returns fake Row objects
        mock_row_a = MagicMock()
        mock_row_a.Karar = sample_karar_objects[0]
        mock_row_a.cosine_distance = 0.15

        mock_row_b = MagicMock()
        mock_row_b.Karar = sample_karar_objects[1]
        mock_row_b.cosine_distance = 0.45

        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = [mock_row_a, mock_row_b]

        query_embedding = np.random.rand(768).astype(np.float32)

        results = search_hybrid(
            query="tazminat davası",
            query_embedding=query_embedding,
            session=mock_session,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            top_k=10,
        )

        assert isinstance(results, list)
        assert len(results) == 2

        # Every result must be a dict with these keys
        required_keys = {"id", "mahkeme", "konu", "olay_ozeti", "hibrit_skor"}
        for result_dict in results:
            assert required_keys.issubset(result_dict.keys()), (
                f"Missing keys in result: {required_keys - result_dict.keys()}"
            )

    def test_returns_empty_list_when_no_candidates_found(
        self, sample_bm25_index
    ):
        """When pgvector returns no results, search_hybrid must return []."""
        bm25_index, id_list = sample_bm25_index

        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = []

        query_embedding = np.random.rand(768).astype(np.float32)

        results = search_hybrid(
            query="boş sorgu",
            query_embedding=query_embedding,
            session=mock_session,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
        )

        assert results == []

    def test_top_k_limits_number_of_results(
        self, sample_karar_objects, sample_bm25_index
    ):
        """top_k=1 must return at most 1 result even if 2 candidates exist."""
        bm25_index, id_list = sample_bm25_index

        mock_row_a = MagicMock()
        mock_row_a.Karar = sample_karar_objects[0]
        mock_row_a.cosine_distance = 0.1

        mock_row_b = MagicMock()
        mock_row_b.Karar = sample_karar_objects[1]
        mock_row_b.cosine_distance = 0.3

        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = [mock_row_a, mock_row_b]

        query_embedding = np.random.rand(768).astype(np.float32)

        results = search_hybrid(
            query="test",
            query_embedding=query_embedding,
            session=mock_session,
            bm25_index=bm25_index,
            bm25_id_list=id_list,
            top_k=1,
        )

        assert len(results) == 1
