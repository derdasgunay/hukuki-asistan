"""
search_engine.py
================
Hybrid search combining pgvector (semantic) and BM25 (lexical) scoring.

Architecture:
  1. pgvector query  → Retrieves a 'candidate pool' of the most semantically
                       similar records from PostgreSQL, applying any metadata
                       filters directly in SQL (WHERE clause).
  2. BM25 scoring   → Looks up pre-computed BM25 scores for those candidates.
  3. Score fusion   → Combines both scores with a tunable alpha weight.
  4. Top-K return   → Returns the top_k results as plain dicts for the Flask API.

Why separate pgvector and BM25 instead of one SQL query?
  PostgreSQL's built-in full-text search (tsvector/tsquery) uses TF-IDF, not
  BM25. Keeping BM25 in Python (via rank_bm25) is more accurate for Turkish
  legal text and consistent with the existing system's ranking behavior.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Karar


# ─────────────────────────────────────────────────────────────────────────────
# SCORE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(scores: list[float]) -> list[float]:
    """
    Applies Min-Max normalization so all scores fall in the [0.0, 1.0] range.
    This makes BM25 and pgvector scores comparable before combining them.

    If all scores are identical (min == max), returns 0.5 for all to avoid
    a division-by-zero error.
    """
    if not scores:
        return []

    min_val = min(scores)
    max_val = max(scores)

    if max_val == min_val:
        # All scores are equal — no meaningful ranking is possible
        return [0.5 for _ in scores]

    normalized = [(s - min_val) / (max_val - min_val) for s in scores]
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# PGVECTOR CANDIDATE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def get_semantic_candidates(
    session: Session,
    query_embedding: np.ndarray,
    konu_filter: str = "",
    mahkeme_filter: str = "",
    candidate_pool_size: int = 100,
) -> list[tuple[Karar, float]]:
    """
    Queries the database for the most semantically similar records using
    pgvector's cosine distance operator (<=>) and optional metadata filters.

    Returns a list of (Karar object, cosine_distance) tuples.
    Lower distance = more similar. Distance of 0.0 = identical vectors.

    Args:
        session:            An active SQLAlchemy session.
        query_embedding:    The query's vector as a numpy array (shape: [768]).
        konu_filter:        If provided, adds WHERE konu = konu_filter to SQL.
        mahkeme_filter:     If provided, adds WHERE mahkeme = mahkeme_filter.
        candidate_pool_size: How many candidates to fetch before BM25 re-ranking.
                             Fetching more candidates improves recall at the cost
                             of slightly more BM25 work. 100 is a good default.
    """
    # Convert numpy array to a Python list; pgvector requires a plain list
    query_vector_as_list: list[float] = query_embedding.flatten().tolist()

    # Compute the cosine distance as a labeled column so we can retrieve it
    cosine_distance_column = Karar.embedding.cosine_distance(query_vector_as_list)

    # SQLAlchemy 2.0 style: build the SELECT statement with select()
    statement = (
        select(Karar, cosine_distance_column.label("cosine_distance"))
        .order_by(cosine_distance_column)  # Ascending: closest first
        .limit(candidate_pool_size)
    )

    # ── Apply optional metadata filters ───────────────────────────────────────
    # Each filter is appended as a WHERE clause only if the user provided it.
    # Using ilike() for case-insensitive matching on Turkish text.
    if konu_filter and konu_filter.strip():
        statement = statement.where(Karar.konu.ilike(f"%{konu_filter.strip()}%"))

    if mahkeme_filter and mahkeme_filter.strip():
        statement = statement.where(Karar.mahkeme.ilike(f"%{mahkeme_filter.strip()}%"))

    # Execute the query and collect results
    db_results = session.execute(statement).all()

    # Each row is a Row(Karar, cosine_distance). Unpack into a clean list.
    candidates: list[tuple[Karar, float]] = [
        (row.Karar, float(row.cosine_distance)) for row in db_results
    ]

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID SCORE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_scores(
    candidates: list[tuple[Karar, float]],
    bm25_index: BM25Okapi,
    bm25_id_list: list[str],
    tokenized_query: list[str],
    alpha: float = 0.5,
) -> list[tuple[float, Karar]]:
    """
    Combines pgvector cosine similarity with BM25 lexical scores for the
    candidate records, then returns them sorted by final hybrid score.

    The formula is:
        hybrid_score = (alpha * bm25_norm) + ((1 - alpha) * semantic_norm)

    Where:
        alpha = 0.5  → Equal weight to both signals (recommended default)
        alpha = 0.0  → Pure semantic search (pgvector only)
        alpha = 1.0  → Pure lexical search (BM25 only)

    Args:
        candidates:      List of (Karar, cosine_distance) from pgvector query.
        bm25_index:      The in-memory BM25Okapi index over the full corpus.
        bm25_id_list:    List of karar IDs in the same order the BM25 index was built.
        tokenized_query: The query split into lowercase tokens for BM25 scoring.
        alpha:           Weight for the BM25 score component.
    """
    if not candidates:
        return []

    # ── Convert cosine DISTANCE to cosine SIMILARITY ──────────────────────────
    # pgvector returns distance (lower = better). We flip it to similarity
    # (higher = better) so it can be compared and combined with BM25 scores.
    cosine_distances = [distance for (_, distance) in candidates]
    cosine_similarities = [1.0 - d for d in cosine_distances]
    normalized_semantic_scores = normalize_scores(cosine_similarities)

    # ── Get BM25 scores for the full corpus ───────────────────────────────────
    # BM25 is computed for all documents; we then pick out the scores for
    # only the candidates returned by pgvector.
    all_bm25_scores: np.ndarray = bm25_index.get_scores(tokenized_query)

    # Build a lookup dict: karar_id -> bm25_score for fast access
    bm25_score_lookup: dict[str, float] = {
        karar_id: float(all_bm25_scores[index])
        for index, karar_id in enumerate(bm25_id_list)
    }

    # Extract BM25 scores only for our candidates (in the same order)
    candidate_bm25_scores: list[float] = [
        bm25_score_lookup.get(karar.id, 0.0)
        for (karar, _) in candidates
    ]
    normalized_bm25_scores = normalize_scores(candidate_bm25_scores)

    # ── Fuse scores ───────────────────────────────────────────────────────────
    scored_candidates: list[tuple[float, Karar]] = []

    for i, (karar, _) in enumerate(candidates):
        bm25_component = normalized_bm25_scores[i]
        semantic_component = normalized_semantic_scores[i]

        hybrid_score = (alpha * bm25_component) + ((1 - alpha) * semantic_component)
        scored_candidates.append((hybrid_score, karar))

    # Sort descending: highest hybrid score first
    scored_candidates.sort(key=lambda pair: pair[0], reverse=True)

    return scored_candidates


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEARCH FUNCTION (called by Flask routes)
# ─────────────────────────────────────────────────────────────────────────────

def search_hybrid(
    query: str,
    query_embedding: np.ndarray,
    session: Session,
    bm25_index: BM25Okapi,
    bm25_id_list: list[str],
    konu_filter: str = "",
    mahkeme_filter: str = "",
    alpha: float = 0.5,
    candidate_pool_size: int = 100,
    top_k: int = 10,
) -> list[dict]:
    """
    Main entry point for the hybrid search pipeline.

    Orchestrates semantic retrieval from pgvector, BM25 re-ranking,
    and returns the top_k results as a list of plain dicts.

    Args:
        query:              The raw user query string (used for BM25 tokenization).
        query_embedding:    The query encoded as a vector (numpy array, shape [768]).
        session:            Active SQLAlchemy database session.
        bm25_index:         Pre-built BM25Okapi index (built at app startup from DB).
        bm25_id_list:       Karar IDs in the order the BM25 index was built.
        konu_filter:        Optional subject/category filter (e.g. "Tazminat").
        mahkeme_filter:     Optional court name filter.
        alpha:              BM25 weight [0.0–1.0]. 0.5 = balanced hybrid.
        candidate_pool_size: Number of pgvector candidates to fetch before re-ranking.
        top_k:              Number of results to return.

    Returns:
        A list of dicts, each representing a karar with its hybrid score.
    """
    # ── Step 1: Retrieve semantic candidates from PostgreSQL ──────────────────
    candidates = get_semantic_candidates(
        session=session,
        query_embedding=query_embedding,
        konu_filter=konu_filter,
        mahkeme_filter=mahkeme_filter,
        candidate_pool_size=candidate_pool_size,
    )

    if not candidates:
        return []

    # ── Step 2: Tokenize the query for BM25 ───────────────────────────────────
    tokenized_query = query.lower().split()

    # ── Step 3: Fuse pgvector and BM25 scores ─────────────────────────────────
    ranked_candidates = build_hybrid_scores(
        candidates=candidates,
        bm25_index=bm25_index,
        bm25_id_list=bm25_id_list,
        tokenized_query=tokenized_query,
        alpha=alpha,
    )

    # ── Step 4: Convert top_k results to plain dicts for JSON serialization ───
    results: list[dict] = []

    for hybrid_score, karar in ranked_candidates[:top_k]:
        result_dict = {
            "id": karar.id,
            "esas_no": karar.esas_no,
            "karar_no": karar.karar_no,
            "mahkeme": karar.mahkeme,
            "konu": karar.konu,
            # Convert date objects to ISO strings for JSON compatibility
            "dava_tarihi": karar.dava_tarihi.isoformat() if karar.dava_tarihi else None,
            "karar_tarihi": karar.karar_tarihi.isoformat() if karar.karar_tarihi else None,
            "olay_ozeti": karar.olay_ozeti,
            "tam_olay": karar.tam_olay,
            "gerekce": karar.gerekce,
            "hukum": karar.hukum,
            "summary_for_human": karar.summary_for_human,
            "summary_for_model": karar.summary_for_model,
            "rrl_facts": karar.rrl_facts,
            "rrl_reasoning": karar.rrl_reasoning,
            "rrl_verdict": karar.rrl_verdict,
            "mentioned_laws": karar.mentioned_laws,
            "hibrit_skor": round(hybrid_score, 4),
        }
        results.append(result_dict)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FILTER-ONLY SEARCH (no query, just metadata filtering)
# ─────────────────────────────────────────────────────────────────────────────

def search_by_filters_only(
    session: Session,
    konu_filter: str = "",
    mahkeme_filter: str = "",
    limit: int = 10,
) -> list[dict]:
    """
    Returns records matching the metadata filters when no text query is provided.
    Uses a simple SQL SELECT with WHERE clauses (no vector math needed).
    """
    statement = select(Karar).limit(limit)

    if konu_filter and konu_filter.strip():
        statement = statement.where(Karar.konu.ilike(f"%{konu_filter.strip()}%"))

    if mahkeme_filter and mahkeme_filter.strip():
        statement = statement.where(Karar.mahkeme.ilike(f"%{mahkeme_filter.strip()}%"))

    db_results = session.execute(statement).scalars().all()

    results: list[dict] = []
    for karar in db_results:
        result_dict = {
            "id": karar.id,
            "esas_no": karar.esas_no,
            "karar_no": karar.karar_no,
            "mahkeme": karar.mahkeme,
            "konu": karar.konu,
            "dava_tarihi": karar.dava_tarihi.isoformat() if karar.dava_tarihi else None,
            "karar_tarihi": karar.karar_tarihi.isoformat() if karar.karar_tarihi else None,
            "olay_ozeti": karar.olay_ozeti,
            "tam_olay": karar.tam_olay,
            "gerekce": karar.gerekce,
            "hukum": karar.hukum,
            "summary_for_human": karar.summary_for_human,
            "summary_for_model": karar.summary_for_model,
            "rrl_facts": karar.rrl_facts,
            "rrl_reasoning": karar.rrl_reasoning,
            "rrl_verdict": karar.rrl_verdict,
            "mentioned_laws": karar.mentioned_laws,
        }
        results.append(result_dict)

    return results
