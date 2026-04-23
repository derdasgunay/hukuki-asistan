"""
app.py
======
Flask API for Hukuki Asistan.

Changes from the legacy version:
  - FAISS index replaced with PostgreSQL + pgvector (via search_engine.py).
  - Corpus embeddings are NO LONGER loaded into RAM. Only the BM25 index and
    the AI model are kept in memory.
  - BM25 corpus is loaded from the database at startup (not from kararlar.json).
  - The XAI sentence-highlighting function (xai_cumle_bul) is unchanged.
"""

import re

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from database.db import SessionLocal
from database.models import Karar
from search_engine import search_by_filters_only, search_hybrid

app = Flask(__name__)
CORS(app)

print("Sistem Başlatılıyor... Lütfen bekleyin.")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: Load AI Model
# ─────────────────────────────────────────────────────────────────────────────
# Detect GPU; BERTurk-Legal benefits significantly from CUDA acceleration.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Model '{device}' cihazına yükleniyor...")
model = SentenceTransformer("./BERTurk-Legal_FULL_seed42_ep2_msl192", device=device)
print("  Model hazır.")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: Build BM25 Index from Database
# ─────────────────────────────────────────────────────────────────────────────
# We load only the id, olay_ozeti, and hukum fields — NOT the embeddings.
# This keeps memory usage minimal compared to the old FAISS setup.
print("  BM25 indeksi için veriler veritabanından yükleniyor...")

startup_session = SessionLocal()
try:
    # SQLAlchemy 2.0 style: use select() to fetch only the columns we need
    bm25_statement = select(Karar.id, Karar.olay_ozeti, Karar.hukum)
    bm25_rows = startup_session.execute(bm25_statement).all()
finally:
    startup_session.close()

# bm25_id_list and corpus_texts must stay in the same order for score lookup
bm25_id_list: list[str] = [row.id for row in bm25_rows]
corpus_texts: list[str] = [
    (row.olay_ozeti or "") + " " + (row.hukum or "")
    for row in bm25_rows
]
tokenized_corpus: list[list[str]] = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print(f"  BM25 indeksi {len(bm25_id_list)} kayıt üzerinde oluşturuldu.")
print("Sistem Hazır! Frontend'den arama yapabilirsin.")


# ─────────────────────────────────────────────────────────────────────────────
# XAI: Sentence-Level Explanation (unchanged from legacy)
# ─────────────────────────────────────────────────────────────────────────────

def xai_cumle_bul(
    query_emb: np.ndarray,
    text: str,
    sentence_model: SentenceTransformer,
    top_k: int = 2,
) -> list[str]:
    """
    Identifies the top_k sentences in 'text' most similar to the query embedding.
    Called only for the top 10 results — never for the entire corpus.
    """
    # Split on sentence-ending punctuation or newlines
    cumleler = re.split(r"(?<=[.!?;\n])\s+", text)
    # Filter out very short fragments that aren't real sentences
    cumleler = [c.strip() for c in cumleler if len(c.strip()) > 20]

    if not cumleler:
        return ["Açıklayıcı metin bulunamadı."]

    cumle_embs = sentence_model.encode(cumleler)

    query_emb_flat = query_emb.flatten()
    skorlar = []
    for emb in cumle_embs:
        norm_product = np.linalg.norm(query_emb_flat) * np.linalg.norm(emb)
        if norm_product == 0:
            skorlar.append(0.0)
        else:
            skor = np.dot(query_emb_flat, emb) / norm_product
            skorlar.append(skor)

    # Get indices of the top_k highest-scoring sentences
    en_iyi_idx = np.argsort(skorlar)[-top_k:][::-1]

    sonuclar = []
    for i in en_iyi_idx:
        secilen_cumle = cumleler[i]
        kelimeler = secilen_cumle.split()
        # UI protection: truncate sentences longer than 35 words
        if len(kelimeler) > 35:
            secilen_cumle = " ".join(kelimeler[:35]) + "..."
        sonuclar.append(secilen_cumle)

    return sonuclar


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /api/search
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["GET"])
def search():
    query: str = request.args.get("q", "")
    konu_filter: str = request.args.get("konu", "")
    mahkeme_filter: str = request.args.get("mahkeme", "")

    db_session = SessionLocal()
    try:
        # ── Scenario 1: No query — filter-only search ─────────────────────────
        if not query:
            results = search_by_filters_only(
                session=db_session,
                konu_filter=konu_filter,
                mahkeme_filter=mahkeme_filter,
                limit=10,
            )
            return jsonify(results)

        # ── Scenario 2: Full hybrid search ────────────────────────────────────

        # Encode the query into a vector using BERTurk
        query_embedding: np.ndarray = model.encode([query])
        # model.encode() returns shape [1, 768]; we need shape [768] for search
        query_embedding_flat: np.ndarray = query_embedding[0]

        # Run the hybrid search (pgvector + BM25 fusion)
        search_results: list[dict] = search_hybrid(
            query=query,
            query_embedding=query_embedding_flat,
            session=db_session,
            bm25_index=bm25,
            bm25_id_list=bm25_id_list,
            konu_filter=konu_filter,
            mahkeme_filter=mahkeme_filter,
            alpha=0.5,
            candidate_pool_size=100,
            top_k=10,
        )

        # ── XAI: Add sentence highlights to each of the top 10 results ────────
        # XAI is intentionally calculated ONLY for the final top 10, never for
        # the full corpus. This keeps response time fast.
        for result_dict in search_results:
            hedef_metin = (result_dict.get("gerekce") or "") + " " + (result_dict.get("tam_olay") or "")
            xai_highlights = xai_cumle_bul(
                query_emb=query_embedding,  # Shape [1, 768] is fine here
                text=hedef_metin,
                sentence_model=model,
            )
            result_dict["xai_vurgular"] = xai_highlights

        return jsonify(search_results)

    finally:
        # Always close the session, even if an exception was raised
        db_session.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
