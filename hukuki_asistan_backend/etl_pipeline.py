"""
etl_pipeline.py
===============
Migrates legal cases from 'kararlar.json' into the PostgreSQL database.

ETL Flow:
  Extract  → Load raw JSON from disk
  Transform → Parse dates, compute embeddings, build Karar objects
  Load      → Insert records in batches of BATCH_SIZE, skip duplicates

Fault Tolerance:
  - A failed record is logged to 'etl_errors.log' and skipped.
  - The process continues to the next record; no data is lost.
  - At the end, a summary shows how many records succeeded vs. failed.

Run this script from the hukuki_asistan_backend/ directory:
  python etl_pipeline.py
"""

import json
import logging
import sys
from datetime import date, datetime
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.db import SessionLocal, engine
from database.models import Base, Karar

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
JSON_FILE_PATH: str = "kararlar.json"
MODEL_PATH: str = "./BERTurk-Legal_FULL_seed42_ep2_msl192"

# Number of records inserted per database commit.
# Larger batches are faster but use more memory. 100 is a safe default.
BATCH_SIZE: int = 100

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Log INFO and above to the console so the user sees real-time progress.
# Log ERROR and above to a file so failed records are preserved for review.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etl_errors.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_date_string(date_str: str) -> Optional[date]:
    """
    Converts a date string in 'DD/MM/YYYY' format to a Python date object.
    Returns None if the string is empty, None, or in an unexpected format.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d/%m/%Y").date()
    except (ValueError, TypeError):
        # Do not crash on malformed dates; just store NULL in the DB
        logger.warning(f"Could not parse date string: '{date_str}'. Storing NULL.")
        return None


def build_search_text(record: dict) -> str:
    """
    Combines the most semantically rich fields into a single string
    for embedding generation. This mirrors the corpus used by the old FAISS index.
    """
    olay_ozeti = record.get("olay_ozeti", "") or ""
    hukum = record.get("hukum", "") or ""
    return olay_ozeti + " " + hukum


def transform_record(raw_record: dict, embedding_vector: np.ndarray) -> Karar:
    """
    Takes one raw JSON record and its computed embedding, and constructs
    a Karar ORM object ready to be added to the database session.
    """
    meta = raw_record.get("meta_data", {})
    rrl = raw_record.get("rrl_segments", {})
    structural = raw_record.get("structural_features", {})

    karar = Karar(
        # ── Identifiers ────────────────────────────────────────────────────
        id=raw_record.get("id"),
        esas_no=raw_record.get("esas_no") or meta.get("esas_no"),
        karar_no=meta.get("karar_no"),

        # ── Indexed Metadata ───────────────────────────────────────────────
        mahkeme=raw_record.get("mahkeme") or meta.get("court_name"),
        konu=raw_record.get("konu") or meta.get("case_subject"),

        # ── Parsed Dates ───────────────────────────────────────────────────
        dava_tarihi=parse_date_string(meta.get("dava_tarihi", "")),
        karar_tarihi=parse_date_string(meta.get("karar_tarihi", "")),

        # ── Full Text ──────────────────────────────────────────────────────
        olay_ozeti=raw_record.get("olay_ozeti"),
        tam_olay=raw_record.get("tam_olay"),
        gerekce=raw_record.get("gerekce"),
        hukum=raw_record.get("hukum"),

        # ── Summaries ──────────────────────────────────────────────────────
        summary_for_human=raw_record.get("summary_for_human"),
        summary_for_model=raw_record.get("summary_for_model"),

        # ── RRL Segments ───────────────────────────────────────────────────
        rrl_facts=rrl.get("facts_text"),
        rrl_reasoning=rrl.get("reasoning_text"),
        rrl_verdict=rrl.get("verdict_text"),

        # ── Structural Features ────────────────────────────────────────────
        mentioned_laws=structural.get("mentioned_laws"),

        # ── Vector Embedding ───────────────────────────────────────────────
        # pgvector expects a plain Python list, not a numpy array.
        embedding=embedding_vector.tolist(),
    )
    return karar


def get_existing_ids(session: Session) -> set[str]:
    """
    Queries the database for all existing record IDs.
    Used to skip records that were already inserted (idempotent runs).
    """
    # SQLAlchemy 2.0 style: use select() instead of session.query()
    statement = select(Karar.id)
    result = session.execute(statement)
    # result.scalars() returns the single column values as an iterable
    existing_id_set = set(result.scalars().all())
    logger.info(f"Found {len(existing_id_set)} existing records in the database.")
    return existing_id_set


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ETL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_etl() -> None:
    """
    Orchestrates the full Extract-Transform-Load process.
    """

    # ── Step 1: Create tables if they don't exist ─────────────────────────────
    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables are ready.")

    # ── Step 2: Load the source JSON ──────────────────────────────────────────
    logger.info(f"Loading source data from '{JSON_FILE_PATH}'...")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as json_file:
        all_records: list[dict] = json.load(json_file)
    total_records = len(all_records)
    logger.info(f"Loaded {total_records} records from JSON.")

    # ── Step 3: Load the embedding model ─────────────────────────────────────
    # Detect GPU availability. On your Ubuntu machine with an NVIDIA card,
    # this will use CUDA for much faster embedding generation.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SentenceTransformer model on device: '{device}'...")
    embedding_model = SentenceTransformer(MODEL_PATH, device=device)
    logger.info("Model loaded.")

    # ── Step 4: Determine which records still need to be inserted ─────────────
    session = SessionLocal()
    try:
        existing_ids = get_existing_ids(session)
    finally:
        session.close()

    # Filter out records that are already in the database
    new_records = [r for r in all_records if r.get("id") not in existing_ids]
    logger.info(f"{len(new_records)} new records to insert (skipping {total_records - len(new_records)} duplicates).")

    if not new_records:
        logger.info("Nothing to do. Database is already up to date.")
        return

    # ── Step 5: Process in batches ────────────────────────────────────────────
    success_count = 0
    failure_count = 0

    # We iterate over the new records in chunks of BATCH_SIZE
    for batch_start in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[batch_start : batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(new_records))
        logger.info(f"Processing batch: records {batch_start + 1} to {batch_end}...")

        # Build the text corpus for this batch (for embedding generation)
        batch_texts = [build_search_text(record) for record in batch]

        # Compute all embeddings for this batch in a single model.encode() call.
        # This is much faster than calling encode() one record at a time.
        try:
            batch_embeddings = embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as embedding_error:
            # If the entire batch fails at embedding time, log and skip it
            logger.error(
                f"CRITICAL: Embedding generation failed for batch starting at "
                f"index {batch_start}. Error: {embedding_error}"
            )
            failure_count += len(batch)
            continue

        # ── Insert each record in the batch individually ───────────────────
        # We use a separate session per batch so that a failure in one batch
        # does not roll back the other batches that already succeeded.
        batch_session = SessionLocal()
        try:
            for i, raw_record in enumerate(batch):
                record_id = raw_record.get("id", f"unknown_index_{batch_start + i}")
                try:
                    karar_object = transform_record(raw_record, batch_embeddings[i])
                    batch_session.add(karar_object)
                    success_count += 1

                except Exception as record_error:
                    # Log the failure with enough detail to debug later,
                    # but DO NOT stop the loop — continue to the next record.
                    logger.error(
                        f"FAILED to process record id='{record_id}'. "
                        f"Error: {record_error}"
                    )
                    failure_count += 1

            # Commit all successfully transformed records in this batch at once
            batch_session.commit()
            logger.info(f"Batch committed successfully.")

        except Exception as commit_error:
            batch_session.rollback()
            logger.error(
                f"FAILED to commit batch starting at index {batch_start}. "
                f"Rolling back. Error: {commit_error}"
            )
            failure_count += len(batch)
            success_count -= len(batch)  # Undo the optimistic count from the loop

        finally:
            batch_session.close()

    # ── Step 6: Final Summary ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ETL PIPELINE COMPLETE")
    logger.info(f"  Successfully inserted : {success_count} records")
    logger.info(f"  Failed (logged)       : {failure_count} records")
    logger.info(f"  Check 'etl_errors.log' for details on any failures.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_etl()
