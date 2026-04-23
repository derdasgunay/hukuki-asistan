"""
database/db.py
==============
Creates the SQLAlchemy engine and session factory.

This module is the single source of truth for the database connection.
Both the ETL pipeline and the Flask app import from here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ─────────────────────────────────────────────────────────────────────────────
# Connection URI — format: postgresql://user:password@host/database_name
# ─────────────────────────────────────────────────────────────────────────────

# This file lives at  hukuki_asistan_backend/database/db.py
# The .env file lives at hukuki_asistan_backend/.env  (one level up)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

DATABASE_URI = os.environ.get("DATABASE_URI")

if not DATABASE_URI:
    raise ValueError("ERROR: DATABASE_URI is not set in the .env file!")

# 'echo=False' means SQLAlchemy will NOT print every SQL statement to the
# console. Set to True temporarily for debugging query issues.
engine = create_engine(DATABASE_URI, echo=False)

# SessionLocal is a factory. Calling SessionLocal() creates a new DB session.
# expire_on_commit=False keeps objects readable after a session.commit().
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db_session() -> Session:
    """
    Creates and returns a new database session.

    The caller is responsible for calling session.close() when done,
    or using the session as a context manager:

        with get_db_session() as session:
            ...
    """
    return SessionLocal()
