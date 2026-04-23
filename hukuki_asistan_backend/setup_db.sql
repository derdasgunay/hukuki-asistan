-- =============================================================================
-- setup_db.sql
-- Run this script ONCE as the postgres superuser to prepare the database.
-- Example: psql -U postgres -f setup_db.sql
-- =============================================================================

-- Step 1: Create the application user
-- WARNING: Replace 'YOUR_SECURE_PASSWORD' with your actual password before running this script!
CREATE USER dsg_admin WITH PASSWORD 'YOUR_SECURE_PASSWORD';
-- Step 2: Create the database
CREATE DATABASE hukuki_asistan_db OWNER dsg_admin;

-- Step 3: Connect to the new database to install extensions
\connect hukuki_asistan_db

-- Step 4: Enable pgvector extension (requires pgvector to be installed on the OS)
-- Install on Ubuntu: sudo apt install postgresql-16-pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 5: Grant all privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE hukuki_asistan_db TO dsg_admin;
GRANT ALL ON SCHEMA public TO dsg_admin;

-- =============================================================================
-- NOTE ON INDEXING (run AFTER the ETL pipeline has populated the table):
-- =============================================================================
-- For exact nearest-neighbor search (best for <= 1M rows), no index is needed
-- and pgvector will do a sequential scan, which is accurate and fast enough.
--
-- For large-scale approximate nearest-neighbor (ANN), create an IVFFLAT index:
--   CREATE INDEX ON kararlar
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 32);
--   (Rule of thumb: lists = sqrt(total_rows), so ~32 for 1000 rows)
--
-- B-Tree indexes on categorical columns are created automatically by SQLAlchemy
-- via 'index=True' on the Column definition. They speed up WHERE clauses like:
--   WHERE konu = 'Tazminat'  or  WHERE mahkeme = 'İstanbul ...'
-- =============================================================================
