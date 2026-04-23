# Hukuki Asistan

NLP-based legal case retrieval system with sentence-level XAI highlighting.

## Prerequisites

- PostgreSQL with `pgvector` extension
- Python 3.10+
- Node.js (LTS)

## Quick Start

### 1. Database

```bash
psql -U postgres -f setup_db.sql
```

Ensure the `pgvector` extension is enabled in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Environment

```bash
cp hukuki_asistan_backend/.env.example hukuki_asistan_backend/.env
```

Edit `.env` and set your `DATABASE_URI`:

```
DATABASE_URI=postgresql://user:password@localhost:5432/hukuki_asistan
```

### 3. Backend

```bash
cd hukuki_asistan_backend
python -m venv venv
source venv/bin/activate
pip install -r requirements_pgvector.txt
python app.py
```

### 4. Frontend

```bash
cd hukuki_asistan_frontend
npm install
npm run dev
```

App runs at `http://localhost:5173` by default.
