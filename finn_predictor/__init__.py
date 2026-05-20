"""Finn-Predictor — sentiment-driven crude market predictor on top of Finnhub.

Layered architecture:
    - storage:    SQLAlchemy ORM + repository helpers (SQLite by default)
    - ingestion:  Finnhub client wrapper, news/price ingestion, APScheduler jobs
    - sentiment:  pluggable Scorer interface (VADER now, FinBERT swap-in)
    - predictor:  whole-market (iter 1) and per-sector (iter 2) forecasters
    - ui:         Streamlit dashboard

See progress.md at the repo root for the design document.
"""

__version__ = "0.1.0"
