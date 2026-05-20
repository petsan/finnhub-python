# Finn-Predictor — runtime image.
#
#   docker build -t finn-predictor .
#   docker run -p 8501:8501 -v finn_data:/data finn-predictor
#
# See docker-compose.yml for the canonical orchestrated workflow + the
# RESET_DB env-var contract.

FROM python:3.12-slim AS base

# scikit-optimize / scipy / pandas need compiled wheels; the slim base
# is fine for the wheels we get from PyPI but a tiny build chain helps
# if a fallback ever sdist-installs (e.g. arch without prebuilt wheels).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Where the SQLite DB lives. docker-compose mounts a named volume here.
    FINN_PREDICTOR_DB_URL=sqlite:////data/finn_predictor.db

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first, with the smallest possible build context,
# so source edits don't bust the layer cache.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Now the app source. The upstream library + finn_predictor are both
# importable via `pip install -e .` (setup.py picks up both packages
# via find_packages).
COPY setup.py setup.cfg README.md ./
COPY finnhub ./finnhub
COPY finn_predictor ./finn_predictor
RUN pip install -e .

# DB volume mount point. docker-compose mounts a named volume here so
# the SQLite file survives `docker compose down`.
RUN mkdir -p /data
VOLUME ["/data"]

# Entrypoint script: routes args to the CLI or starts Streamlit.
COPY docker/entrypoint.sh /usr/local/bin/finn-entrypoint
RUN chmod +x /usr/local/bin/finn-entrypoint

# Run as an unprivileged user. UID 1001 is a common choice that doesn't
# collide with most host users — and `--system` means we get a homeless
# account that can't log in. /data and /app are owned by finn so the
# app can write the SQLite file + scratch files but nothing else.
RUN useradd --system --uid 1001 --no-create-home --shell /sbin/nologin finn \
 && chown -R finn:finn /app /data
USER finn

EXPOSE 8501

# Streamlit health-check Docker can use for `depends_on: service_healthy`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/local/bin/finn-entrypoint"]
CMD ["serve"]
