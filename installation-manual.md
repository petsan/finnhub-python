# Finn-Predictor — Installation & Operations Manual

How to install, deploy, upgrade, reset, and back up Finn-Predictor.
Pairs with `user-manual.md` (which covers how to drive the dashboard
once it's running).

There are two supported install modes:

| Mode | When to use |
|---|---|
| **Local (venv)** | Development, iterating on the code, running tests. |
| **Docker** | Production / unattended deploys, easy reset/upgrade. |

Both produce the same UI; both default to a file-on-disk SQLite DB.
You can switch between them by pointing both at the same DB URL if you
want (just make sure only one is writing at a time).

---

## 1. Prerequisites

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12+ | Local install only |
| Docker | 24+ (with `compose` v2 plugin) | Docker install only |
| git | any | Cloning the repo |
| ~150 MB free disk | — | Image + dependencies |
| A Finnhub account | free tier is fine | News ingestion |

The application is read-only against the DB if you don't have a
Finnhub key, so you can demo it without a key.

---

## 2. Local install (venv)

### 2.1 Clone + create the virtualenv

```bash
git clone <repo-url> finn-predictor
cd finn-predictor

# Use the finn-predictor branch (where the app lives; master is the
# upstream library).
git checkout finn-predictor

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
```

### 2.2 Install dependencies

```bash
# App runtime
.venv/bin/pip install -r requirements.txt

# The upstream finnhub library is installed editable so `import finnhub`
# works alongside `import finn_predictor`.
.venv/bin/pip install -e .

# Test toolchain (only if you plan to run the test suite)
.venv/bin/pip install pytest pytest-cov pytest-mock requests-mock freezegun
```

### 2.3 Run the dashboard

```bash
.venv/bin/streamlit run finn_predictor/ui/app.py
```

Then open <http://localhost:8501>. Paste your Finnhub key in the
sidebar and follow §4 of `user-manual.md`.

### 2.4 Run the test suite

```bash
.venv/bin/python -m pytest --cov
```

305 tests pass at ~97% line+branch coverage as of the
`learning + docker: holdout gate + container deploy` commit. No test
talks to the real Finnhub API.

---

## 3. Docker install

### 3.1 Build + start

From the repo root:

```bash
docker compose up --build
```

That single command:

1. Builds the image from `Dockerfile` (~50 s on a fresh machine,
   most of it `pandas` and `scikit-optimize` wheels).
2. Creates a Docker named volume `finn_data` for the SQLite file.
3. Starts the container, binds host `8501` → container `8501`.
4. Streams startup logs to your terminal. Hit `Ctrl-C` to stop.

Open <http://localhost:8501>.

### 3.2 Detached mode (production-ish)

```bash
docker compose up -d            # start in the background
docker compose logs -f          # tail logs
docker compose ps               # status
docker compose down             # stop, KEEP the volume
docker compose down -v          # stop, DELETE the volume
```

The container has a `/_stcore/health` healthcheck — Docker will report
`healthy` after ~20 s. If it stays `starting` longer than a minute,
inspect with `docker compose logs`.

### 3.3 Reset the database

Two ways:

**Wipe on next startup (one-shot, sticky):**
```bash
RESET_DB=1 docker compose up --build
```

**Wipe explicitly, then start clean:**
```bash
docker compose run --rm app reset-db --yes
docker compose up
```

Either way the underlying named volume is reused — the *schema* is
recreated, but the volume itself isn't deleted.

To delete the volume entirely (e.g. you want yfinance to refetch
every bar):
```bash
docker compose down -v
docker volume rm finn_data        # only if down didn't already
```

### 3.4 Retrain from the command line

Useful for cron:

```bash
docker compose run --rm app retrain                       # use persisted policy
docker compose run --rm app retrain --n-calls 50          # more iterations
docker compose run --rm app retrain --activate yes        # force-activate
docker compose run --rm app retrain --activate no         # save inactive
```

Returns exit `0` on success, `3` if there aren't enough closed
predictions (need ≥10).

### 3.5 Shell into the container

For debugging or ad-hoc Python:

```bash
docker compose run --rm app shell
# inside:
python -c "from finn_predictor.storage import create_engine_and_session; \
           from finn_predictor.storage.models import NewsArticle; \
           e, SL = create_engine_and_session('sqlite:////data/finn_predictor.db'); \
           print(SL().query(NewsArticle).count())"
```

`shell` can also take a command to exec:

```bash
docker compose run --rm app shell -c 'cat /etc/os-release'
```

### 3.6 Port conflicts

The compose file maps host port `8501` → container port `8501`. If
`8501` is in use:

```bash
# One-off: pass via env var
HOST_PORT=8510 docker compose up
```

(That would require editing `docker-compose.yml` to substitute
`${HOST_PORT:-8501}` for the host side of the mapping; if you'd like
that ergonomic, ask for it as a small follow-up.)

Or just bind to a different host port for a one-shot run:

```bash
docker compose build
docker run -d --name finn-test \
  -p 127.0.0.1:8510:8501 \
  -v finn_data:/data \
  finn-predictor:latest
# stop: docker rm -f finn-test
```

### 3.7 Exposing on the LAN

**Don't** without an auth layer in front. The UI is unauthenticated;
anyone who can reach the port can paste an API key into the sidebar
and trigger ingestion (and see your existing data).

If you really need it, terminate TLS + auth (e.g. nginx with HTTP basic,
or Tailscale's `tailscale serve`) in front, and change the compose port
mapping from the default localhost-only binding.

---

## 4. Configuration

### 4.1 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `FINNHUB_API_KEY` | unset | Required for the CLI tools `retrain` and `ingest`. UI accepts the key in-session, so unset is fine for UI-only deployments. |
| `FINN_PREDICTOR_DB_URL` | `sqlite:///finn_predictor.db` (local), `sqlite:////data/finn_predictor.db` (Docker) | SQLAlchemy URL. Set to `postgresql+psycopg://user:pw@host/db` to use Postgres — the upserts are dialect-aware. |
| `FINN_PREDICTOR_TIMEOUT` | `15` | Per-request HTTP timeout in seconds. |
| `FINN_PREDICTOR_RATE_LIMIT` | `55` | Finnhub calls per minute (free tier is ~60). |
| `FINN_PREDICTOR_PASSWORD_HASH` | unset | When set to a bcrypt hash, the UI gates every render behind a password prompt until the right plaintext is supplied. Unset = no auth (safe default for localhost dev). Generate with `python -m finn_predictor.cli hash-password`. |
| `FINN_PREDICTOR_LOG_FORMAT` | `text` | `text` for human stdout, `json` for one-record-per-line structured output suitable for log aggregators. |
| `FINN_PREDICTOR_LOG_LEVEL` | `INFO` | Standard `logging` level name. |
| `RESET_DB` | `0` | Docker only. Set to `1` (or `true`) to wipe the DB on container start. |

### 4.1.1 Enabling the UI password gate

```bash
# generate a hash (prompts for password on stdin so it doesn't end up
# in shell history)
$ python -m finn_predictor.cli hash-password
Password: ********
$2b$12$abc...verylongbcrypthash

# stash it in the environment
$ export FINN_PREDICTOR_PASSWORD_HASH='$2b$12$abc...verylongbcrypthash'

# or in docker-compose.yml, add under `environment:`:
#   FINN_PREDICTOR_PASSWORD_HASH: "$2b$12$abc...verylongbcrypthash"
```

The plaintext is never stored anywhere — only the hash. Wiping the
SQLite volume does NOT lock you out, because the auth state lives in
the env var, not the DB.

### 4.1.2 PostgreSQL

```bash
export FINN_PREDICTOR_DB_URL="postgresql+psycopg://finn:secret@db.example.com:5432/finn_predictor"
```

`psycopg[binary]` ships with the default `requirements.txt`. The
SQLAlchemy upserts (article dedupe, price-bar dedupe) automatically
switch to the Postgres `ON CONFLICT DO NOTHING` builder when the
dialect is detected at runtime. Schema migrations are still
idempotent `Base.metadata.create_all(engine)` — for production you
probably want Alembic on top of this, but the basics work.

### 4.2 Settings stored in the DB

Cross-session UI preferences live in the `app_settings` table:

| Key | Default | Set via |
|---|---|---|
| `activation_policy` | `AUTO` | Learning tab radio |
| `holdout_tolerance` | `0.01` | Learning tab slider |

These survive Streamlit restarts, container restarts, and volume
mounts.

### 4.3 Files / directories created at runtime

| Path | What it holds |
|---|---|
| `finn_predictor.db` (local) or `/data/finn_predictor.db` (Docker) | The whole application state — news articles, sentiment scores, prices, predictions, outcomes, related entities, learned weights, app settings. |
| `htmlcov/` (local, after `pytest --cov-report=html`) | Coverage HTML output. |
| `/tmp/finn_streamlit.log` (local, if you run via the dev recipe) | Streamlit's stdout/stderr. |

Backups: copy the single SQLite file while the app is stopped. To
restore: drop it back in place. (For Docker: `docker compose down`,
then `cp` into the volume's host path, then `docker compose up`.)

---

## 5. CLI cheat-sheet

The CLI is `python -m finn_predictor.cli` locally, or the Docker
entrypoint when running in a container.

```bash
# Print the Streamlit launch command (no-op when in Docker — the
# entrypoint runs Streamlit itself).
python -m finn_predictor.cli serve

# Drop + recreate every table. Refuses without --yes.
python -m finn_predictor.cli reset-db --yes

# Run one training cycle. Prints the TrainingReport as JSON.
python -m finn_predictor.cli retrain --n-calls 30
python -m finn_predictor.cli retrain --activate auto     # default
python -m finn_predictor.cli retrain --activate yes      # force-activate
python -m finn_predictor.cli retrain --activate no       # save inactive

# Generate a bcrypt hash for FINN_PREDICTOR_PASSWORD_HASH.
# Prompts on stdin if you omit the argument (recommended — keeps the
# plaintext out of shell history).
python -m finn_predictor.cli hash-password
python -m finn_predictor.cli hash-password 'my plaintext'

# Run one daily-ingest cycle headlessly. Reads FINNHUB_API_KEY from
# env. Suitable for cron / scheduled-task sidecar.
FINNHUB_API_KEY=... python -m finn_predictor.cli ingest
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | OK |
| `2` | Argparse error, missing arg (`reset-db` without `--yes`, `ingest` without `FINNHUB_API_KEY`, `hash-password` with too-short input) |
| `3` | `retrain` couldn't run (fewer than 10 closed predictions) |

### 5.1 Scheduled ingestion via cron

The Streamlit process can't reliably run a background scheduler — it
only fires while the page is open. For unattended deployment, use host
cron + the `ingest` subcommand:

```cron
# crontab -e, then add:
#   M H DoM Mon DoW
30 21 * * 1-5  FINNHUB_API_KEY=... docker compose -f /path/to/finn/docker-compose.yml run --rm app ingest >> /var/log/finn-ingest.log 2>&1
```

21:30 UTC weekdays = ~30 min after the NYSE close. Adjust as needed.
Each `ingest` run reuses the same `finn_data` volume, so predictions
accumulate. Pair with a separate weekly `retrain` line if you want
auto-improvement.

---

## 6. Upgrades

### 6.1 Local upgrade

```bash
git fetch origin
git checkout finn-predictor && git pull --ff-only
.venv/bin/pip install -r requirements.txt --upgrade
.venv/bin/pip install -e . --upgrade
# Then restart Streamlit.
```

Schema migrations are idempotent — `init_db` only creates missing
tables. New columns added in a future release would need an explicit
migration; nothing along that path exists today.

### 6.2 Docker upgrade

```bash
git pull --ff-only
docker compose down
docker compose up --build -d
```

The volume is reused, so all your articles / predictions / learned
weights are preserved. If you want to start completely fresh after an
upgrade, add `-v` to the `down` line.

---

## 7. Resetting

### 7.1 Local

```bash
rm -f finn_predictor.db
.venv/bin/python -m finn_predictor.cli reset-db --yes   # equivalent
```

Either recreates an empty schema. The UI handles a missing file on
first request — `create_engine_and_session` + `init_db` are called at
the top of every render.

### 7.2 Docker

See §3.3.

### 7.3 Restoring from a backup

Stop the app, copy your backup over the live SQLite file, start again:

```bash
# Local
cp ~/backups/finn-2026-05-20.db ./finn_predictor.db

# Docker (path inside the volume varies by Docker root, easiest is:)
docker compose down
docker run --rm -v finn_data:/data -v "$PWD/backups:/in" alpine \
  cp /in/finn-2026-05-20.db /data/finn_predictor.db
docker compose up -d
```

---

## 8. Troubleshooting installs

### Build error: `Could not build wheels for skopt`

You're on a platform where scikit-optimize doesn't ship a prebuilt
wheel. The Dockerfile installs `build-essential` to cover this, but
locally you may need `python3-dev` and a C compiler. On Ubuntu:

```bash
sudo apt install -y build-essential python3-dev
```

Then `.venv/bin/pip install -r requirements.txt` again.

### `ModuleNotFoundError: No module named 'finn_predictor'`

You haven't run `pip install -e .` in the venv yet. The
`finn_predictor` package is installed alongside the upstream
`finnhub` package by that command.

### Streamlit boots but the page is blank with a Python traceback

The traceback is rendered inline by Streamlit and is the actual error
— read it. Most common: an old SQLite DB from a much earlier version
with a missing column. Fix is `reset-db --yes` or restore from a
known-good backup.

### Docker build hangs at `[2/12] Pulling base image`

Slow network or rate-limited Docker Hub anonymous pull. Wait, or
log in (`docker login`) for higher anonymous limits.

### `docker compose up` succeeds but `localhost:8501` doesn't load

Inspect:

```bash
docker compose ps
docker compose logs app
```

Common: another process on the host already owns 8501. Use a
different host port — see §3.6.

### Healthcheck stays `starting` then flips to `unhealthy`

Almost always means the container failed to start Streamlit. Check
`docker compose logs app` for the actual Python error. After an
upgrade, the most likely cause is a missing dependency in
`requirements.txt`.

### "SSL: CERTIFICATE_VERIFY_FAILED" inside Docker

You have `HTTPS_PROXY` set in your host environment and it leaked
through to Docker. The application's *own* ingestion path already
bypasses `HTTPS_PROXY` via `Session.trust_env=False`, so this only
shows up if some other library inside the image (e.g. `pip`)
inherits the host env. Pass `--build-arg http_proxy=` /
`--build-arg https_proxy=` to disable.

---

## 9. Production hardening (not done by default)

The defaults are tuned for a single-user dev machine. Before exposing
this beyond `localhost` you should at minimum:

1. **Put TLS + auth in front of Streamlit.** The UI is unauthenticated.
2. **Switch to PostgreSQL** by setting `FINN_PREDICTOR_DB_URL`. The
   ORM is SQLAlchemy-portable; only the `ON CONFLICT DO NOTHING`
   upsert paths use SQLite-specific syntax (in `storage/repo.py`),
   and those would need to swap to a Postgres dialect call.
3. **Rotate the Finnhub key periodically** and treat session-state
   leakage seriously — the dashboard never persists the key, but if a
   prediction error ever leaks one (the system has three scrubbing
   layers but tests can't catch everything), rotate immediately.
4. **Mount a logging volume** and configure `logging.basicConfig`
   to write to it.
5. **Schedule the daily ingest job** via `docker compose run --rm app
   …` in a host cron, rather than relying on APScheduler running
   inside the Streamlit process (which only fires while the page is
   active).

---

## 10. Where to go next

- **`user-manual.md`** — how to drive the dashboard once it's
  running.
- **`summary.md`** — high-level architecture overview.
- **`progress.md`** — design decisions and implementation status.
- **`diff.md`** — per-commit change log.
