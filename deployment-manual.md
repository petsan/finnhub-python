# Finn-Predictor — Deployment Manual

This manual covers running Finn-Predictor as a long-lived service. For
local development, see `installation-manual.md`. For end-user
walkthroughs of the dashboard once it's running, see `user-manual.md`.

Three deployment paths are supported:

| Path | When to pick it | Effort |
|---|---|---|
| **Proxmox LXC** (preferred) | Home lab / small fleet; you already run Proxmox VE. Single unprivileged container, native Python, one systemd unit. | One script run. |
| **Docker / Docker Compose** | You run Docker hosts; want image-pinned reproducibility; or you're on a managed container platform. | `docker compose up`. |
| **Bare metal / VM** | Existing Linux host with no container runtime, or you need the dashboard alongside other workloads on the same OS. | A handful of `pip` + systemd commands. |

All three end up running the same application — Streamlit on port 8501,
SQLite (or Postgres) backing store, optional bcrypt auth gate. Pick
whichever fits your existing platform.

---

## 1. Proxmox LXC (preferred)

### 1.1 Why LXC, not a VM

The app is a single-process Streamlit + SQLite service. Hardware-level
isolation (KVM) buys nothing here, while LXC's lower overhead lets a
small Proxmox node host this alongside other services without a
memory tax. The provided installer runs an **unprivileged** LXC, so
container escape buys very little even on a shared host.

### 1.2 Why not Docker-in-LXC

Stacking Docker inside an LXC means either a privileged LXC (loses
isolation) or fiddling with `nesting=1` plus cgroup-v2 quirks. Native
Python in an unprivileged LXC is cleaner: one systemd unit, one
logging path, one update procedure. If you want Docker semantics, run
the Docker path on its own VM or LXC and skip this section.

### 1.3 Prerequisites

* Proxmox VE 7.x or 8.x.
* The `ubuntu-24.04-standard` LXC template available on a storage the
  installer can reach (default `local`). The script downloads it
  automatically via `pveam download` if it's not already cached.
* Network reachable from the Proxmox host's bridge (`vmbr0` by default).
* Root access on the Proxmox host. The script must run there, not
  from a remote workstation — `pct` and `pveam` are host tools.

### 1.4 One-shot install

```bash
# On the Proxmox host, as root:
git clone -b finn-predictor https://github.com/petsan/finnhub-python /tmp/finn
cd /tmp/finn
./deploy/proxmox/install.sh
```

That's it. The script:

1. Picks the next free CTID ≥ 200 (or honours `--ctid`).
2. Downloads the Ubuntu 24.04 template if missing.
3. Creates an unprivileged LXC (1 GB RAM, 2 cores, 8 GB rootfs, DHCP
   on `vmbr0` by default).
4. Inside the container: installs Python 3.12 + dependencies, creates
   a `finn` user, clones the repo into `/opt/finn-predictor/src`,
   builds a venv at `/opt/finn-predictor/venv`, and writes a
   hardened systemd unit at `/etc/systemd/system/finn-predictor.service`.
5. Starts the service. The dashboard is reachable at
   `http://<lxc-ip>:8501`.

The final log block prints the URL and `pct enter` recipes for
inspecting / debugging.

### 1.5 Sizing

| Resource | Default | When to bump |
|---|---|---|
| RAM | 1024 MB | Bump to 2048 MB if you enable FinBERT (`--with-finbert`) or sentence-transformers (`--with-embeddings`). |
| Cores | 2 | Bump to 4 if you run heavy backfills concurrently with live ingest. |
| Disk | 8 GB | Bump to 16 GB if you keep > 1 year of news + price history, or > 30 K predictions. |

Override at install time:

```bash
./deploy/proxmox/install.sh --memory 2048 --cores 4 --disk 16
```

### 1.6 Networking

The default `--ip dhcp` picks up a lease from your LAN. For a stable
URL, give the LXC a static address:

```bash
./deploy/proxmox/install.sh \
    --ip 192.168.1.50/24 \
    --gateway 192.168.1.1
```

Then set up DNS (or a `/etc/hosts` entry) to point a name at it. The
Streamlit server binds to `0.0.0.0:8501` inside the LXC by default.

### 1.7 Optional features at install time

```bash
# Activate FinBERT (downloads ~440 MB on first ingest)
./deploy/proxmox/install.sh --with-finbert
# then inside the LXC:
#   echo 'Environment="FINN_PREDICTOR_SCORER=finbert"' >> /etc/systemd/system/finn-predictor.service.d/scorer.conf
#   systemctl daemon-reload && systemctl restart finn-predictor

# Activate the embedding clusterer (~80 MB on first ingest)
./deploy/proxmox/install.sh --with-embeddings

# Drop the Postgres driver (~30 MB saved if you only use SQLite)
./deploy/proxmox/install.sh --no-postgres
```

The env vars (`FINN_PREDICTOR_SCORER`, `FINN_PREDICTOR_CLASSIFIER`,
`FINN_PREDICTOR_CLUSTERER`, `FINN_PREDICTOR_PASSWORD_HASH`, etc.) can
also be passed straight to the installer; they land in the systemd
unit's `Environment=` block. Example:

```bash
FINN_PREDICTOR_PASSWORD_HASH='$2b$12$abc...' \
FINN_PREDICTOR_SCORER=finbert \
FINN_PREDICTOR_CLUSTERER=embedding \
    ./deploy/proxmox/install.sh
```

### 1.8 Upgrading

Re-run the installer against the same CTID. It detects the existing
container, `git fetch + reset --hard`s the branch, re-runs
`pip install`, and `systemctl restart`s the service. SQLite data
under `/opt/finn-predictor/data` is untouched.

```bash
./deploy/proxmox/install.sh --ctid 210
```

### 1.9 Removing

```bash
./deploy/proxmox/install.sh --ctid 210 --remove
```

Stops and destroys the LXC. The SQLite DB inside the rootfs is
deleted along with the container — back it up first if you care
(see §4).

### 1.10 Reverse proxy + TLS

The installer doesn't set up TLS. Front the LXC with a reverse proxy
(nginx, Caddy, Traefik) that terminates TLS and forwards to
`http://<lxc-ip>:8501`. A minimal Caddyfile:

```
finn.your-domain.example {
    reverse_proxy 192.168.1.50:8501 {
        # Streamlit uses websockets for the live update channel.
        # Caddy proxies them automatically; nginx needs explicit
        # Upgrade/Connection headers (see below).
    }
}
```

Nginx equivalent (websocket headers matter — Streamlit's UI updates
go over wss):

```nginx
server {
    server_name finn.your-domain.example;
    listen 443 ssl http2;
    # ... cert config ...
    location / {
        proxy_pass http://192.168.1.50:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 1d;
    }
}
```

Once the proxy is in place, set the auth gate
(`FINN_PREDICTOR_PASSWORD_HASH`) — exposed dashboards are a
credential-stuffing magnet otherwise.

---

## 2. Docker / Docker Compose

The project ships a `Dockerfile` and `docker-compose.yml` at repo
root. They run the same Streamlit + SQLite stack as the LXC path,
just inside a Docker container.

### 2.1 Quickstart

```bash
docker compose up --build
# → http://localhost:8501
```

The compose file mounts a named volume `finn_data` at `/data` and
points `FINN_PREDICTOR_DB_URL` there, so `docker compose down` keeps
your articles, predictions, and outcomes. `docker compose down -v`
deletes the volume.

### 2.2 Env vars

Add them under `environment:` in `docker-compose.yml`. The same set
the LXC supports applies — `FINN_PREDICTOR_SCORER`,
`FINN_PREDICTOR_CLASSIFIER`, `FINN_PREDICTOR_CLUSTERER`,
`FINN_PREDICTOR_PASSWORD_HASH`, `FINN_PREDICTOR_DB_URL`, etc. See
`installation-manual.md` §4.1 for the full reference table.

### 2.3 One-shot commands

```bash
RESET_DB=1 docker compose up --build       # wipe DB on the way up
docker compose run --rm app reset-db --yes
docker compose run --rm app retrain
docker compose run --rm app fit-classifier
docker compose run --rm app shell
docker compose run --rm app hash-password   # generate bcrypt for auth gate
```

### 2.4 Sizing & upgrades

The container runs as non-root (`appuser`). Set CPU / memory limits
in `docker-compose.yml` if you need them. Upgrade with
`docker compose pull && docker compose up -d` against a tagged image,
or `--build` against a local source tree.

---

## 3. Bare metal / VM

For when you can't or don't want to run a container runtime.

### 3.1 System prereqs

* Python ≥ 3.10 (3.12 recommended — that's what CI tests against).
* `git`, `build-essential` (or distro equivalent), `libxml2`,
  `libxslt1.1` (yfinance dependencies).
* A dedicated unprivileged user (`finn` is conventional).

Ubuntu 24.04 example:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential \
                    pkg-config libxml2 libxslt1.1
sudo useradd --system --create-home --home-dir /opt/finn-predictor --shell /bin/bash finn
```

### 3.2 App install

```bash
sudo -u finn -H bash <<'EOF'
cd /opt/finn-predictor
git clone -b finn-predictor https://github.com/petsan/finnhub-python src
python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel
venv/bin/pip install -r src/requirements.txt
venv/bin/pip install -e src
mkdir -p data
EOF
```

### 3.3 systemd unit

`/etc/systemd/system/finn-predictor.service` — the LXC installer
writes the same shape:

```ini
[Unit]
Description=Finn-Predictor Streamlit UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=finn
Group=finn
WorkingDirectory=/opt/finn-predictor/src
Environment="FINN_PREDICTOR_DB_URL=sqlite:////opt/finn-predictor/data/finn_predictor.db"
# Add other FINN_PREDICTOR_* env vars as needed.
ExecStart=/opt/finn-predictor/venv/bin/streamlit run finn_predictor/ui/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
Restart=on-failure
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/finn-predictor/data
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now finn-predictor
sudo systemctl status finn-predictor
```

---

## 4. Operations

### 4.1 Backups

The default SQLite file is the entire state — articles, sentiment
scores, predictions, outcomes, learned weights, and the logistic
calibration. Copy it while the service is stopped or use SQLite's
online backup API:

```bash
# Hot backup (LXC path):
pct exec 210 -- sqlite3 /opt/finn-predictor/data/finn_predictor.db \
    ".backup '/tmp/finn-$(date +%F).db'"
pct pull 210 /tmp/finn-$(date +%F).db ./finn-$(date +%F).db

# Restore (LXC path):
pct push 210 finn-2026-05-20.db /opt/finn-predictor/data/finn_predictor.db
pct exec 210 -- systemctl restart finn-predictor
```

For the Docker path, the same file lives in the `finn_data` volume;
back it up via `docker compose run --rm app cp /data/finn_predictor.db /data/backup.db`
plus a volume snapshot, or switch to Postgres
(`FINN_PREDICTOR_DB_URL=postgresql+psycopg://...`) and use your
existing Postgres backup pipeline.

### 4.2 Logs

* **LXC**: `journalctl -u finn-predictor -f` inside the container, or
  `pct exec 210 -- journalctl -u finn-predictor --since today` from
  the host.
* **Docker**: `docker compose logs -f app`.
* **Bare metal**: `journalctl -u finn-predictor -f`.

Switch to JSON output for log aggregators by setting
`FINN_PREDICTOR_LOG_FORMAT=json`. Tokens and bcrypt hashes are
scrubbed before emission by the package's `SecretScrubFilter` — safe
to ship to any external system.

### 4.3 Health checks

`curl -fsS http://<host>:8501/_stcore/health` returns `ok` when
Streamlit is alive. Use it as your liveness probe.

For a deeper readiness check (DB reachable, schema present):

```bash
/opt/finn-predictor/venv/bin/python -m finn_predictor.cli retrain --n-calls 0 2>&1 | head -1
# exits 3 with "not enough data" when DB is reachable but unfit;
# exits non-zero with a different message when DB is unreachable.
```

### 4.4 Monitoring

The structured-JSON log emits one record per request / job. Pipe it
to your existing log aggregator (Loki, ELK, Datadog Logs) and alert on:

* `ERROR`-level records from `finn_predictor.ingestion.*` (a streak
  means Finnhub is throwing 5xx or your key expired).
* `WARNING` records matching `scorer mismatch:` — the live scorer
  drifted from the saved predictions; rerun `retrain`.
* HTTP 5xx from the reverse proxy targeting port 8501.

### 4.5 Upgrades

Re-run the deployment script (LXC) or rebuild the image (Docker).
SQLite migrations land on import via `init_db`, so they apply
automatically on restart. For a major-version skip, run
`python -m finn_predictor.cli reset-db --yes` and re-backfill — the
data model is forward-compatible within a minor version but not
beyond.

### 4.6 Rotating the API key

The Finnhub key is **never stored on disk** by default — it lives in
`st.session_state` and in the headless `FINNHUB_API_KEY` env var.
Rotate it by:

* **UI path**: just paste the new key in the sidebar.
* **Headless path**: update the env var (`/etc/systemd/system/finn-predictor.service`'s
  `Environment=` block, or the compose file), then
  `systemctl restart finn-predictor` (or `docker compose restart app`).

The triple-layer scrubber (gateway / helper / display) catches any
log lines that would have referenced the old key. Once the service is
restarted, the in-memory copy is gone.

---

## 5. Hardening checklist

| | |
|---|---|
| ☐ | Set `FINN_PREDICTOR_PASSWORD_HASH` to a real bcrypt hash. Generate via `python -m finn_predictor.cli hash-password`. |
| ☐ | Front the dashboard with a reverse proxy that terminates TLS. Streamlit doesn't speak TLS natively. |
| ☐ | Set `FINN_PREDICTOR_LOG_FORMAT=json` and ship to a log aggregator. |
| ☐ | Limit `FINN_PREDICTOR_RATE_LIMIT` if your Finnhub plan has a different per-minute cap from the default 55. |
| ☐ | If using Postgres, store the connection string in a secret store, not the systemd unit file. |
| ☐ | Schedule nightly backups of the SQLite file (LXC: cron + `sqlite3 .backup` + `rsync` to a separate host). |
| ☐ | Run unprivileged. Both the Docker image and the LXC default to a non-root user; don't change that unless you have a hard reason. |
| ☐ | Pin the upstream branch / image tag for prod; `finn-predictor` is the active dev branch and will move under you. |
