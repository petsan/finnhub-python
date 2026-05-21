#!/usr/bin/env bash
# Finn-Predictor — Proxmox LXC installer.
#
# Provisions an unprivileged Ubuntu 24.04 LXC container, installs the
# Finn-Predictor app and its dependencies, and starts it under systemd
# on port 8501. Designed to run on the Proxmox VE host (root).
#
# Why LXC over a VM? The app is a single-process Streamlit + SQLite
# service — there's nothing to gain from KVM's hardware-level
# isolation, and LXC's lower overhead lets a small home-lab Proxmox
# node host this alongside other services without a memory tax. The
# container runs unprivileged so escape buys very little.
#
# Why native Python instead of Docker-in-LXC? The project's existing
# Dockerfile is fine for `docker compose up` deployments, but stacking
# Docker inside an LXC means either a privileged LXC (loses isolation)
# or fiddling with nesting=1 + cgroup quirks. Native Python in LXC is
# cleaner: one systemd unit, one logging path, one update procedure.
#
# Usage:
#   # On the Proxmox host, as root:
#   ./install.sh                                  # all defaults
#   CTID=210 CT_HOSTNAME=finn.lan ./install.sh       # override via env
#   ./install.sh --ctid 210 --hostname finn.lan   # or via flags
#
# Idempotency:
#   * Re-running on an existing CTID skips creation and just refreshes
#     the app code + dependencies + systemd unit.
#   * Re-running with a different CTID provisions a second container.
#   * Use --remove to destroy a container created by this script.
#
# Required env (or flags):
#   CTID         (optional)  Container ID. Default: next free ≥ 200.
#   CT_HOSTNAME     (optional)  Container hostname. Default: finn-predictor.
#   FINNHUB_API_KEY (optional)  If set, written into the service env so the
#                              UI sidebar can pre-fill and the headless
#                              `ingest` CLI works without manual setup.
#                              Otherwise the UI accepts a session-only
#                              key from the sidebar (still recommended
#                              for prod since the env var lands on disk
#                              inside the container).
#
# Optional env (see DEFAULTS section below for the full list).

set -euo pipefail

# ---------- Defaults ----------------------------------------------------

: "${CTID:=}"                                         # auto-pick if blank
: "${CT_HOSTNAME:=finn-predictor}"
: "${TEMPLATE_NAME:=ubuntu-24.04-standard_24.04-2_amd64.tar.zst}"
: "${TEMPLATE_STORAGE:=local}"
: "${ROOTFS_STORAGE:=local-lvm}"
: "${ROOTFS_SIZE_GB:=8}"
: "${MEMORY_MB:=1024}"
: "${SWAP_MB:=512}"
: "${CORES:=2}"
: "${BRIDGE:=vmbr0}"
: "${IP:=dhcp}"                                       # or CIDR like 192.168.1.50/24
: "${GATEWAY:=}"                                      # required when IP is not dhcp
: "${DNS:=}"                                          # blank → use host DNS
: "${UNPRIVILEGED:=1}"                                # 0 → privileged (rarely needed)
: "${NESTING:=0}"                                     # 1 only if you must run docker inside
: "${TIMEZONE:=UTC}"

# App config
: "${REPO_URL:=https://github.com/petsan/finnhub-python.git}"
: "${BRANCH:=finn-predictor}"
: "${LOCAL_SOURCE:=}"                                 # path to a local checkout; if set, skips git clone
: "${APP_USER:=finn}"
: "${APP_HOME:=/opt/finn-predictor}"
: "${PORT:=8501}"
: "${BIND_ADDRESS:=0.0.0.0}"
: "${INSTALL_TORCH:=0}"                               # 1 → also install torch + transformers (FinBERT)
: "${INSTALL_EMBEDDINGS:=0}"                          # 1 → also install sentence-transformers
: "${INCLUDE_POSTGRES_DRIVER:=1}"                     # 0 → strip psycopg from requirements

# Optional runtime env passed through to the systemd unit
: "${FINNHUB_API_KEY:=}"
: "${FINN_PREDICTOR_DB_URL:=}"                        # default sqlite path used if blank
: "${FINN_PREDICTOR_PASSWORD_HASH:=}"                 # bcrypt hash for UI auth gate
: "${FINN_PREDICTOR_SCORER:=}"                        # vader (default) | finbert
: "${FINN_PREDICTOR_CLASSIFIER:=}"                    # rule (default) | logreg
: "${FINN_PREDICTOR_CLUSTERER:=}"                     # prefix (default) | embedding
: "${FINN_PREDICTOR_LOG_FORMAT:=}"                    # text (default) | json
: "${FINN_PREDICTOR_LOG_LEVEL:=}"                     # INFO (default)

REMOVE_MODE=0
DRY_RUN=0

# ---------- Argument parsing -------------------------------------------

usage() {
    sed -n '2,40p' "$0" | sed 's/^# *//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctid) CTID="$2"; shift 2 ;;
        --hostname) CT_HOSTNAME="$2"; shift 2 ;;
        --ip) IP="$2"; shift 2 ;;
        --gateway) GATEWAY="$2"; shift 2 ;;
        --bridge) BRIDGE="$2"; shift 2 ;;
        --memory) MEMORY_MB="$2"; shift 2 ;;
        --cores) CORES="$2"; shift 2 ;;
        --disk) ROOTFS_SIZE_GB="$2"; shift 2 ;;
        --rootfs-storage) ROOTFS_STORAGE="$2"; shift 2 ;;
        --template-storage) TEMPLATE_STORAGE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --local-source) LOCAL_SOURCE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --with-finbert) INSTALL_TORCH=1; shift ;;
        --with-embeddings) INSTALL_EMBEDDINGS=1; shift ;;
        --no-postgres) INCLUDE_POSTGRES_DRIVER=0; shift ;;
        --remove) REMOVE_MODE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

# ---------- Helpers -----------------------------------------------------

log() { printf '[install] %s\n' "$*" >&2; }
die() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }
run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] %s\n' "$*" >&2
    else
        eval "$@"
    fi
}

require_root() {
    [[ $EUID -eq 0 ]] || die "this script must be run as root on the Proxmox host"
}

require_pve() {
    command -v pct >/dev/null 2>&1 || die "pct not found — run this on a Proxmox VE host"
    command -v pveam >/dev/null 2>&1 || die "pveam not found — Proxmox tools missing"
}

next_free_ctid() {
    # Prefer Proxmox's cluster-aware /cluster/nextid endpoint — it knows
    # about CTIDs on every node in the cluster, not just this host's
    # local view (which `pct status` is limited to). Falls back to a
    # local scan if pvesh isn't available or returns nothing usable.
    local id
    if command -v pvesh >/dev/null 2>&1; then
        id=$(pvesh get /cluster/nextid 2>/dev/null || true)
        if [[ "$id" =~ ^[0-9]+$ && "$id" -ge 200 ]]; then
            echo "$id"
            return 0
        fi
    fi
    for id in $(seq 200 999); do
        if ! pct status "$id" >/dev/null 2>&1; then
            echo "$id"
            return 0
        fi
    done
    die "no free CTID found in 200-999"
}

ensure_template() {
    local path="/var/lib/vz/template/cache/${TEMPLATE_NAME}"
    if [[ -f "$path" ]]; then
        log "template ${TEMPLATE_NAME} already cached"
        return 0
    fi
    log "downloading template ${TEMPLATE_NAME}"
    run "pveam update"
    run "pveam download ${TEMPLATE_STORAGE} ${TEMPLATE_NAME}"
}

ctid_exists() { pct status "$1" >/dev/null 2>&1; }

# ---------- Remove mode -------------------------------------------------

if [[ "$REMOVE_MODE" == "1" ]]; then
    require_root
    require_pve
    [[ -n "$CTID" ]] || die "--remove requires --ctid"
    ctid_exists "$CTID" || die "CTID $CTID does not exist"
    log "stopping CTID $CTID"
    run "pct stop $CTID || true"
    log "destroying CTID $CTID"
    run "pct destroy $CTID"
    log "done"
    exit 0
fi

# ---------- Pre-flight --------------------------------------------------

require_root
require_pve

if [[ -z "$CTID" ]]; then
    CTID="$(next_free_ctid)"
    log "auto-picked CTID=$CTID"
fi

[[ "$IP" == "dhcp" || -n "$GATEWAY" ]] || die "static IP requires --gateway"

ensure_template

# ---------- Provision (create LXC if missing) --------------------------

if ctid_exists "$CTID"; then
    log "CTID $CTID already exists — skipping create, will refresh app inside"
else
    log "creating LXC $CTID ($CT_HOSTNAME) on $ROOTFS_STORAGE"
    NET_ARGS="name=eth0,bridge=${BRIDGE},ip=${IP}"
    if [[ "$IP" != "dhcp" ]]; then
        NET_ARGS="${NET_ARGS},gw=${GATEWAY}"
    fi
    FEATURES_ARG=()
    if [[ "$UNPRIVILEGED" == "1" ]]; then
        # keyctl=1 is required for systemd in unprivileged Ubuntu LXC
        # so journal/login keyring work; nesting=0 by default, set to 1
        # only when the user explicitly opted in via env.
        FEATURES_ARG=(--features "keyctl=1,nesting=${NESTING}")
    fi

    # We don't pass --password — the container has no root login by
    # default. SSH access goes via `pct enter` from the host; if you
    # want SSH-from-network, drop a key into /root/.ssh/authorized_keys
    # after creation (or extend this script).
    run "pct create $CTID ${TEMPLATE_STORAGE}:vztmpl/${TEMPLATE_NAME} \\
            --hostname ${CT_HOSTNAME} \\
            --cores ${CORES} \\
            --memory ${MEMORY_MB} \\
            --swap ${SWAP_MB} \\
            --rootfs ${ROOTFS_STORAGE}:${ROOTFS_SIZE_GB} \\
            --net0 ${NET_ARGS} \\
            --onboot 1 \\
            --unprivileged ${UNPRIVILEGED} \\
            --start 0 \\
            ${FEATURES_ARG[*]:-}"

    if [[ -n "$DNS" ]]; then
        run "pct set $CTID --nameserver $DNS"
    fi

    run "pct start $CTID"

    # Wait for the container's network to come up. DHCP can take a few
    # seconds; static IP is instant. 30 s is generous for either.
    log "waiting for network in CTID $CTID"
    for _ in $(seq 1 30); do
        if pct exec "$CTID" -- bash -c 'getent hosts deb.debian.org || getent hosts archive.ubuntu.com' >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi

# ---------- Inside-container setup -------------------------------------

# Everything below is shipped into the container as a single script
# (rather than a chain of `pct exec` calls) so it's atomic on the
# inside — easier to debug, less round-trip noise from pct.

INNER_SCRIPT=$(mktemp)
LOCAL_SOURCE_TARBALL=""
cleanup() {
    rm -f "$INNER_SCRIPT" "$LOCAL_SOURCE_TARBALL"
}
trap cleanup EXIT

# When --local-source is set, push a tarball of the checkout into the
# LXC at $APP_HOME/src ahead of the inner script. The inner script
# detects the pre-seeded directory and skips the git clone step.
if [[ -n "$LOCAL_SOURCE" ]]; then
    [[ -d "$LOCAL_SOURCE" ]] || die "--local-source path '$LOCAL_SOURCE' is not a directory"
    log "packaging local source from $LOCAL_SOURCE"
    LOCAL_SOURCE_TARBALL=$(mktemp --suffix=.tar.gz)
    # Exclude build/dev artefacts and .git — we ship a flat source
    # tree, not a clone. The inner script's "pre-seeded source" branch
    # specifically looks for "has requirements.txt AND no .git dir".
    tar -czf "$LOCAL_SOURCE_TARBALL" \
        --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
        --exclude='*.egg-info' --exclude='htmlcov' --exclude='*.sqlite*' \
        --exclude='finn_predictor.db' --exclude='.git' \
        -C "$LOCAL_SOURCE" .
    if [[ "$DRY_RUN" != "1" ]]; then
        pct exec "$CTID" -- mkdir -p /opt/finn-predictor
        pct push "$CTID" "$LOCAL_SOURCE_TARBALL" /tmp/finn-source.tar.gz
        pct exec "$CTID" -- bash -c '
            rm -rf /opt/finn-predictor/src
            mkdir -p /opt/finn-predictor/src
            tar -xzf /tmp/finn-source.tar.gz -C /opt/finn-predictor/src
            rm -f /tmp/finn-source.tar.gz
        '
    fi
fi

# Compose env-var lines for the systemd unit. Only non-empty values
# get written; empty ones use the app's built-in defaults.
ENV_LINES=""
add_env() {
    local key="$1" val="$2"
    if [[ -n "$val" ]]; then
        ENV_LINES+="Environment=\"${key}=${val}\"\n"
    fi
}
add_env FINNHUB_API_KEY "$FINNHUB_API_KEY"
add_env FINN_PREDICTOR_DB_URL "$FINN_PREDICTOR_DB_URL"
add_env FINN_PREDICTOR_PASSWORD_HASH "$FINN_PREDICTOR_PASSWORD_HASH"
add_env FINN_PREDICTOR_SCORER "$FINN_PREDICTOR_SCORER"
add_env FINN_PREDICTOR_CLASSIFIER "$FINN_PREDICTOR_CLASSIFIER"
add_env FINN_PREDICTOR_CLUSTERER "$FINN_PREDICTOR_CLUSTERER"
add_env FINN_PREDICTOR_LOG_FORMAT "$FINN_PREDICTOR_LOG_FORMAT"
add_env FINN_PREDICTOR_LOG_LEVEL "$FINN_PREDICTOR_LOG_LEVEL"

cat > "$INNER_SCRIPT" <<INNER
#!/usr/bin/env bash
set -euo pipefail

APP_USER='${APP_USER}'
APP_HOME='${APP_HOME}'
REPO_URL='${REPO_URL}'
BRANCH='${BRANCH}'
PORT='${PORT}'
BIND_ADDRESS='${BIND_ADDRESS}'
TIMEZONE='${TIMEZONE}'
INSTALL_TORCH='${INSTALL_TORCH}'
INSTALL_EMBEDDINGS='${INSTALL_EMBEDDINGS}'
INCLUDE_POSTGRES_DRIVER='${INCLUDE_POSTGRES_DRIVER}'

log() { printf '[inner] %s\n' "\$*" >&2; }

# ---------- Base packages -----------------------------------------------

export DEBIAN_FRONTEND=noninteractive
log "apt update + install base packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \\
    ca-certificates curl git tzdata \\
    python3 python3-venv python3-pip \\
    build-essential pkg-config \\
    libxml2 libxslt1.1

# tzdata
ln -fs "/usr/share/zoneinfo/\${TIMEZONE}" /etc/localtime
dpkg-reconfigure -f noninteractive tzdata >/dev/null 2>&1 || true

# ---------- App user ----------------------------------------------------

if ! id -u "\$APP_USER" >/dev/null 2>&1; then
    log "creating user \$APP_USER"
    useradd --system --create-home --home-dir "\$APP_HOME" --shell /bin/bash "\$APP_USER"
else
    log "user \$APP_USER already exists"
fi

mkdir -p "\$APP_HOME"
chown "\$APP_USER:\$APP_USER" "\$APP_HOME"

# ---------- Source ------------------------------------------------------

if [[ -f "\$APP_HOME/src/requirements.txt" && ! -d "\$APP_HOME/src/.git" ]]; then
    # Pre-seeded by --local-source on the outside. Don't touch the tree;
    # just make sure the app user owns it.
    log "using pre-seeded source at \$APP_HOME/src"
    chown -R "\$APP_USER:\$APP_USER" "\$APP_HOME/src"
elif [[ -d "\$APP_HOME/src/.git" ]]; then
    log "refreshing existing checkout"
    sudo -u "\$APP_USER" git -C "\$APP_HOME/src" fetch origin --quiet
    sudo -u "\$APP_USER" git -C "\$APP_HOME/src" checkout "\$BRANCH" --quiet
    sudo -u "\$APP_USER" git -C "\$APP_HOME/src" reset --hard "origin/\$BRANCH" --quiet
else
    log "cloning \$REPO_URL (branch \$BRANCH)"
    sudo -u "\$APP_USER" git clone --depth 1 --branch "\$BRANCH" "\$REPO_URL" "\$APP_HOME/src"
fi

# ---------- Python virtualenv ------------------------------------------

if [[ ! -d "\$APP_HOME/venv" ]]; then
    log "creating venv"
    sudo -u "\$APP_USER" python3 -m venv "\$APP_HOME/venv"
fi

sudo -u "\$APP_USER" "\$APP_HOME/venv/bin/pip" install --upgrade --quiet pip setuptools wheel

# Strip the optional Postgres driver from requirements when the user opts
# out; sentence-transformers / torch live in their own optional install
# steps below.
REQ_FILE="\$APP_HOME/src/requirements.txt"
if [[ "\$INCLUDE_POSTGRES_DRIVER" == "0" ]]; then
    REQ_FILE="\$APP_HOME/src/requirements.no-pg.txt"
    grep -v '^psycopg' "\$APP_HOME/src/requirements.txt" > "\$REQ_FILE"
fi

log "installing app dependencies"
sudo -u "\$APP_USER" "\$APP_HOME/venv/bin/pip" install --quiet -r "\$REQ_FILE"
sudo -u "\$APP_USER" "\$APP_HOME/venv/bin/pip" install --quiet -e "\$APP_HOME/src"

if [[ "\$INSTALL_TORCH" == "1" ]]; then
    log "installing torch + transformers (FinBERT scorer opt-in)"
    sudo -u "\$APP_USER" "\$APP_HOME/venv/bin/pip" install --quiet torch transformers
fi
if [[ "\$INSTALL_EMBEDDINGS" == "1" ]]; then
    log "installing sentence-transformers (embedding clusterer opt-in)"
    sudo -u "\$APP_USER" "\$APP_HOME/venv/bin/pip" install --quiet sentence-transformers
fi

# ---------- Data dir + DB URL default ----------------------------------

mkdir -p "\$APP_HOME/data"
chown "\$APP_USER:\$APP_USER" "\$APP_HOME/data"

# ---------- systemd service --------------------------------------------

log "writing systemd unit"
cat > /etc/systemd/system/finn-predictor.service <<UNIT
[Unit]
Description=Finn-Predictor Streamlit UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=\$APP_USER
Group=\$APP_USER
WorkingDirectory=\$APP_HOME/src

# Default DB lives under /opt/finn-predictor/data so it survives an
# in-place app refresh (the src/ checkout is reset --hard each run).
Environment="FINN_PREDICTOR_DB_URL=sqlite:///\$APP_HOME/data/finn_predictor.db"
$(printf '%b' "\$(echo -en '${ENV_LINES}')")

ExecStart=\$APP_HOME/venv/bin/streamlit run finn_predictor/ui/app.py \\\\
    --server.port \$PORT \\\\
    --server.address \$BIND_ADDRESS \\\\
    --server.headless true \\\\
    --browser.gatherUsageStats false

Restart=on-failure
RestartSec=5

# Hardening — the unit doesn't need write access outside its own
# state dir, doesn't need new privileges, and shouldn't see real
# devices. NoNewPrivileges + PrivateTmp are cheap wins; the rest are
# Streamlit-compatible.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=\$APP_HOME/data
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable finn-predictor.service >/dev/null 2>&1 || true
systemctl restart finn-predictor.service

log "deployment complete; service status:"
systemctl --no-pager --lines=0 status finn-predictor.service || true
INNER

chmod +x "$INNER_SCRIPT"

# Copy script in and run it
log "running in-container setup (CTID $CTID)"
if [[ "$DRY_RUN" == "1" ]]; then
    log "(dry-run) would push $INNER_SCRIPT into CTID $CTID and execute"
else
    pct push "$CTID" "$INNER_SCRIPT" /root/install-inner.sh --perms 0755
    pct exec "$CTID" -- bash /root/install-inner.sh
    pct exec "$CTID" -- rm -f /root/install-inner.sh
fi

# ---------- Report ------------------------------------------------------

LXC_IP=$(pct exec "$CTID" -- bash -c "ip -4 -o addr show eth0 | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null || true)

cat <<DONE

============================================================
Finn-Predictor deployed.
  CTID:       $CTID
  Hostname:   $CT_HOSTNAME
  IP:         ${LXC_IP:-<pending>}
  Port:       $PORT
  Service:    finn-predictor.service (inside the LXC)
  Data dir:   $APP_HOME/data (DB lives here; persists across upgrades)

Open the UI at:   http://${LXC_IP:-<container-ip>}:$PORT

Inside the LXC:
  pct enter $CTID
  systemctl status finn-predictor
  journalctl -u finn-predictor -f
  $APP_HOME/venv/bin/python -m finn_predictor.cli --help

To upgrade (pull latest branch + restart):
  ./install.sh --ctid $CTID

To remove:
  ./install.sh --ctid $CTID --remove
============================================================
DONE
