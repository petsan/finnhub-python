# Finn-Predictor — Security Audit

**Status:** First pass — static analysis + dependency review.
**Date:** 2026-05-21.
**Scope:** Everything under `finn_predictor/`, the Dockerfile + compose
file, `docker/entrypoint.sh`, `deploy/proxmox/install.sh`, the upstream
`finnhub/` library (read-only — no changes there), and the runtime
dependencies in `requirements.txt`.
**Out of scope (this pass):** Network penetration testing, fuzzing,
dynamic analysis under load, social-engineering / phishing, and the
Finnhub.io upstream itself.
**Methodology:** Read every Python module + container/install artifact;
grep for dangerous primitives (`subprocess`, `os.system`, `shell=True`,
`eval`, `exec`, `pickle`, `yaml.load`, raw SQL, `verify=False`); inspect
authn/authz; verify secret handling against the comment claims; sample
the SQLAlchemy call sites for parameter binding; review container user
+ filesystem + healthcheck setup; review the systemd unit produced by
the Proxmox installer.

---

## TL;DR

The project takes secret handling seriously — the API token has **four**
independent scrubbing layers and never touches disk or the DB, the
container runs as a non-root user, all SQLAlchemy calls use ORM/Core
parameter binding, and there is **zero** use of `subprocess`,
`os.system`, `shell=True`, `eval`, `exec`, `pickle`, or `yaml.load`
inside `finn_predictor/`. The auth gate is bcrypt with cost 12 and a
constant-time compare.

What it lacks for a public-internet production deploy:

1. **No transport security in-app.** Streamlit listens on plain HTTP
   on `0.0.0.0:8501`. TLS is delegated to a reverse proxy that isn't
   provisioned by either deployment path.
2. **No CSRF / clickjacking / CSP headers.** Streamlit's defaults are
   permissive; a hostile origin can frame the app or replay form posts
   from a logged-in browser session.
3. **No per-user authn.** The optional password gate is a single
   shared secret with no user records, no rate limiting, no lockout,
   no audit trail.
4. **API key in browser localStorage by default.** Survives Ctrl+R as
   designed — but any XSS in the Streamlit page exfiltrates it. Mitigated
   by the env-var opt-out (`FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1`) and
   by the absence of user-supplied HTML rendering, but the risk class is
   broader than the current attack surface.
5. **Dependency CVE check has never been run in CI.** `pip-audit` is
   already a pinned dev dep but isn't part of the test workflow; this
   audit could not run it (sandbox blocks `pypi.org`).
6. **Container is `latest`-tagged and unpinned.** A `docker compose
   pull` will silently replace the image with whatever was last pushed.
7. **Proxmox installer writes the API key into a systemd unit file
   on disk** — its own header calls this out, but the deployment
   manual doesn't surface the trade-off prominently.

None of the above are exploit primitives on a localhost dev box. They
become real risks the day this is exposed to the public internet.

---

## Severity legend

- **Critical** — confidentiality/integrity loss without user interaction.
- **High** — exposes secrets, allows unauthenticated state mutation,
  or breaks the trust boundary the docs promise.
- **Medium** — defence-in-depth gap; needs a chained weakness or attacker
  presence to exploit.
- **Low** — hardening opportunity; documents stay honest, but no realistic
  exploit on supported deploys.
- **Info** — observation, no action required (or already mitigated).

---

## Inventory: what the codebase already does well

| # | Control | Evidence |
|---|---|---|
| ✅ | API token never written to disk or DB | `finn_predictor/ui/app.py:6-11, 120-124`; no DB column persists it; localStorage is opt-out via `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE` |
| ✅ | Token scrubbed at gateway-error site | `finn_predictor/ingestion/client.py:43-51, 120-161` (`scrub_token`, `_scrub`, `IngestionError`) |
| ✅ | Token scrubbed at UI-display site | `finn_predictor/ui/app.py:1434` (UI catches and scrubs ingestion errors before `st.error`) |
| ✅ | Regex-based fallback scrub on every log record | `finn_predictor/security.py:91-118` (`SecretScrubFilter` on root logger) |
| ✅ | Trust-env disabled to defeat HTTPS_PROXY MITM during dev | `Client._session.trust_env = False` (verified in upstream `finnhub/client.py`) |
| ✅ | bcrypt cost 12 + constant-time compare | `finn_predictor/security.py:35-67` (`hash_password`, `verify_password`) |
| ✅ | Auth gate clears typed password from `session_state` post-verify | `finn_predictor/ui/app.py:262-267` |
| ✅ | Auth-disabled is a deliberate default for localhost only | `finn_predictor/security.py:70-79` (`auth_enabled` honours empty env var) |
| ✅ | Non-root container user (`UID 1001`, no shell, no home) | `Dockerfile:56-58` |
| ✅ | `pip install --no-cache-dir` + slim base | `Dockerfile:14-17` |
| ✅ | Docker healthcheck (lets `depends_on: service_healthy` work) | `Dockerfile:63-64`, `docker-compose.yml:35-40` |
| ✅ | Restrictive `.dockerignore` (no .git, no .venv, no host DB) | `.dockerignore:1-37` |
| ✅ | No raw SQL anywhere; all `session.execute(stmt)` calls use SQLAlchemy ORM/Core constructs | `finn_predictor/storage/repo.py:101, 184, 242` (every site is `select()`/`insert()` from sqlalchemy) |
| ✅ | No `subprocess`, `os.system`, `shell=True`, `eval`, `exec`, `pickle`, `yaml.load` in `finn_predictor/` | Grep returns zero matches |
| ✅ | Rate-limited Finnhub egress (55 calls/min, headroom under free tier 60) | `finn_predictor/ingestion/client.py:54-92` (`RateLimiter`); `finn_predictor/config.py:39` (default `rate_limit_per_minute=55`) |
| ✅ | Retry uses bounded exponential backoff on transient classes only | `finn_predictor/ingestion/client.py:123-150` (`RETRYABLE_STATUS = {429, 500, 502, 503, 504}`, max 3 retries) |
| ✅ | Settings is a frozen `dataclass(frozen=True)` — no late mutation | `finn_predictor/config.py:20-40` |
| ✅ | All UTC, all timezone-aware on the DB layer | `finn_predictor/storage/models.py:35-37` (`_utcnow`); every DateTime column has `timezone=True` |
| ✅ | Streamlit usage stats disabled in container | `docker/entrypoint.sh:50` (`--browser.gatherUsageStats false`) |
| ✅ | Proxmox LXC is **unprivileged** by design | `deploy/proxmox/install.sh` header (lines 9-15) |

---

## Findings

### F-01 — TLS terminates outside the app, not provisioned by either deployment path

**Severity:** High (when exposed beyond localhost).
**Status:** Documented, not implemented.

Streamlit is started with `--server.address 0.0.0.0` on port 8501,
plain HTTP. The Dockerfile, docker-compose, and Proxmox installer all
emit a plain-HTTP service. `deployment-manual.md` mentions a reverse
proxy + TLS, but the artifact that would actually provision it
(Caddyfile / nginx site / Traefik label) isn't shipped.

**Impact:** Browser → server traffic is in the clear. Anyone on the
same LAN sees the password gate exchange, the rendered ticker list,
and (if `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1` is **not** set) every
re-hydrated request that carries the Finnhub key in component
postbacks.

**Recommendations:**
1. Ship a `deploy/caddy/Caddyfile.example` and a `docker-compose.tls.yml`
   override that sits Caddy in front of Streamlit with automatic Let's
   Encrypt. Reverse-proxy mode + HSTS + HTTP→HTTPS upgrade.
2. Add a `FINN_PREDICTOR_REQUIRE_TLS=1` env var that makes the app
   refuse to start if `X-Forwarded-Proto != https` (or if it can't
   detect a proxy) on requests from non-loopback origins.
3. Document the trade-off in the installation manual: "the localhost
   default is HTTP because development requires it; the production
   default must be HTTPS via a reverse proxy."

---

### F-02 — No CSRF / clickjacking / CSP protection on the Streamlit page

**Severity:** Medium.
**Status:** Not present.

Streamlit does not emit `X-Frame-Options`, `Content-Security-Policy`,
`X-Content-Type-Options`, or `Referrer-Policy` headers by default. The
WebSocket-based interaction model means classic CSRF is muted, but the
form-submission path Streamlit components use is not. A malicious site
that frames the app could still capture clicks and reads from a
logged-in browser.

**Impact:** A targeted attacker who tricks a logged-in operator into
visiting a hostile page could iframe the dashboard and read its
rendered state, including the unhidden Finnhub key (only the **input**
field is `type="password"`; the rest of the UI doesn't carry secrets
but a future feature might).

**Recommendations:**
1. Add the reverse-proxy snippet from F-01 with these headers baked in:
   `Strict-Transport-Security: max-age=31536000; includeSubDomains`,
   `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
   `Referrer-Policy: no-referrer`, and a CSP that allows
   `'self'` plus the small set of CDNs Streamlit pulls (we should
   measure these, not guess).
2. Audit any future `st.html` / `st.components.v1.html` usage — those
   surfaces are arbitrary-content injection points. Today, the
   codebase uses neither (grep returns no hits).

---

### F-03 — API key in browser localStorage is exfiltrated by any XSS in Streamlit components

**Severity:** Medium (today) → High (if any user-content rendering is added).
**Status:** Opt-out env var exists; default-on is the deliberate UX choice.

The sidebar persists the Finnhub token to `window.localStorage` under
the key `finn_predictor_finnhub_api_key` so the field survives
`Ctrl+R`. Server-side, the token still never lands on disk or in the
DB — but any JavaScript executing in the page origin can read it.

Today the attack surface is small: Streamlit renders Markdown via its
own renderer (Markdown injection is sanitised), the codebase doesn't
use `st.components.v1.html`, and there is no user-supplied HTML in any
rendered headline or summary (we display `headline`, `summary`, and
`source` as plain text). But the moment a contributor adds an
unescaped string render — say, an HTML email of a prediction summary,
or an admin tool that displays raw article HTML — the key is gone.

**Recommendations:**
1. Make `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1` the **default for the
   container/deploy paths**, and keep on-by-default only for the bare
   `streamlit run` dev case. The Proxmox installer + Dockerfile should
   set `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1` in their env unless the
   operator explicitly opts in.
2. Add a sentinel test that grep-fails CI if `st.html(`,
   `st.components.v1.html(`, or `unsafe_allow_html=True` ever appear in
   `finn_predictor/ui/`.
3. Long-term: move to a server-side session store (encrypted at rest,
   server-only) keyed by an HttpOnly+SameSite=Strict cookie. The
   current "API key is per-tab, owned by the operator" UX can stay; the
   storage substrate moves out of the browser.

---

### F-04 — Single-shared-secret auth gate has no rate limiting, no lockout, no audit

**Severity:** Medium.
**Status:** Bcrypt is correct but the surrounding controls are missing.

The `_enforce_auth_gate` function (`finn_predictor/ui/app.py:236-275`)
implements a single bcrypt-compare per submission with no:

- per-IP throttle,
- failed-attempt counter,
- lockout,
- audit log entry on success or failure.

Streamlit's default rerun behaviour means the bcrypt verify happens on
the server's main thread for every wrong attempt. Cost factor 12 is
~150ms — fast enough that a determined attacker on a fast LAN can
mount a credential-stuffing attack against the single shared password.

**Impact:** On a network-exposed deploy with a weak shared password,
the gate can be brute-forced in days, not centuries.

**Recommendations:**
1. Add a per-IP failure counter in memory (token-bucket; 5 attempts
   per 10 minutes per remote address). Streamlit's `st.context.headers`
   gives access to `X-Forwarded-For` when a reverse proxy is in front.
2. Log every successful + failed authentication attempt with the
   client IP (scrubbed via the existing `SecretScrubFilter`). Today
   neither outcome is logged.
3. Document that the auth gate is a **first line, not the only line**:
   pair with a reverse proxy that does mTLS, basic-auth, or
   network-layer ACL on the management endpoint.
4. Surface in the user manual: "this is one shared secret. Multiple
   operators must accept that they cannot distinguish each other's
   actions in the logs."

---

### F-05 — Docker image is `latest`-tagged, base is `python:3.12-slim` (also unpinned)

**Severity:** Medium.
**Status:** Documented gap.

`Dockerfile:9` uses `FROM python:3.12-slim` (no digest, no patch
version), and the produced image is tagged `finn-predictor:latest`. A
future `docker compose pull` silently swaps the base image, which means
the build is not reproducible and any rollback path goes through "find
the old image hash in your local cache."

**Impact:** Supply-chain integrity — a compromised upstream image or a
silent base-image vulnerability appears in your deploy without a code
change.

**Recommendations:**
1. Pin the base image by digest:
   `FROM python:3.12.7-slim@sha256:<digest>`. Refresh on a cadence with
   a `make refresh-base` target that explicitly updates the digest in
   the Dockerfile.
2. Tag the produced image with the git commit SHA (or semantic
   version) and treat `latest` as an alias produced only after a green
   CI run.
3. Add a `docker scout cves` (or Trivy) step to CI before pushing.

---

### F-06 — `pip-audit` exists locally but is not in CI

**Severity:** Medium.
**Status:** Tool present; no automation.

`pip-audit` is now installed locally (this audit attempted to run it;
the sandbox blocks `pypi.org` TLS). Neither `.travis.yml`,
`.gitlab-ci.yml`, nor any GitHub Actions workflow runs it. The
project's CVE posture is therefore "whatever was current the day a
contributor last looked."

**Recommendations:**
1. Add to CI:
   ```yaml
   - name: Audit Python dependencies
     run: |
       pip install pip-audit
       pip-audit --requirement requirements.txt --strict
   ```
2. Run weekly on a cron, with the result posted as an issue if it
   fails (so we notice CVEs in deps even between commits).
3. Run against `requirements.txt` *and* the resolved set from a
   `pip install -e .` to catch transitive vulns.

---

### F-07 — Proxmox installer writes `FINNHUB_API_KEY` into a systemd unit on disk

**Severity:** Medium.
**Status:** Documented in the installer header; not surfaced in deployment manual.

`deploy/proxmox/install.sh` accepts an optional `FINNHUB_API_KEY` env
var and embeds it in the systemd service's `Environment=` directive.
Root on the container can `cat /etc/systemd/system/finn-predictor.service`
and recover the key, and the file is backed up by Proxmox's default
container snapshots. Compare to the localhost UX (key lives only in
`st.session_state`), and the trust model is meaningfully different.

**Impact:** A compromised LXC root (or anyone who can read snapshot
images at the Proxmox level) trivially recovers the Finnhub key.

**Recommendations:**
1. Switch the installer to write the key into a root-owned, mode-0600
   `/etc/finn-predictor/env` file, and source it via `EnvironmentFile=`
   in the systemd unit. Same on-disk problem, but at least readable
   only by `root` + the service user. Plus the snapshot footprint
   shrinks from "in the service file the kernel logs" to "in one
   explicitly-marked secrets file."
2. Document the alternative: don't bake the key in at all — let the
   UI's session-only flow handle it. The installer should print a
   notice at the end describing both options.
3. For real production: use a secrets store (HashiCorp Vault, AWS
   Secrets Manager, age-encrypted file with a systemd `LoadCredential`)
   instead of an env var on disk.

---

### F-08 — No request-size / rate limit on the Streamlit endpoint

**Severity:** Medium.
**Status:** **Partially resolved 2026-05-21** in PR-1 — user-supplied
ticker input is now capped at 50 tickers and 8 KiB (in
`finn_predictor.ingestion.symbols.parse_ticker_list`), and malformed
tokens are rejected with per-token reasons. **Still open:** per-session
"next-allowed-run-at" throttle on the ingestion + backfill buttons,
and per-IP rate limit on the auth gate (see F-04). Tracked for PR-3.

The Streamlit server happily accepts arbitrary WebSocket frames. There
is no application-level cap on:

- size of pasted ticker list (the sidebar comma-separated input),
- frequency of "Run ingestion now" / "Backfill" button clicks,
- frequency of password-submit attempts (see F-04).

A bored attacker who has password access can hammer the ingestion
button and exhaust the rate-limited Finnhub budget for the rest of the
hour, making the dashboard unusable for legitimate operators.

**Recommendations:**
1. Add a per-session "next allowed run-at" timestamp in `session_state`
   for ingestion + backfill (one minute apart is generous).
2. Cap the parsed ticker list at, say, 50 symbols and reject pasted
   payloads larger than 8 KiB. The current ingestion loop will happily
   try to fetch 5000 tickers if someone pastes them in.
3. Cap backfill lookback at the documented max (≤ 1 year) by validating
   the input rather than relying on Finnhub returning empty.

---

### F-09 — Container has `build-essential` + `curl` baked into the runtime image

**Severity:** Low.
**Status:** Tracked.

`Dockerfile:21-26` installs `build-essential`, `ca-certificates`, and
`curl`. `build-essential` is there in case a wheel has to be compiled
from sdist (the comment calls this out); `curl` is only used by the
healthcheck.

**Impact:** Larger image, larger attack surface inside the container
(e.g. a foothold can compile follow-up tools instead of having to drop
them in).

**Recommendations:**
1. Use a multi-stage build: `python:3.12.7-slim AS builder` installs
   `build-essential` and builds wheels; the runtime stage uses a stock
   `python:3.12.7-slim` and only `COPY --from=builder` the resulting
   wheels + the app source.
2. Switch the healthcheck to a `python -c` one-liner (`urllib.request.urlopen`)
   and drop `curl` entirely.

---

### F-10 — DB file is world-readable inside the container's volume mount

**Severity:** Low.
**Status:** Implicit, not deliberately set.

The named volume `finn_data` is created with default permissions. The
SQLite DB ends up owned by `finn:finn` (per the `chown` in the
Dockerfile) and mode `0644` after SQLAlchemy writes it. Any other
process running as a different UID in the same container shares the
namespace — today there's only one process, but the design budget
hasn't been spent yet.

**Impact:** Defence-in-depth gap only — the DB doesn't store the API
key, but it does store article URLs, prediction confidences, and the
list of tickers an operator cares about. That last one is mildly
sensitive (it's a watchlist).

**Recommendations:**
1. Set `umask 0027` in the entrypoint so future-created files default
   to 0640.
2. Document explicitly: the SQLite DB is **not** an encrypted store.
   Don't put PII or credentials in `app_settings` "just in case."

---

### F-11 — Logging filter is regex-based; misses non-standard token shapes

**Severity:** Low.
**Status:** Mitigated three other ways.

`SecretScrubFilter` (`finn_predictor/security.py:91-118`) matches
40-character lowercase-hex tokens (`\b[0-9a-z]{40}\b`) and bcrypt
hashes. If Finnhub ever changes their token format, the filter goes
silent. Today this is fine — three earlier scrubbing layers catch the
token before the filter sees it.

**Recommendations:**
1. Add an env-var override (`FINN_PREDICTOR_TOKEN_REGEX`) so an
   operator can broaden the filter without a code change if the shape
   changes upstream.
2. Add a "canary" log line in a test that emits a fake token via
   `logger.info` and asserts it ends up redacted — guards against the
   regex silently breaking after a refactor. (One such test exists for
   the gateway scrub but not for the logging filter end-to-end.)

---

### F-12 — Streamlit runtime is `--browser.gatherUsageStats false` in container but not in local `streamlit run`

**Severity:** Low (privacy hygiene).
**Status:** Container-only.

`docker/entrypoint.sh:46-50` disables Streamlit's analytics ping. A
developer running `streamlit run finn_predictor/ui/app.py` locally
does not get the same default — Streamlit's first-run flow may ping
home with project metadata.

**Recommendations:**
1. Ship a `.streamlit/config.toml` with
   `[browser] gatherUsageStats = false` so the local-dev path matches
   the container.
2. Document in the README that the project does not phone home and
   the config makes that explicit.

---

### F-13 — Upstream `finnhub-python` library has no input validation on user-supplied symbols

**Severity:** Low.
**Status:** **Resolved 2026-05-21** in PR-1 — every UI-supplied symbol now passes
`finn_predictor.ingestion.symbols.valid_ticker` (regex `[A-Z0-9.^-]{1,16}`
via `re.fullmatch`) before reaching `FinnhubGateway`. Validator at 100%
line+branch test coverage. The upstream library itself is unchanged.

The upstream `finnhub.Client` builds URLs from user-supplied symbols
without escaping. Today this is fine — every call goes via
`requests.Session(params=...)` which URL-encodes the query string for
us — but the library has no allow-list / shape check on the symbol.
A future endpoint that interpolates symbols into the URL path instead
of the query string could SSRF or break out of the API namespace.

**Recommendations:**
1. Add a `_validate_symbol(s)` helper in `finn_predictor/ingestion/client.py`
   that bounds-checks user-supplied tickers (e.g. `^[A-Z0-9.^-]{1,16}$`)
   before they reach the upstream client. Defence in depth.

---

### F-14 — No subresource integrity / pinning on Streamlit's CDN-loaded assets

**Severity:** Low.
**Status:** Inherited from Streamlit's design.

Streamlit pulls some assets from `unpkg.com` / its own CDN. We don't
control these. A compromised Streamlit-CDN edge could push hostile JS
into our page.

**Recommendations:**
1. Document the dependency posture in `security.md` (this file) and
   list it as an accepted risk — there's no clean fix short of forking
   Streamlit.
2. The CSP recommended in F-02 limits the blast radius by allow-listing
   only the CDN hostnames we observe in production, so a hijacked
   alternate domain can't load script into our origin.

---

### F-15 — No SBOM produced at build time

**Severity:** Low.
**Status:** Not present.

A reproducible deploy includes a software bill of materials. Today,
the container is a black box: `docker image inspect` shows layers but
not "this build pulled requests 2.34.2."

**Recommendations:**
1. Add a `make sbom` target using `cyclonedx-py` (or `syft`) that
   produces `sbom.json` alongside the image.
2. Ship the SBOM as a release artifact. Optional but cheap.

---

### F-16 — `.travis.yml` and `.gitlab-ci.yml` are present but stale; coverage is the only enforced gate

**Severity:** Info / process gap.
**Status:** Tracked.

The CI configs in the repo predate the Finn-Predictor work and don't
run `pip-audit`, `bandit`, `ruff`, `mypy --strict`, or any
container-image scan.

**Recommendations:**
1. Move to a single GitHub Actions workflow (the file is `.github/`
   already exists). Steps in order:
   `ruff check` → `mypy --strict finn_predictor/` →
   `pytest --cov` (current gate) → `pip-audit` (F-06) →
   `bandit -ll -r finn_predictor/` (catches the `subprocess`/`pickle`
   accidents that grep would miss in a refactor) →
   `docker build` → `trivy image` → release.

---

### F-17 — No protection against Streamlit `session_state` cross-tab key leak under shared cookie

**Severity:** Info.
**Status:** Out of normal usage.

Streamlit's session is keyed by a browser session cookie. Two tabs in
the same browser share `session_state`. If one tab pastes the Finnhub
key and a second tab logs into a different account on the same shared
machine, the second tab inherits the first tab's key.

**Recommendations:**
1. Mostly a documentation issue — surface in `user-manual.md`: "do not
   share the auth-gated dashboard between accounts on the same browser
   profile." Already true; not currently stated.

---

## Threat model (assets, actors, surfaces)

### Assets

- **Finnhub API token** (per-user). Highest-value secret. Capable of
  exhausting the user's free-tier quota or, on paid tiers, racking up
  per-request costs.
- **Operator passwords** (bcrypt hashes). Stored only in `FINN_PREDICTOR_PASSWORD_HASH`
  env var, never in DB.
- **Watchlist + prediction history**. Mildly sensitive — leaks an
  operator's market interest. Lives in SQLite (or Postgres in that
  deployment).
- **Learned model weights**. The Bayesian-trained source weights +
  thresholds. Reveals which news sources the operator finds high-signal;
  also implicitly reveals the closed-outcome history that fit them.

### Actors

- **Localhost user** (default). Already has shell on the host — model
  trusts them. No additional controls beyond OS file perms.
- **LAN observer** (one hop away). Sees plaintext HTTP without F-01's
  fix. Trust model: must not see secrets in transit.
- **Remote attacker over the public internet**. Should be unable to
  observe traffic (F-01), brute-force the password gate (F-04), or
  recover the API token from a misconfigured iframe (F-02 / F-03).
- **Supply-chain attacker**. Compromises a PyPI or Docker-Hub artifact
  we depend on (F-05, F-06, F-15).
- **Insider with LXC/container access**. Reads `FINNHUB_API_KEY` from
  the systemd unit (F-07) or sniffs `/proc/<pid>/environ`.

### Surfaces

| Surface | Trust boundary | Current control | Gap |
|---|---|---|---|
| Browser ↔ Streamlit WebSocket | Outside-host → app | None (TLS via proxy is out-of-tree) | F-01, F-02 |
| Browser localStorage | Inside-browser | Opt-out env var; no XSS path today | F-03 |
| Streamlit auth gate | Outside → app | bcrypt, constant-time compare | F-04 |
| Container env vars | Host → container | UID 1001 read-only-ish | F-07 |
| Finnhub egress | App → upstream | Rate-limited, retry-bounded, scrubbed | (none — solid) |
| SQLite DB | Container FS | ORM-only access, no raw SQL | F-10 |
| Logs | App → stdout / json sink | 4-layer token scrub | F-11 |
| Image registry | CI → runtime | Pulled fresh each build | F-05 |

---

## Hardening roadmap (suggested order)

1. **Now** (no code, just docs): document the trust model precisely in
   `installation-manual.md` and `deployment-manual.md` — what is
   localhost-only, what is LAN-acceptable, what needs TLS + auth gate.
2. **Next sprint** (config / Docker): F-01 (Caddy compose override),
   F-02 (security headers in proxy), F-05 (pin base by digest),
   F-12 (`.streamlit/config.toml`).
3. **Then** (application code): F-03 (default-off localStorage in
   container), F-04 (rate limit + audit log on auth gate),
   F-08 (input caps + per-session throttle), F-13 (symbol validator).
4. **Continuously**: F-06 (pip-audit in CI), F-16 (full CI rebuild +
   bandit + ruff + mypy + trivy), F-15 (SBOM).
5. **When the deploy story matures**: F-07 (proper secrets store),
   F-10 (umask in entrypoint), F-09 (multi-stage Dockerfile).

Each item on the roadmap is small enough to ship in one PR with tests
that exercise the new control (e.g. a Streamlit smoke test that asserts
the security-headers list, a unit test that asserts the symbol
validator rejects garbage, an integration test that asserts the
auth-gate rate limit blocks after N failures).

---

## Re-audit cadence

- Re-run `pip-audit` weekly via CI (F-06).
- Re-read this file every release; any new finding gets a new entry
  with its own F-NN ID, and any addressed finding gets a "**Resolved
  in 2026-MM-DD commit <sha>**" stamp instead of being deleted.
- Add a `security.md` review to the PR template so changes that touch
  ingestion, auth, or logging force a fresh look at the relevant
  findings.
