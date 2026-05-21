"""Streamlit AppTest smoke tests for the dashboard.

These are coarse — they boot the actual ``finn_predictor.ui.app`` script
inside Streamlit's headless test runner and assert that the key page
chrome lands. They're not unit tests of the helpers (that's
``test_ui_helpers``); they catch import-time regressions, blank-page
states from missing fixtures, and tab-level rendering breakage.

Each test points the app at a per-test SQLite file via the
``FINN_PREDICTOR_DB_URL`` env var so the live SQLite default never gets
touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# AppTest landed in streamlit 1.28 and is API-stable as of 1.57. Skip the
# whole module if the host streamlit is too old or the import fails
# entirely (CI may run without the optional package).
streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


APP_PATH = Path(__file__).parent.parent / "finn_predictor" / "ui" / "app.py"


def _seed_predictions(db_path: Path) -> None:
    """Create a small fixture in ``db_path``: 1 market + 1 sector prediction.

    Enough state that the *Today* and *Sectors* tabs render something
    non-trivial; bare enough that initialisation is fast.
    """
    from finn_predictor.storage import create_engine_and_session, init_db
    from finn_predictor.storage.models import Prediction, Sector

    engine, SL = create_engine_and_session(f"sqlite:///{db_path}")
    init_db(engine)
    now = datetime.now(timezone.utc)
    with SL() as s:
        s.add(Sector(code="TECH", name="Information Technology", etf_symbol="XLK"))
        s.add(
            Prediction(
                target_symbol="^GSPC",
                prediction_date=now - timedelta(hours=1),
                label="UP",
                confidence=0.42,
                sentiment_index=0.18,
                article_count=12,
                model_version="vader-3.3.2",
            )
        )
        s.add(
            Prediction(
                target_symbol="XLK",
                prediction_date=now - timedelta(hours=1),
                label="DOWN",
                confidence=0.31,
                sentiment_index=-0.09,
                article_count=8,
                model_version="vader-3.3.2",
            )
        )
        s.commit()
    engine.dispose()


def _boot(monkeypatch, tmp_path: Path, *, seed: bool = True):
    """Helper: point the app at a fresh SQLite file and boot AppTest."""
    db_path = tmp_path / "ui.db"
    # The auth gate is disabled when FINN_PREDICTOR_PASSWORD_HASH is
    # unset, so the AppTest run reaches the main tab strip directly.
    monkeypatch.delenv("FINN_PREDICTOR_PASSWORD_HASH", raising=False)
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{db_path}")
    # Disable the localStorage bridge — its frontend component polls
    # session_state forever in AppTest mode (the JS that posts back
    # never runs), which would hang the boot. Production runs do not
    # set this and get the persistence behaviour for free.
    monkeypatch.setenv("FINN_PREDICTOR_DISABLE_LOCAL_STORAGE", "1")
    if seed:
        _seed_predictions(db_path)

    at = AppTest.from_file(str(APP_PATH), default_timeout=20)
    at.run()
    return at


def test_app_boots_clean_with_seeded_db(monkeypatch, tmp_path) -> None:
    """Headless boot lands without exceptions and shows the page title."""
    at = _boot(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    # The header is set via st.title("Finn-Predictor") in the locked +
    # unlocked branches. We don't care which one we land in — just that
    # *some* title with "Finn-Predictor" rendered.
    titles = [t.value for t in at.title]
    assert any("Finn-Predictor" in t for t in titles), titles


def test_app_boots_with_empty_db(monkeypatch, tmp_path) -> None:
    """Without seeded predictions the *Today* tab shows the bootstrap nudge."""
    at = _boot(monkeypatch, tmp_path, seed=False)
    assert not at.exception, [e.value for e in at.exception]
    # The bootstrap copy lives in the *Today* tab — its presence
    # confirms the tab strip rendered AND the empty-state branch ran.
    blob = " ".join(
        (m.value for m in at.info)
        if hasattr(at, "info") and at.info
        else (
            getattr(b, "value", "") for b in at.markdown
        )
    )
    assert "Run ingestion now" in blob or "API key" in blob.lower(), blob[:500]


def test_app_renders_tab_strip(monkeypatch, tmp_path) -> None:
    """All six tab labels are present somewhere in the rendered tree."""
    at = _boot(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]

    # AppTest exposes tabs via .tabs (list of Tab nodes, each with .label).
    tab_labels = {t.label for t in at.tabs}
    expected = {"Today", "History", "Sectors", "Performance", "Focus", "Learning"}
    assert expected <= tab_labels, tab_labels


def test_app_today_tab_surfaces_seeded_prediction(monkeypatch, tmp_path) -> None:
    """The seeded market UP call surfaces somewhere on the page.

    AppTest doesn't simulate clicks on st.tabs (tabs render in parallel),
    so the prediction's label/confidence/article_count metrics all land
    in the rendered tree on the initial boot.
    """
    at = _boot(monkeypatch, tmp_path)
    assert not at.exception, [e.value for e in at.exception]

    metric_values = [
        getattr(m, "value", "") for m in at.metric
    ]
    metric_labels = [getattr(m, "label", "") for m in at.metric]
    # The Today tab renders three metrics: Call, Confidence, Articles.
    assert "Call" in metric_labels
    assert "UP" in metric_values
