"""Tests for finn_predictor.security."""

from __future__ import annotations

import logging

import pytest

from finn_predictor.security import (
    MAX_PASSWORD_LEN,
    MIN_PASSWORD_LEN,
    SecretScrubFilter,
    auth_enabled,
    current_password_hash,
    hash_password,
    verify_password,
)


# ---------------- password hashing ----------------


def test_hash_password_returns_bcrypt_hash() -> None:
    h = hash_password("hunter2-2026")
    assert h.startswith("$2")  # bcrypt marker
    assert len(h) >= 50


def test_verify_password_round_trip() -> None:
    h = hash_password("hunter2-2026")
    assert verify_password("hunter2-2026", h) is True
    assert verify_password("wrong", h) is False
    assert verify_password("", h) is False


def test_verify_password_safe_with_malformed_hash() -> None:
    assert verify_password("anything", "not-a-bcrypt-hash") is False
    assert verify_password("anything", "") is False


def test_hash_password_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        hash_password("")


def test_hash_password_rejects_overlong_input() -> None:
    with pytest.raises(ValueError):
        hash_password("x" * (MAX_PASSWORD_LEN + 1))


def test_two_hashes_of_same_password_differ() -> None:
    """bcrypt salts each hash so the stored value differs every time."""
    a = hash_password("same")
    b = hash_password("same")
    assert a != b
    assert verify_password("same", a) is True
    assert verify_password("same", b) is True


# ---------------- env helpers ----------------


def test_auth_enabled_reads_env() -> None:
    assert auth_enabled({}) is False
    assert auth_enabled({"FINN_PREDICTOR_PASSWORD_HASH": ""}) is False
    assert auth_enabled({"FINN_PREDICTOR_PASSWORD_HASH": "  "}) is False
    assert auth_enabled({"FINN_PREDICTOR_PASSWORD_HASH": "$2b$..."}) is True


def test_current_password_hash_strips_whitespace() -> None:
    assert current_password_hash({"FINN_PREDICTOR_PASSWORD_HASH": "  abc  "}) == "abc"
    assert current_password_hash({}) == ""


# ---------------- log scrubber ----------------


def _emit_then_capture(caplog, msg: str, *args: object) -> str:
    """Emit a log record through the scrubber, return the rendered message."""
    logger = logging.getLogger("test_security")
    logger.addFilter(SecretScrubFilter())
    with caplog.at_level(logging.INFO, logger="test_security"):
        logger.info(msg, *args)
    return caplog.records[-1].getMessage()


def test_log_scrubber_masks_finnhub_token(caplog) -> None:
    tok = "d867phhr01qnvdehc6ggd867phhr01qnvdehc6h0"
    out = _emit_then_capture(caplog, f"hitting api with token={tok}")
    assert tok not in out
    assert "<REDACTED-TOKEN>" in out


def test_log_scrubber_masks_bcrypt_hash(caplog) -> None:
    h = hash_password("very-secret")
    out = _emit_then_capture(caplog, f"stored hash {h}")
    assert h not in out
    assert "<REDACTED-HASH>" in out


def test_log_scrubber_leaves_normal_text_alone(caplog) -> None:
    out = _emit_then_capture(caplog, "boring log line — nothing secret")
    assert "boring log line" in out
    assert "<REDACTED" not in out


def test_log_scrubber_handles_args_substitution(caplog) -> None:
    """Records with %-style args render correctly post-scrub."""
    tok = "d867phhr01qnvdehc6ggd867phhr01qnvdehc6h0"
    out = _emit_then_capture(caplog, "user %s key %s", "alice", tok)
    assert "alice" in out
    assert tok not in out
