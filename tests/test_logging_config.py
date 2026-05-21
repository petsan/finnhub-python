"""Tests for the structured-logging configuration helper."""

from __future__ import annotations

import io
import json
import logging

import pytest

from finn_predictor.logging_config import (
    DEFAULT_FORMAT,
    _resolve_format,
    _resolve_level,
    setup_logging,
)


def test_resolve_format_defaults_to_text() -> None:
    assert _resolve_format({}) == "text"


def test_resolve_format_picks_json_when_set() -> None:
    assert _resolve_format({"FINN_PREDICTOR_LOG_FORMAT": "json"}) == "json"
    assert _resolve_format({"FINN_PREDICTOR_LOG_FORMAT": "JSON"}) == "json"


def test_resolve_format_unknown_falls_back_to_text() -> None:
    assert _resolve_format({"FINN_PREDICTOR_LOG_FORMAT": "xml"}) == "text"


def test_resolve_level_defaults_to_info() -> None:
    assert _resolve_level({}) == logging.INFO


def test_resolve_level_reads_env() -> None:
    assert _resolve_level({"FINN_PREDICTOR_LOG_LEVEL": "DEBUG"}) == logging.DEBUG
    assert _resolve_level({"FINN_PREDICTOR_LOG_LEVEL": "warning"}) == logging.WARNING


def test_setup_logging_json_format_emits_json(capsys) -> None:
    setup_logging(fmt="json", level=logging.INFO, force=True)
    try:
        logging.getLogger("test_logging").info("hello world")
        out = capsys.readouterr().out.strip().splitlines()
        assert out, "expected at least one log line"
        payload = json.loads(out[-1])
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test_logging"
        assert "ts" in payload
    finally:
        # Reset so subsequent tests aren't surprised.
        setup_logging(fmt="text", level=logging.WARNING, force=True)


def test_setup_logging_text_format_is_human_readable(capsys) -> None:
    setup_logging(fmt="text", level=logging.INFO, force=True)
    try:
        logging.getLogger("test_logging").info("hello again")
        out = capsys.readouterr().out
        assert "hello again" in out
        assert "INFO" in out
        # Should NOT look like JSON.
        assert not out.strip().startswith("{")
    finally:
        setup_logging(fmt="text", level=logging.WARNING, force=True)


def test_setup_logging_scrubs_token_through_handler(capsys) -> None:
    """The SecretScrubFilter is attached to every handler set up here."""
    setup_logging(fmt="text", level=logging.INFO, force=True)
    try:
        tok = "d867phhr01qnvdehc6ggd867phhr01qnvdehc6h0"
        logging.getLogger("test_logging").info(f"bad url ?token={tok}")
        out = capsys.readouterr().out
        assert tok not in out
        assert "<REDACTED-TOKEN>" in out
    finally:
        setup_logging(fmt="text", level=logging.WARNING, force=True)


def test_setup_logging_idempotent_without_force(capsys) -> None:
    """Library mode (force=False) doesn't trample existing handlers."""
    setup_logging(fmt="text", level=logging.INFO, force=True)
    root = logging.getLogger()
    n_before = len(root.handlers)
    setup_logging(fmt="json", level=logging.DEBUG, force=False)
    n_after = len(root.handlers)
    assert n_before == n_after  # untouched
