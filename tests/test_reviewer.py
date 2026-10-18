import pytest
from unittest.mock import patch
from agents.reviewer import review_output, should_escalate_to_human

def test_low_confidence_auto_reject():
    r = review_output("describe X", "some output", "", confidence=0.2)
    assert r["approved"] is False
    assert r["rejection_count"] == 1

def test_above_threshold_approved_by_llm():
    with patch("agents.reviewer.llm_call", return_value='{"approved": true, "issues": []}'):
        r = review_output("describe X", "some output", "context", confidence=0.8)
    assert r["approved"] is True
    assert r["rejection_count"] == 0

def test_llm_rejects_unsupported_claim():
    with patch("agents.reviewer.llm_call", return_value='{"approved": false, "issues": ["claim not in source"]}'):
        r = review_output("describe X", "some output", "context", confidence=0.9)
    assert r["approved"] is False
    assert r["rejection_count"] == 1

def test_prior_rejections_accumulate():
    with patch("agents.reviewer.llm_call", return_value='{"approved": false, "issues": ["bad claim"]}'):
        r = review_output("describe X", "some output", "context", confidence=0.9, prior_rejections=1)
    assert r["rejection_count"] == 2

def test_escalate_on_rejection_limit():
    assert should_escalate_to_human(2, 0.9) is True

def test_escalate_on_low_confidence():
    assert should_escalate_to_human(0, 0.29) is True

def test_no_escalate_normal():
    assert should_escalate_to_human(1, 0.8) is False
