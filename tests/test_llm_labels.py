"""Mocked tests for the Bedrock report-label extraction CLI."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("anthropic")
llm_labels = importlib.import_module("scripts.llm_labels")


def _label_payload():
    payload = {
        label: {"state": "not_mentioned", "evidence": ""}
        for label in llm_labels.LABELS
    }
    payload["ACL"] = {"state": "positive", "evidence": "complete ACL tear"}
    return payload


def _response(payload=None, *, text=None, stop_reason="stop"):
    if text is None:
        text = json.dumps(payload if payload is not None else _label_payload())
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=40,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        ),
    )


def _client(create):
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_schema_requires_all_labels_and_closed_state_values():
    schema = llm_labels.build_schema()

    assert schema["required"] == llm_labels.LABELS
    assert set(schema["properties"]) == set(llm_labels.LABELS)
    assert schema["additionalProperties"] is False
    for label_schema in schema["properties"].values():
        assert label_schema["required"] == ["state", "evidence"]
        assert label_schema["properties"]["state"]["enum"] == llm_labels.STATES
        assert label_schema["additionalProperties"] is False


def test_call_maps_structured_response_and_token_usage():
    create = Mock(return_value=_response())

    result = llm_labels.call(
        _client(create), "model-a", "low", "study-1", "ACL torn.", retries=1
    )

    assert result["uid"] == "study-1"
    assert result["model"] == "model-a"
    assert result["effort"] == "low"
    assert result["states"]["ACL"] == "positive"
    assert result["evidence"]["ACL"] == "complete ACL tear"
    assert result["states"]["MCL"] == "not_mentioned"
    assert result["usage"] == {"in": 100, "out": 40, "cache_read": 10, "cache_write": 5}
    request = create.call_args.kwargs
    assert request["model"] == "model-a"
    assert request["messages"][0]["content"] == "<report>\nACL torn.\n</report>"
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["schema"] == llm_labels.SCHEMA


def test_call_returns_refusal_without_retrying():
    create = Mock(return_value=_response(stop_reason="refusal"))

    result = llm_labels.call(_client(create), "model-a", "low", "study-1", "report")

    assert result == {"uid": "study-1", "model": "model-a", "error": "refusal"}
    assert create.call_count == 1


def test_call_retries_transient_connection_errors(monkeypatch):
    class TransientError(Exception):
        pass

    monkeypatch.setattr(llm_labels, "APIConnectionError", TransientError)
    sleeps = []
    monkeypatch.setattr(llm_labels.time, "sleep", sleeps.append)
    create = Mock(side_effect=[TransientError(), TransientError(), _response()])

    result = llm_labels.call(
        _client(create), "model-a", "low", "study-1", "report", retries=3
    )

    assert result["states"]["ACL"] == "positive"
    assert create.call_count == 3
    assert sleeps == [2, 3]


@pytest.mark.parametrize("bad_text", ["not-json", ""])
def test_call_retries_malformed_json_then_reports_exhaustion(monkeypatch, bad_text):
    sleeps = []
    monkeypatch.setattr(llm_labels.time, "sleep", sleeps.append)
    create = Mock(return_value=_response(text=bad_text))

    result = llm_labels.call(
        _client(create), "model-a", "low", "study-1", "report", retries=2
    )

    assert result["error"] == "retries_exhausted"
    assert create.call_count == 2
    assert sleeps == [1, 1]


def test_call_retries_when_response_has_no_text_block(monkeypatch):
    monkeypatch.setattr(llm_labels.time, "sleep", lambda _: None)
    response = _response()
    response.content = []
    create = Mock(return_value=response)

    result = llm_labels.call(
        _client(create), "model-a", "low", "study-1", "report", retries=2
    )

    assert result["error"] == "retries_exhausted"
    assert create.call_count == 2


def test_call_does_not_retry_client_status_errors(monkeypatch):
    class StatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"status {status_code}")

    monkeypatch.setattr(llm_labels, "APIStatusError", StatusError)
    sleeps = []
    monkeypatch.setattr(llm_labels.time, "sleep", sleeps.append)
    create = Mock(side_effect=StatusError(400))

    result = llm_labels.call(_client(create), "model-a", "low", "study-1", "report")

    assert result["error"] == "400: status 400"
    assert create.call_count == 1
    assert sleeps == []


def test_zero_retries_returns_error_without_calling_api():
    create = Mock()

    result = llm_labels.call(
        _client(create), "model-a", "low", "study-1", "report", retries=0
    )

    assert result["error"] == "retries_exhausted"
    create.assert_not_called()
