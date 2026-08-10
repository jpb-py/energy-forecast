import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from energy_forecast.mia import tools as mia_tools
from energy_forecast.mia.agent import run_query
from energy_forecast.mia.registry import Session


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, input_):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_)


def _response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def _scripted_client(responses):
    """A mock client whose messages.create returns each response in order and
    records a *snapshot* of the messages list at call time. run_query mutates
    the same list object in place across iterations, so inspecting recorded
    call_args after the loop ends would otherwise only ever show the final
    state -- snapshotting with list(...) at call time avoids that trap."""
    remaining = list(responses)
    calls = []

    def fake_create(**kwargs):
        calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return remaining.pop(0)

    client = MagicMock()
    client.messages.create.side_effect = fake_create
    return client, calls


@pytest.fixture
def session(synthetic_raw_demand):
    return Session(raw_data=synthetic_raw_demand(14), total_calls_budget=3)


def test_run_query_returns_text_on_end_turn(session):
    client, calls = _scripted_client([_response([_text_block("Final answer.")], "end_turn")])

    answer = run_query(client, session, "What's the MAE?")

    assert answer == "Final answer."
    assert len(calls) == 1


def test_run_query_dispatches_tool_call_and_feeds_result_back(session, monkeypatch):
    handler_calls = []

    def fake_handler(sess, args):
        handler_calls.append(args)
        return {"mae": 1.23}

    monkeypatch.setitem(mia_tools.TOOL_HANDLERS, "run_backtest", fake_handler)

    client, calls = _scripted_client(
        [
            _response(
                [_tool_use_block("call_1", "run_backtest", {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"})],
                "tool_use",
            ),
            _response([_text_block("Done.")], "end_turn"),
        ]
    )

    answer = run_query(client, session, "Backtest the naive model.")

    assert answer == "Done."
    assert handler_calls == [{"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}]
    assert session.total_calls_used == 1

    tool_result_message = calls[1]["messages"][-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["type"] == "tool_result"
    assert json.loads(tool_result_message["content"][0]["content"]) == {"mae": 1.23}


def test_run_query_surfaces_handler_exception_as_tool_error(session, monkeypatch):
    def failing_handler(sess, args):
        raise ValueError("boom")

    monkeypatch.setitem(mia_tools.TOOL_HANDLERS, "run_backtest", failing_handler)

    client, calls = _scripted_client(
        [
            _response(
                [_tool_use_block("call_1", "run_backtest", {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"})],
                "tool_use",
            ),
            _response([_text_block("Hit an error, here's what I know.")], "end_turn"),
        ]
    )

    answer = run_query(client, session, "Backtest the naive model.")

    assert answer == "Hit an error, here's what I know."
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "boom" in tool_result["content"]


def test_run_query_enforces_total_call_budget(session):
    session.total_calls_budget = 1

    client, calls = _scripted_client(
        [
            _response(
                [
                    _tool_use_block("call_1", "run_backtest", {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}),
                    _tool_use_block("call_2", "run_backtest", {"forecast_fn_id": "linear_lagged", "split_date": "2024-01-11"}),
                ],
                "tool_use",
            ),
            _response([_text_block("Budget exhausted, here's what I found.")], "end_turn"),
        ]
    )

    answer = run_query(client, session, "Compare both models on lots of windows.")

    assert answer == "Budget exhausted, here's what I found."
    assert session.total_calls_used == 1

    # second request should have no budget left, so no tools are offered
    assert calls[1]["tools"] == []

    tool_results = calls[1]["messages"][-1]["content"]
    assert tool_results[0].get("is_error") is not True
    assert tool_results[1]["is_error"] is True
