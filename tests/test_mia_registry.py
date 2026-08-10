import pandas as pd
import pytest

from energy_forecast.mia.registry import (
    FORECAST_FUNCTIONS,
    ResultStore,
    Session,
    resolve_forecast_fn,
)


def test_resolve_forecast_fn_known_id():
    fn = resolve_forecast_fn("linear_lagged")
    assert fn is FORECAST_FUNCTIONS["linear_lagged"]


def test_resolve_forecast_fn_unknown_id_raises():
    with pytest.raises(KeyError):
        resolve_forecast_fn("does_not_exist")


def test_result_store_put_get_roundtrip():
    store = ResultStore()
    obj = pd.DataFrame({"a": [1, 2, 3]})

    object_id = store.put("pred", obj)

    assert store.get(object_id) is obj


def test_result_store_ids_are_unique():
    store = ResultStore()
    obj = pd.DataFrame({"a": [1]})

    id_one = store.put("pred", obj)
    id_two = store.put("pred", obj)

    assert id_one != id_two


def test_result_store_unknown_id_raises():
    store = ResultStore()
    with pytest.raises(KeyError):
        store.get("pred_999")


def test_two_sessions_do_not_share_a_store():
    raw = pd.DataFrame({"ND": [1, 2, 3]})
    session_a = Session(raw_data=raw)
    session_b = Session(raw_data=raw)

    predictions_id = session_a.store.put("pred", pd.DataFrame({"a": [1]}))

    assert session_a.store is not session_b.store
    with pytest.raises(KeyError):
        session_b.store.get(predictions_id)


def test_session_default_budgets():
    session = Session(raw_data=pd.DataFrame({"ND": [1]}))

    assert session.total_calls_budget == 10
    assert session.dispatch_solves_budget == 50
    assert session.total_calls_used == 0
    assert session.dispatch_solves_used == 0
