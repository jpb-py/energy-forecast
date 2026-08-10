import json

import anthropic

from energy_forecast.mia.registry import FORECAST_FUNCTIONS, Session
from energy_forecast.mia.schema import TOOL_SCHEMAS
from energy_forecast.mia.tools import TOOL_HANDLERS

MODEL = "claude-sonnet-5"
MAX_RESPONSE_TOKENS = 2048

SYSTEM_PROMPT_TEMPLATE = """You are the Model Investigation Assistant (MIA) for a UK \
electricity demand forecasting and battery dispatch project. Answer diagnostic questions \
about forecast and dispatch performance using the tools available to you. All tools are \
read-only / simulate-only: nothing you call writes to the repository, retrains a model, or \
modifies a saved artifact.

Registered forecast function ids: {forecast_fn_ids}.

You have a hard budget of {total_calls_budget} tool calls for this investigation, and a \
separate budget of {dispatch_solves_budget} underlying daily dispatch solves (a multi-day \
run_dispatch_scenario call spends one unit of this budget per calendar day it covers). Once \
a budget is exhausted, further tool calls will be refused -- plan your investigation \
accordingly and answer with the evidence you've gathered so far if you run out.

If a request is ambiguous, outside what your tools support, or you're not sure how to \
proceed, say so explicitly and either ask for clarification or state clearly that you're \
addressing a narrower version of the question. Never substitute a different question \
without saying so. When you know which tool to call next, call it in the same turn rather \
than describing your plan and stopping."""


def _build_system_prompt(session: Session) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        forecast_fn_ids=", ".join(sorted(FORECAST_FUNCTIONS)),
        total_calls_budget=session.total_calls_budget,
        dispatch_solves_budget=session.dispatch_solves_budget,
    )


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _run_tool(session: Session, tool_use) -> dict:
    handler = TOOL_HANDLERS.get(tool_use.name)
    if handler is None:
        return _tool_result(tool_use.id, f"Unknown tool '{tool_use.name}'.", is_error=True)

    try:
        result = handler(session, tool_use.input)
        return _tool_result(tool_use.id, json.dumps(result, default=str))
    except Exception as exc:  # noqa: BLE001 -- handler boundary: any domain error must
        # become a tool_result the model can see and react to, never a crashed loop.
        return _tool_result(tool_use.id, f"{type(exc).__name__}: {exc}", is_error=True)


def run_query(client: anthropic.Anthropic, session: Session, question: str) -> str:
    """Run one diagnostic investigation to completion and return MIA's final answer.

    Enforces both hard guardrails in code (not just via the system prompt): once
    session.total_calls_budget is exhausted, no further tool calls are executed --
    the loop either withholds tool definitions from the next request or returns a
    budget-exhausted error for any further tool_use block, forcing a text answer.
    """
    messages = [{"role": "user", "content": question}]
    system_prompt = _build_system_prompt(session)

    while True:
        budget_left = session.total_calls_budget - session.total_calls_used
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_RESPONSE_TOKENS,
            system=system_prompt,
            tools=TOOL_SCHEMAS if budget_left > 0 else [],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if session.total_calls_used >= session.total_calls_budget:
                tool_results.append(
                    _tool_result(
                        block.id,
                        "Tool call budget exhausted for this investigation. "
                        "Answer with the evidence gathered so far.",
                        is_error=True,
                    )
                )
                continue

            session.total_calls_used += 1
            print(f"[tool call {session.total_calls_used}/{session.total_calls_budget}] {block.name}")
            tool_results.append(_run_tool(session, block))

        messages.append({"role": "user", "content": tool_results})
