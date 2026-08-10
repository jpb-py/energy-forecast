import os
import sys
from pathlib import Path

import anthropic

from energy_forecast.data import load_demand_data
from energy_forecast.mia.agent import run_query
from energy_forecast.mia.registry import Session

DEFAULT_DATA_PATH = Path("data/demanddata_2024.csv")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set.")

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("Usage: python -m energy_forecast.mia.cli <question>")

    raw_data = load_demand_data(DEFAULT_DATA_PATH)
    session = Session(raw_data=raw_data)
    client = anthropic.Anthropic(api_key=api_key)

    answer = run_query(client, session, question)
    print(answer)


if __name__ == "__main__":
    main()
