"""AI-assisted orchestration helpers for data collection and curation.

Many users prefer a conversational workflow when preparing datasets.  The
:class:`AIHousekeeper` couples the :class:`LocalDataHub` with an optional LLM
client so that the system can suggest missing artefacts, draft to-do lists, or
respond to ad-hoc data requests.  The class is intentionally conservative—it
never sends files automatically and falls back to scripted prompts when an LLM
client is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from .data_hub import LocalDataHub


class LLMClient(Protocol):
    """Simple protocol for large-language-model integrations."""

    def complete(self, prompt: str) -> str:
        ...


@dataclass
class ConsoleLLMClient:
    """Offline fallback that returns templated responses."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - deterministic formatting
        header = "AI Housekeeper (offline mode)"
        body = (
            "I do not have access to an external language model in this environment.\n"
            "However, based on the supplied prompt I recommend reviewing the data\n"
            "hub summary and manually fetching any missing artefacts.\n"
            f"Prompt received:\n{prompt}"
        )
        return f"{header}\n{'-' * len(header)}\n{body}"


class AIHousekeeper:
    """Coordinates data requirements with an optional conversational agent."""

    def __init__(self, hub: LocalDataHub, llm_client: Optional[LLMClient] = None) -> None:
        self.hub = hub
        self.llm = llm_client or ConsoleLLMClient()

    def _build_requirements_prompt(self, tickers: Iterable[str]) -> str:
        tickers = [t.upper() for t in tickers]
        summary = self.hub.summary_table()
        if summary.empty:
            catalog_text = "No artefacts recorded yet."
        else:
            filtered = summary[summary["ticker"].isin(tickers)] if tickers else summary
            if filtered.empty:
                catalog_text = "No artefacts recorded for the requested tickers."
            else:
                catalog_text = filtered.to_string(index=False)
        prompt = (
            "You are assisting with an equity research knowledge base.\n"
            "The current local data catalogue looks like this:\n"
            f"{catalog_text}\n\n"
            "Given this view, produce an actionable checklist describing which\n"
            "market data, fundamentals, news sentiment, and filings should be\n"
            "downloaded next.  Focus on filling the gaps and prioritise critical\n"
            "signals for near-term model training."
        )
        return prompt

    def plan_requirements(self, tickers: Iterable[str]) -> str:
        """Generate a data shopping list for the supplied tickers."""

        prompt = self._build_requirements_prompt(tickers)
        return self.llm.complete(prompt)

    def interactive_session(self, intro: Optional[str] = None) -> None:  # pragma: no cover - interactive
        """Enter a REPL-style conversation with the configured LLM client."""

        print(intro or "AI housekeeper ready. Type 'exit' to leave.")
        while True:
            user = input("housekeeper> ").strip()
            if user.lower() in {"exit", "quit"}:
                print("Session ended.")
                break
            if not user:
                continue
            response = self.llm.complete(user)
            print(response)


__all__ = ["AIHousekeeper", "ConsoleLLMClient", "LLMClient"]
