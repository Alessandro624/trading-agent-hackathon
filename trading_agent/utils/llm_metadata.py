from __future__ import annotations

from trading_agent.core import LlmClient


def llm_metadata(llm_client: LlmClient) -> dict:
    """Normalize provider/fallback metadata for journal entries."""
    metadata = getattr(llm_client, "metadata", None)
    if callable(metadata):
        return metadata()
    return {
        "llm_provider": llm_client.__class__.__name__.lower(),
        "llm_fallback_used": False,
        "llm_fallback_provider": None,
        "llm_fallback_reason": None,
    }
