"""LangGraph routing functions for the ingestion pipeline.

Routing functions are called after each node to decide the next step.
The primary decision: continue to the next node, or short-circuit to END
if the pipeline status is FAILED.
"""

from __future__ import annotations

from typing import Any

from execo_rag.models.pipeline_state import PipelineStatus

StateDict = dict[str, Any]

# Sentinel returned by routing functions to end the graph execution
END = "__end__"


def route_or_fail(next_node: str) -> Any:
    """Return a routing function that continues to *next_node* or ends on failure.

    Args:
        next_node: Name of the node to transition to on success.

    Returns:
        A routing function compatible with LangGraph ``add_conditional_edges``.
    """

    def _route(state: StateDict) -> str:
        status = state.get("status", PipelineStatus.PENDING)
        if status == PipelineStatus.FAILED:
            return END
        return next_node

    return _route


def always_continue(next_node: str) -> Any:
    """Return a routing function that always advances to *next_node*.

    Use for nodes that cannot fail (e.g. purely in-memory operations that
    already handle all exceptions internally).

    Args:
        next_node: Destination node name.

    Returns:
        Routing function that unconditionally returns *next_node*.
    """

    def _route(state: StateDict) -> str:
        return next_node

    return _route


def is_failed(state: StateDict) -> bool:
    """Return True if the pipeline is in a FAILED state."""
    return state.get("status") == PipelineStatus.FAILED
