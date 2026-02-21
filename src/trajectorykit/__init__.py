"""
TrajectoryKit: An agentic system with tool calling capabilities powered by vLLM.
"""

from .agent import dispatch
from .tracing import EpisodeTrace, render_trace_html, render_trace_file
from .live_trace import LiveTraceServer

__all__ = ["dispatch", "EpisodeTrace", "render_trace_html", "render_trace_file", "LiveTraceServer"]
__version__ = "0.1.0"
