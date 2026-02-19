"""
TrajectoryKit: An agentic system with tool calling capabilities powered by vLLM.
"""

from .agent import dispatch
from .tracing import EpisodeTrace, render_trace_html, render_trace_file

__all__ = ["dispatch", "EpisodeTrace", "render_trace_html", "render_trace_file"]
__version__ = "0.1.0"
