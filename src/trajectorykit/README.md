# TrajectoryKit

> A local-first agentic framework for running LLM agents with recursive sub-agents, sandboxed code execution, and full execution tracing.

This is the core Python package. For full documentation, setup instructions, and experiment configs, see the [project README](../../README.md).

## Package contents

| Module | Purpose |
|--------|---------|
| `agent.py` | Agentic loop — iterative tool calling with budget management and recursive dispatch |
| `config.py` | YAML config loader with fallback chain |
| `tool_store.py` | Tool definitions, wrappers, and dispatch routing |
| `tracing.py` | Trace dataclasses, JSON serialization, and HTML renderer |
| `memory.py` | `MemoryStore` — compressed external storage for tool outputs |
| `utils.py` | Shared utilities |
| `apptainer.sh` | Pull and run the SandboxFusion Apptainer container |

## Quick usage

```python
from trajectorykit import dispatch

result = dispatch(
    user_input="What is the population of Tokyo?",
    turn_length=10,
    verbose=True,
)

print(result["final_response"])
result["trace"].save()
```

## License

MIT
