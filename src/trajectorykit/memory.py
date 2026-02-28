"""
MemoryStore: Zero-information-loss storage for agent tool outputs.

Accumulates full tool outputs during the agent loop, then serializes
them as a gzip+base64 compressed archive that synthesis sub-agents
can programmatically decode and query via execute_code.

Symbolic Memory: Tool outputs are stored on the sandbox filesystem
and only summaries + symbolic references go into the LLM context.
The model can recall full outputs via the recall() tool.
"""

import base64
import gzip
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single stored tool output."""
    key: str                    # semantic key, e.g. "search_3_quantum_computing"
    source_tool: str            # tool that produced this output
    turn: int                   # turn number when captured
    content: str                # full output text
    content_length: int = 0     # original char count
    
    def __post_init__(self):
        self.content_length = len(self.content)


class MemoryStore:
    """Key-value store for tool outputs with compression support.
    
    Usage in the agent loop:
        store = MemoryStore()
        store.add("search_1_topic", "search_web", 1, full_output_text)
        ...
        archive_b64 = store.to_compressed_archive()  # gzip+base64
    
    Usage in a synthesis sub-agent (via execute_code):
        import base64, gzip, json
        data = json.loads(gzip.decompress(base64.b64decode(archive_b64)))
        for entry in data["entries"]:
            print(entry["key"], len(entry["content"]))
    """
    
    def __init__(self, max_entries: int = 50, max_total_chars: int = 500_000):
        self.entries: List[MemoryEntry] = []
        self.max_entries = max_entries
        self.max_total_chars = max_total_chars
        self._total_chars = 0
    
    def add(
        self,
        tool_name: str,
        turn: int,
        content: str,
        description: str = "",
    ) -> str:
        """Add a tool output to the store.
        
        Args:
            tool_name: Name of the tool (e.g. "search_web", "fetch_url")
            turn: Turn number when this output was produced
            content: Full output text
            description: Optional human-readable description for the key
            
        Returns:
            The generated key for this entry
        """
        # Generate a semantic key
        key = self._make_key(tool_name, turn, description, content)
        
        # Enforce limits: drop oldest entries if we exceed bounds
        while (
            self.entries
            and (
                len(self.entries) >= self.max_entries
                or self._total_chars + len(content) > self.max_total_chars
            )
        ):
            dropped = self.entries.pop(0)
            self._total_chars -= dropped.content_length
        
        entry = MemoryEntry(
            key=key,
            source_tool=tool_name,
            turn=turn,
            content=content,
        )
        self.entries.append(entry)
        self._total_chars += entry.content_length
        return key
    
    def get(self, key: str) -> Optional[str]:
        """Retrieve content by key."""
        for entry in self.entries:
            if entry.key == key:
                return entry.content
        return None
    
    def keys(self) -> List[str]:
        """List all stored keys."""
        return [e.key for e in self.entries]
    
    def summary(self) -> str:
        """Return a compact summary of stored entries (for prompt injection)."""
        lines = []
        for e in self.entries:
            lines.append(
                f"  {e.key} ({e.source_tool}, turn {e.turn}, "
                f"{e.content_length:,} chars)"
            )
        header = f"Memory Store: {len(self.entries)} entries, {self._total_chars:,} chars total"
        return header + "\n" + "\n".join(lines)
    
    @property
    def total_chars(self) -> int:
        return self._total_chars
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __bool__(self) -> bool:
        return len(self.entries) > 0
    
    # ── Serialization ──────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "entry_count": len(self.entries),
            "total_chars": self._total_chars,
            "entries": [
                {
                    "key": e.key,
                    "source_tool": e.source_tool,
                    "turn": e.turn,
                    "content": e.content,
                    "content_length": e.content_length,
                }
                for e in self.entries
            ],
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def to_compressed_archive(self) -> str:
        """Serialize to gzip + base64 string.
        
        Returns a base64-encoded string that can be passed to a sub-agent,
        which decodes it programmatically via execute_code:
        
            import base64, gzip, json
            data = json.loads(gzip.decompress(base64.b64decode(archive)))
            for entry in data["entries"]:
                print(entry["key"], ":", entry["content"][:200])
        """
        json_bytes = self.to_json().encode("utf-8")
        compressed = gzip.compress(json_bytes, compresslevel=6)
        return base64.b64encode(compressed).decode("ascii")
    
    def compression_stats(self) -> Dict[str, Any]:
        """Return stats about the compression ratio."""
        json_str = self.to_json()
        json_bytes = json_str.encode("utf-8")
        compressed = gzip.compress(json_bytes, compresslevel=6)
        b64 = base64.b64encode(compressed).decode("ascii")
        return {
            "entries": len(self.entries),
            "raw_chars": self._total_chars,
            "json_bytes": len(json_bytes),
            "compressed_bytes": len(compressed),
            "base64_chars": len(b64),
            "compression_ratio": round(len(b64) / len(json_bytes), 3) if json_bytes else 0,
            "savings_pct": round((1 - len(b64) / len(json_bytes)) * 100, 1) if json_bytes else 0,
        }
    
    # ── Key generation ─────────────────────────────────────────────────
    
    @staticmethod
    def _make_key(
        tool_name: str,
        turn: int,
        description: str,
        content: str,
    ) -> str:
        """Generate a semantic key: {tool_abbrev}_{turn}_{description}.
        
        If no description is provided, extract keywords from the content.
        """
        # Tool name abbreviations
        abbrev_map = {
            "search_web": "search",
            "fetch_url": "page",
            "read_pdf": "pdf",
            "execute_code": "code",
            "spawn_agent": "agent",
        }
        tool_abbrev = abbrev_map.get(tool_name, tool_name[:6])
        
        if description:
            # Clean the description into a slug
            slug = re.sub(r"[^a-z0-9]+", "_", description.lower().strip())
            slug = slug.strip("_")[:40]
        else:
            # Auto-extract from content: take first meaningful words
            # Skip common prefixes like "Search Results for"
            text = content[:200].lower()
            text = re.sub(r"^(search results for|exit code|current time|error).*?\n", "", text)
            words = re.findall(r"[a-z]{3,}", text)
            # Take first 4 unique words
            seen = set()
            slug_words = []
            for w in words:
                if w not in seen and w not in ("the", "and", "for", "that", "this", "with", "from", "are", "was", "http", "https", "www"):
                    seen.add(w)
                    slug_words.append(w)
                    if len(slug_words) >= 4:
                        break
            slug = "_".join(slug_words) if slug_words else "data"
        
        return f"{tool_abbrev}_t{turn}_{slug}"
    
    # ── Class methods for deserialization ──────────────────────────────
    
    @classmethod
    def from_compressed_archive(cls, archive_b64: str) -> "MemoryStore":
        """Reconstruct a MemoryStore from a compressed archive string.
        
        This is mainly for testing/debugging. In production, the synthesis
        sub-agent decodes the archive directly via execute_code.
        """
        json_bytes = gzip.decompress(base64.b64decode(archive_b64))
        data = json.loads(json_bytes)
        store = cls()
        for entry_dict in data.get("entries", []):
            entry = MemoryEntry(
                key=entry_dict["key"],
                source_tool=entry_dict["source_tool"],
                turn=entry_dict["turn"],
                content=entry_dict["content"],
            )
            store.entries.append(entry)
            store._total_chars += entry.content_length
        return store


# ── Summary extraction ────────────────────────────────────────────────────
# Heuristic extractors that produce ~200-800 char summaries from full
# tool outputs. These go into the LLM context instead of the full output.

# Max chars to keep in LLM context per tool result (hybrid approach)
CONTEXT_TRUNCATE_CHARS = 800


def extract_summary(tool_name: str, content: str, tool_args: dict = None) -> str:
    """Extract a compact summary from a tool output.
    
    Returns a string suitable for injection into the LLM context in place
    of the full output. Includes the most informative parts + a note about
    how to recall the full content.
    
    Args:
        tool_name: The tool that produced this output
        content: Full output text
        tool_args: Original tool arguments (for context)
        
    Returns:
        Summary string, typically 200-800 chars
    """
    if not content or content.startswith("ERROR:"):
        return content  # errors go through verbatim
    
    tool_args = tool_args or {}
    
    if tool_name == "search_web":
        return _summarize_search(content)
    elif tool_name in ("fetch_url", "smart_fetch"):
        return _summarize_page(content, tool_args.get("url", ""))
    elif tool_name == "read_pdf":
        return _summarize_page(content, tool_args.get("url", ""))
    elif tool_name == "download_file":
        return _summarize_generic(content)  # download results are already compact JSON
    elif tool_name == "spawn_agent":
        return _summarize_agent(content)
    elif tool_name == "sandbox_shell":
        return _summarize_shell(content)
    else:
        return _summarize_generic(content)


def _summarize_search(content: str) -> str:
    """Extract titles + URLs from search results, skip snippets."""
    lines = content.split("\n")
    summary_lines = []
    for line in lines:
        line = line.strip()
        # Keep header, numbered titles, and URLs
        if line.startswith("Search Results for"):
            summary_lines.append(line)
        elif re.match(r'^\d+\.', line):
            summary_lines.append(line)
        elif line.startswith("URL:"):
            summary_lines.append(f"   {line}")
    
    result = "\n".join(summary_lines)
    if len(result) > CONTEXT_TRUNCATE_CHARS:
        result = result[:CONTEXT_TRUNCATE_CHARS] + "\n... (truncated — use recall to see full results)"
    return result


def _summarize_page(content: str, url: str) -> str:
    """First ~600 chars of a fetched page, plus a note about the full content."""
    total = len(content)
    # Take the first meaningful chunk
    preview = content[:600].strip()
    # Try to cut at a sentence boundary
    last_period = preview.rfind(".")
    if last_period > 200:
        preview = preview[:last_period + 1]
    
    lines = [preview]
    if total > 600:
        lines.append(f"\n... [{total:,} chars total — use recall to read the full page]")
    return "\n".join(lines)


def _summarize_agent(content: str) -> str:
    """Sub-agent results: keep first 800 chars (usually concise already)."""
    if len(content) <= CONTEXT_TRUNCATE_CHARS:
        return content
    preview = content[:CONTEXT_TRUNCATE_CHARS]
    last_period = preview.rfind(".")
    if last_period > 400:
        preview = preview[:last_period + 1]
    return preview + f"\n... [{len(content):,} chars total — use recall for full result]"


def _summarize_shell(content: str) -> str:
    """Shell output: keep exit code + first/last lines of stdout."""
    lines = content.split("\n")
    
    # Always keep exit code and duration lines
    header_lines = []
    body_lines = []
    for line in lines:
        if line.startswith(("Exit Code:", "Duration:", "⚠️ TIMED OUT", "STDOUT:", "STDERR:")):
            header_lines.append(line)
        else:
            body_lines.append(line)
    
    if len(body_lines) <= 15:
        # Short output — keep it all
        return content
    
    # Long output — keep first 8 and last 5 lines
    kept = body_lines[:8] + [f"... ({len(body_lines) - 13} lines omitted — use recall for full output)"] + body_lines[-5:]
    return "\n".join(header_lines + kept)


def _summarize_generic(content: str) -> str:
    """Generic fallback: first 800 chars."""
    if len(content) <= CONTEXT_TRUNCATE_CHARS:
        return content
    preview = content[:CONTEXT_TRUNCATE_CHARS]
    return preview + f"\n... [{len(content):,} chars total — use recall for full output]"


def build_memory_index(entries: List[MemoryEntry]) -> str:
    """Build a scannable index of all stored memories.
    
    Returns a compact string listing all entries with their keys,
    sources, turn numbers, and content lengths. This can be injected
    into the LLM context periodically.
    """
    if not entries:
        return "📦 Memory: empty"
    
    lines = [f"📦 Memory Index ({len(entries)} items stored on disk):"]
    for e in entries:
        lines.append(f"  • {e.key} | {e.source_tool} | turn {e.turn} | {e.content_length:,} chars")
    return "\n".join(lines)
