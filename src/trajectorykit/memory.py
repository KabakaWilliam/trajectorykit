"""
MemoryStore: Zero-information-loss storage for agent tool outputs.

Accumulates full tool outputs during the agent loop, then serializes
them as a gzip+base64 compressed archive that synthesis sub-agents
can programmatically decode and query via execute_code.
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
