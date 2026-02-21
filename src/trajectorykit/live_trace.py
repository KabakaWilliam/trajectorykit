"""
Live Trace Server — real-time SSE-based trace viewer.

Uses Python's stdlib ``http.server`` to serve both the live HTML page and a
Server-Sent Events (SSE) stream on a **single HTTP port**.  No external
dependencies required.

Endpoints:
    GET /         → self-contained live HTML page
    GET /events   → SSE stream (text/event-stream) — turns & final response
    *             → 404

The browser uses ``new EventSource('/events')`` which natively supports
auto-reconnection.  Works through VS Code Remote SSH port-forwarding.

Usage (integrated into dispatch):
    server = LiveTraceServer(episode)
    server.start()       # prints URL, no browser auto-launch on headless
    ...
    server.push_turn(turn_record)   # after each turn
    ...
    server.finalize(final_response) # when done
    server.stop()
"""

import json
import logging
import socket
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from .tracing import (
    EpisodeTrace,
    TurnRecord,
    ToolCallRecord,
    _CSS,
    _esc,
    _render_turn_card,
    _flatten_trace,
    _extract_images_from_output,
    _collect_all_images,
    _CHEVRON_SVG,
    _depth_badge,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _find_free_port(start: int = 8500, end: int = 8600) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


# ──────────────────────────────────────────────────────────────────────
# Live HTML template  (JS uses EventSource instead of WebSocket)
# ──────────────────────────────────────────────────────────────────────

def _render_live_html(episode: EpisodeTrace) -> str:
    """Generate the live trace viewer HTML with embedded SSE client JS."""
    tid = _esc(episode.trace_id)
    model = _esc(episode.model)
    started = _esc(episode.started_at)
    user_input = _esc(episode.user_input)

    # ── JS ──────────────────────────────────────────────────────────
    # EventSource connects to /events on the same origin.  Built-in
    # reconnection means we don't need manual retry logic.
    live_js = r"""
var done = false;

function connectSSE() {
  var es = new EventSource('/events');

  es.addEventListener('open', function() {
    document.getElementById('conn-status').className = 'status status-ok';
    document.getElementById('conn-label').textContent = 'connected';
  });

  es.addEventListener('turn', function(e) {
    var msg = JSON.parse(e.data);
    addTurnCard(msg.data);
    updateStats(msg.stats);
  });

  es.addEventListener('finalize', function(e) {
    var msg = JSON.parse(e.data);
    showFinal(msg.data);
    updateStats(msg.stats);
    done = true;
    es.close();
    document.getElementById('conn-status').className = 'status status-ok';
    document.getElementById('conn-label').textContent = 'completed';
    document.getElementById('status-dot').style.background = 'var(--success)';
  });

  es.addEventListener('error', function() {
    if (!done) {
      document.getElementById('conn-status').className = 'status status-warn';
      document.getElementById('conn-label').textContent = 'reconnecting...';
    }
  });
}

function addTurnCard(cardHtml) {
  var container = document.getElementById('cards-container');
  var div = document.createElement('div');
  div.innerHTML = cardHtml;
  while (div.firstChild) {
    container.appendChild(div.firstChild);
  }
  var cards = container.querySelectorAll('.turn-card');
  if (cards.length > 0) {
    cards[cards.length - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  updateTimeline();
}

function updateTimeline() {
  var cards = document.querySelectorAll('.turn-card');
  var timeline = document.getElementById('timeline-list');
  timeline.innerHTML = '';
  cards.forEach(function(card) {
    var id = card.id;
    var title = card.querySelector('.turn-title');
    var label = title ? title.textContent : id;
    var li = document.createElement('li');
    li.className = 'tl-item';
    li.setAttribute('data-target', id);
    li.textContent = label;
    li.addEventListener('click', function() {
      document.querySelectorAll('.tl-item').forEach(function(i){ i.classList.remove('active'); });
      li.classList.add('active');
      scrollToTurn(id);
    });
    timeline.appendChild(li);
  });
}

function updateStats(stats) {
  if (!stats) return;
  document.getElementById('stat-dur').textContent = stats.duration_s.toFixed(1) + 's';
  document.getElementById('stat-turns').textContent = stats.total_turns;
  document.getElementById('stat-subs').textContent = stats.total_subs;
  document.getElementById('stat-prompt').textContent = stats.prompt_tokens.toLocaleString();
  document.getElementById('stat-comp').textContent = stats.completion_tokens.toLocaleString();

  var total = stats.prompt_tokens + stats.completion_tokens;
  var inPct = total ? (stats.prompt_tokens / total * 100) : 50;
  var outPct = 100 - inPct;
  document.getElementById('ratio-in').style.width = inPct.toFixed(1) + '%';
  document.getElementById('ratio-out').style.width = outPct.toFixed(1) + '%';
  document.getElementById('ratio-in-label').textContent = 'Input ' + Math.round(inPct) + '%';
  document.getElementById('ratio-out-label').textContent = 'Output ' + Math.round(outPct) + '%';
  document.getElementById('ratio-total').textContent = total.toLocaleString() + ' total';
}

function showFinal(data) {
  var fc = document.getElementById('final-card');
  fc.style.display = 'block';
  document.getElementById('final-text').textContent = data.final_response || '(no response)';

  var imgContainer = document.getElementById('final-images');
  if (data.images && data.images.length > 0) {
    data.images.forEach(function(img) {
      var div = document.createElement('div');
      div.className = 'tc-img';
      div.style.margin = '8px 0';

      var extMap = {png:'image/png', jpg:'image/jpeg', jpeg:'image/jpeg', gif:'image/gif', svg:'image/svg+xml'};
      var ext = img.filename.split('.').pop().toLowerCase();
      var mime = extMap[ext] || 'image/png';

      div.innerHTML =
        '<div style="font-size:10px;color:var(--text-light);margin-bottom:4px;">\u{1F4CE} ' + img.filename + '</div>' +
        '<img src="data:' + mime + ';base64,' + img.data + '" alt="' + img.filename + '" style="max-width:100%;border-radius:6px;border:1px solid var(--border);"/>';
      imgContainer.appendChild(div);
    });
  }
}

function toggle(header) {
  header.closest('.turn-card').classList.toggle('open');
}

function toggleTool(header) {
  header.closest('.tool-block').classList.toggle('open');
}

function scrollToTurn(id) {
  var el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  el.classList.add('highlight');
  setTimeout(function() { el.classList.remove('highlight'); }, 1200);
}

document.addEventListener('DOMContentLoaded', function() {
  connectSSE();

  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var cards = document.querySelectorAll('.turn-card');
    if (e.key === 'j' || e.key === 'ArrowDown') {
      e.preventDefault();
      for (var i = 0; i < cards.length; i++) {
        if (cards[i].getBoundingClientRect().top > 100) { cards[i].scrollIntoView({ behavior:'smooth', block:'start' }); break; }
      }
    }
    if (e.key === 'k' || e.key === 'ArrowUp') {
      e.preventDefault();
      for (var i = cards.length - 1; i >= 0; i--) {
        if (cards[i].getBoundingClientRect().top < -10) { cards[i].scrollIntoView({ behavior:'smooth', block:'start' }); break; }
      }
    }
  });
});
"""

    live_css_extra = """
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.live-dot { animation: pulse 1.5s ease-in-out infinite; }
#final-card { display: none; }
.status { display: inline-flex; align-items: center; gap: 5px; font-family: var(--mono); font-size: 10px; }
.status-ok #conn-label { color: var(--success); }
.status-warn #conn-label { color: var(--warn); }
.status-err #conn-label { color: var(--error); }
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Live Trace — {tid}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&family=JetBrains+Mono:wght@300;400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>{_CSS}\n{live_css_extra}</style>
</head>
<body>

<div class="viewer">

  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title"><span class="dot live-dot" id="status-dot"></span> trajectorykit <span style="color:var(--warn);font-size:9px;">LIVE</span></div>
      <div class="sidebar-meta">{model}<br/>{started}</div>
      <div style="margin-top:8px;">
        <div id="conn-status" class="status status-warn">
          <span class="status-dot" style="width:6px;height:6px;border-radius:50%;background:var(--warn);"></span>
          <span id="conn-label">connecting...</span>
        </div>
      </div>
    </div>
    <ul class="timeline" id="timeline-list"></ul>
  </aside>

  <!-- Main content -->
  <main class="main">

    <!-- Prompt -->
    <div class="prompt-banner">
      <div class="prompt-label">User Prompt</div>
      <div class="prompt-text">{user_input}</div>
    </div>

    <!-- Stats header -->
    <div class="trace-header">
      <div class="trace-header-bar">
        <div class="trace-header-left"><div class="dot live-dot"></div> Trace {tid}</div>
        <div class="trace-header-right">{model}</div>
      </div>
      <div class="trace-stats">
        <div class="trace-stat"><span class="trace-stat-val" id="stat-dur">0.0s</span><span class="trace-stat-lbl">Duration</span></div>
        <div class="trace-stat"><span class="trace-stat-val" id="stat-turns">0</span><span class="trace-stat-lbl">Turns</span></div>
        <div class="trace-stat"><span class="trace-stat-val" id="stat-subs">0</span><span class="trace-stat-lbl">Sub-Agents</span></div>
        <div class="trace-stat"><span class="trace-stat-val" id="stat-prompt">0</span><span class="trace-stat-lbl">Input Tokens</span></div>
        <div class="trace-stat"><span class="trace-stat-val" id="stat-comp">0</span><span class="trace-stat-lbl">Output Tokens</span></div>
      </div>
      <div style="padding:0 16px 14px;">
        <div class="token-ratio-bar">
          <div class="token-ratio-in" id="ratio-in" style="width:50%"></div>
          <div class="token-ratio-out" id="ratio-out" style="width:50%"></div>
        </div>
        <div class="token-ratio-legend">
          <span><span class="swatch" style="background:var(--accent)"></span> <span id="ratio-in-label">Input 50%</span></span>
          <span><span class="swatch" style="background:var(--success)"></span> <span id="ratio-out-label">Output 50%</span></span>
          <span style="color:var(--text-mid);" id="ratio-total">0 total</span>
        </div>
      </div>
    </div>

    <!-- Turn cards injected here by SSE -->
    <div id="cards-container"></div>

    <!-- Final response (hidden until done) -->
    <div class="final-card" id="final-card">
      <div class="final-header">\u2705 Final Response</div>
      <div class="final-body">
        <div class="output-block" id="final-text"></div>
        <div id="final-images"></div>
      </div>
    </div>

  </main>
</div>

<script>{live_js}</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# SSE HTTP Handler
# ──────────────────────────────────────────────────────────────────────

class _SSEHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the live trace server.

    Routes:
        GET /        → live HTML page
        GET /events  → SSE stream
        *            → 404
    """

    # Silence per-request log lines on stderr
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        else:
            self.send_error(404)

    # Also handle HEAD (some browsers/proxies probe with HEAD)
    def do_HEAD(self):  # noqa: N802
        if self.path == "/" or self.path == "":
            html = self.server.live_html_bytes
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
        else:
            self.send_error(404)

    def _serve_html(self):
        html = self.server.live_html_bytes
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html)

    def _serve_sse(self):
        """Hold the connection open and stream SSE events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
        self.end_headers()

        # Register this connection
        event = threading.Event()
        client = _SSEClient(self.wfile, event)
        self.server.sse_clients_lock.acquire()
        self.server.sse_clients.add(client)
        self.server.sse_clients_lock.release()

        # Replay any events that happened before we connected
        for past_event in list(self.server.sse_event_log):
            try:
                self.wfile.write(past_event)
                self.wfile.flush()
            except Exception:
                break

        try:
            # Block until the server signals us to stop or connection dies.
            # We wake up periodically to send an SSE comment as a keep-alive
            # so proxies / browsers don't time out the connection.
            while not self.server.sse_shutdown.is_set():
                triggered = event.wait(timeout=15)
                if triggered:
                    # Drain queued messages
                    event.clear()
                    for msg_bytes in client.drain():
                        try:
                            self.wfile.write(msg_bytes)
                            self.wfile.flush()
                        except Exception:
                            return
                else:
                    # Keep-alive comment
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except Exception:
                        return
        finally:
            self.server.sse_clients_lock.acquire()
            self.server.sse_clients.discard(client)
            self.server.sse_clients_lock.release()


class _SSEClient:
    """Lightweight wrapper around a single SSE connection."""

    __slots__ = ("wfile", "event", "_queue", "_lock")

    def __init__(self, wfile, event: threading.Event):
        self.wfile = wfile
        self.event = event
        self._queue: list[bytes] = []
        self._lock = threading.Lock()

    def enqueue(self, data: bytes):
        with self._lock:
            self._queue.append(data)
        self.event.set()

    def drain(self) -> list[bytes]:
        with self._lock:
            items = self._queue[:]
            self._queue.clear()
        return items


# ──────────────────────────────────────────────────────────────────────
# LiveTraceServer
# ──────────────────────────────────────────────────────────────────────

class LiveTraceServer:
    """SSE-based live trace server — stdlib only, zero external dependencies.

    Runs ``http.server.ThreadingHTTPServer`` in a daemon thread.
    ``GET /`` serves the live HTML; ``GET /events`` opens an SSE stream.
    The agent loop calls ``push_turn()`` / ``finalize()`` which fan out
    events to all connected browsers.
    """

    def __init__(
        self,
        episode: EpisodeTrace,
        port: Optional[int] = None,
        open_browser: bool = True,
    ):
        self.episode = episode
        self.port = port or _find_free_port()
        self.open_browser = open_browser
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._start_time = time.time()
        self._card_counter = [0, 0]  # [card_counter, sub_agent_counter]
        self._ready = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> str:
        """Start the HTTP+SSE server in a background thread.

        Returns the ``http://…`` URL for the live viewer.
        """
        html_bytes = _render_live_html(self.episode).encode("utf-8")

        httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _SSEHandler)
        httpd.daemon_threads = True

        # Attach shared state that the handler reads
        httpd.live_html_bytes = html_bytes
        httpd.sse_clients: set[_SSEClient] = set()
        httpd.sse_clients_lock = threading.Lock()
        httpd.sse_shutdown = threading.Event()
        httpd.sse_event_log: list[bytes] = []   # replay buffer for late joiners

        self._httpd = httpd

        self._thread = threading.Thread(
            target=httpd.serve_forever, daemon=True, name="live-trace-sse"
        )
        self._thread.start()
        self._ready.set()

        url = f"http://127.0.0.1:{self.port}"
        logger.info("Live trace viewer: %s", url)
        return url

    def stop(self):
        """Shut down the server gracefully."""
        if self._httpd:
            self._httpd.sse_shutdown.set()
            # Wake all SSE clients so their handler threads exit
            self._httpd.sse_clients_lock.acquire()
            for c in self._httpd.sse_clients:
                c.event.set()
            self._httpd.sse_clients_lock.release()
            self._httpd.shutdown()

    # ── push helpers ─────────────────────────────────────────────────

    def _broadcast_sse(self, event_type: str, payload: dict):
        """Format and send an SSE event to all connected clients."""
        if not self._httpd:
            return
        data_json = json.dumps(payload, default=str)
        # SSE format:  event: <type>\ndata: <json>\n\n
        msg = f"event: {event_type}\ndata: {data_json}\n\n".encode("utf-8")

        # Store in replay log so late-connecting browsers catch up
        self._httpd.sse_event_log.append(msg)

        self._httpd.sse_clients_lock.acquire()
        clients = list(self._httpd.sse_clients)
        self._httpd.sse_clients_lock.release()

        for client in clients:
            client.enqueue(msg)

    def _get_stats(self) -> dict:
        """Compute current aggregate stats from the episode."""
        prompt_tok = 0
        comp_tok = 0
        subs = 0

        def _walk(trace_dict):
            nonlocal prompt_tok, comp_tok, subs
            for turn in trace_dict.get("turns", []):
                prompt_tok += turn.get("prompt_tokens", 0)
                comp_tok += turn.get("completion_tokens", 0)
                for tc in turn.get("tool_calls", []):
                    if tc.get("child_trace"):
                        subs += 1
                        _walk(tc["child_trace"])

        _walk(self.episode.to_dict())
        return {
            "duration_s": round(time.time() - self._start_time, 1),
            "total_turns": len(self.episode.turns),
            "total_subs": subs,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
        }

    # ── public push API ──────────────────────────────────────────────

    def push_turn(self, turn_record: TurnRecord, depth: int = 0):
        """Push a completed turn to all connected browsers."""
        self._card_counter[0] += 1
        card_id = f"card-{self._card_counter[0]}"
        agent_label = "Root" if depth == 0 else "Sub-Agent"

        card = {
            "type": "turn",
            "id": card_id,
            "depth": depth,
            "turn": _serialize_turn(turn_record),
            "turn_num": turn_record.turn_number,
            "agent_label": agent_label,
        }
        card_html = _render_turn_card(card)

        # Inline sub-agent cards for spawn_agent tool calls
        for tc in turn_record.tool_calls:
            if tc.child_trace is not None:
                child_cards = _flatten_trace(
                    tc.child_trace.to_dict(), depth + 1, self._card_counter
                )
                for child_card in child_cards:
                    card_html += "\n" + _render_turn_card(child_card)

        self._broadcast_sse("turn", {
            "data": card_html,
            "stats": self._get_stats(),
        })

    def finalize(self, final_response: str):
        """Send the final response and mark the trace as completed."""
        all_images = _collect_all_images(self.episode.to_dict())
        self._broadcast_sse("finalize", {
            "data": {
                "final_response": final_response,
                "images": all_images,
            },
            "stats": self._get_stats(),
        })


# ──────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────

def _serialize_turn(turn_record: TurnRecord) -> dict:
    """Recursively serialize a TurnRecord to a plain dict."""
    def _ser(obj):
        if isinstance(obj, (TurnRecord, ToolCallRecord)):
            return {k: _ser(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, EpisodeTrace):
            return obj.to_dict()
        elif isinstance(obj, list):
            return [_ser(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        return obj
    return _ser(turn_record)
