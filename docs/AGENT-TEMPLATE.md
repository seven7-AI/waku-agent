# The Agent Blueprint — a template for building agents from small to scalable

A distilled, reusable architecture derived from **waku-agent**. It captures the four
pillars behind every serious agent — **Harness · Loop · Memory · Eval/LLM-Ops** — as a
pattern you can drop onto any project, then grow one boring default at a time.

The guiding principle: **every layer has a zero-setup default and a documented upgrade
path.** Start local, single-file, stdlib-only. Swap in a hosted service only when a real
constraint forces it — and never change more than one layer at a time.

---

## 0. The mental model

An agent is not a framework. It is a `while` loop that calls an LLM, runs the tools the
LLM asks for, feeds results back, and repeats until the model stops asking. Everything
else — memory, tracing, evals, gateways — hangs off that loop as **seams**, not tangles.

```
          ┌─────────────────────────────────────────────────────────┐
          │                     GATEWAY (moves text)                 │
          │            cli · web · telegram · voice · api            │
          └───────────────┬─────────────────────────▲───────────────┘
                          │ user message            │ reply
                          ▼                          │
    ┌──────────────────────────────────────────────────────────────────┐
    │  EPHEMERAL RUN  (rebuilt every turn, then thrown away)            │
    │                                                                  │
    │   WORKING MEMORY  =  system prompt (persona)                     │
    │                    + retrieved memory (gated!)                   │
    │                    + sliding window of chat history             │
    │                    + the new user message                       │
    │                            │                                     │
    │                            ▼                                     │
    │        ┌──────────── THE LOOP ────────────┐                      │
    │        │  llm(messages, tools) ── reason   │                      │
    │        │      │ tool calls?                │                      │
    │        │      ▼                            │                      │
    │        │  run(tools) ── act                │                      │
    │        │      │ results                    │                      │
    │        │      ▼                            │                      │
    │        │  messages += results ── observe   │  guardrails:         │
    │        │      └── loop until no tools ─────┘  no-tool exit ·      │
    │        └──────────────────────────────────┘  max iterations      │
    └──────────────────────────┬───────────────────────────────────────┘
             every event │                     │ save / retrieve
                         ▼                     ▼
    ┌───────────────────────────┐   ┌────────────────────────────────────┐
    │   LLM-OPS                 │   │   MEMORY (persists between turns)    │
    │  trace (JSONL + OTel)    │   │  gate → "does this turn need memory?"│
    │  deterministic evals     │   │  semantic  — durable facts           │
    │  judge evals             │   │  episodic  — dated events            │
    │  release gate            │   │  procedural — SKILL.md "how to act"  │
    └───────────────────────────┘   │  consolidation — distill every N     │
                                    │  one store file (SQLite)             │
                                    └────────────────────────────────────┘
```

Four rules that keep it from getting muddy as it grows:

1. **The loop stays tiny.** If your core loop is more than ~100 lines, indirection has
   crept in. Push complexity into tools and memory, not the loop.
2. **Gateways only move text.** No business logic in cli/telegram/web. They call one
   `respond()` function and render what comes back.
3. **Observers, not wires.** Tracing and live UI subscribe to loop events through a
   callback seam; the loop never imports the dashboard or the tracer.
4. **Boring default, documented upgrade.** SQLite before Postgres. JSONL before Phoenix.
   Keyword search before embeddings. The default must be zero-signup.

---

## 1. Project layout

A package per pillar. The names *are* the diagram.

```
myagent/
├── gateway/          # interfaces — move text only (cli, web, telegram, voice, api)
├── runtime/          # working-memory assembly (per-turn, ephemeral)
│   └── session.py
├── loop/             # THE loop + pluggable model providers
│   ├── agent.py      #   ~95 lines: observe → reason → act → repeat
│   └── models.py     #   provider adapters (2 wire formats, ~60-line bridge)
├── tools/            # name + JSON schema + python fn; a registry that runs them
│   └── registry.py
├── memory/           # the pillar that makes it feel personal
│   ├── semantic/     #   durable facts (FTS5 keyword search → pgvector upgrade)
│   ├── episodic/     #   dated events (SQLite → Notion/other upgrade)
│   ├── procedural/   #   SKILL.md files: how to act, loaded on match
│   ├── retrieval_gate.py   # decides IF a turn needs memory   (hero #1)
│   └── consolidation.py    # distills chats into facts every N (hero #2)
├── ops/              # LLM-Ops: tracing, dashboard, release gate, scoring
│   └── tracing.py
├── config.py         # one dataclass, every knob an env var
├── db.py             # one SQLite file + schema
└── app.py            # WIRING — config → db → tools → memory → session → loop

evals/
├── deterministic/    # 0/1 pytest — "did the right tool fire?"  (unit tests)
└── judge/            # scored % via LLM-as-judge — "was the reply good?"
```

**`app.py` is the assembly diagram in code.** Read it first, understand the whole system:

```python
class Agent:
    def __init__(self, settings=None, client=None, conn=None):
        # client & conn are INJECTABLE — evals swap a scripted model,
        # the web server injects a cross-thread connection. Same seam.
        self.settings = settings or load_settings()
        self.conn     = conn   or connect(self.settings.home)
        self.client   = client or get_client(self.settings)   # provider adapter
        self.memory   = Memory(self.conn, self.settings, self.client)
        self.tools    = build_registry(self.conn, self.settings, self.memory)
        self.session  = Session(self.settings, memory=self.memory)
        self.tracer   = Tracer(self.settings)

    def respond(self, user_message, observer=None, source="cli", stream=False):
        # one full turn: assemble working memory → run loop → persist
        ...
```

> **Injectable seams are the single most important design choice.** Because `client` and
> `conn` are constructor arguments, the *entire* agent is testable with a fake model and a
> temp database — no mocking framework, no network. This is what makes the eval pillar cheap.

---

## 2. Pillar I — The Harness (config, wiring, gateways)

The harness is the boring plumbing that makes everything else swappable.

### Config: one dataclass, every knob an env var

No settings framework. If you can read the file, you know everything the agent can do.

```python
@dataclass
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("AGENT_PROVIDER", "anthropic"))
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", ""))
    small_model: str = field(default_factory=lambda: os.getenv("AGENT_SMALL_MODEL", ""))
    home: Path = field(default_factory=lambda: Path(os.getenv("AGENT_HOME", ".agent")))
    max_iterations: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "10")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "8192")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("AGENT_HISTORY_TURNS", "12")))
    # ... memory knobs, tool toggles, otel endpoint
```

Two knobs worth understanding early:

- **`max_tokens` needs headroom for reasoning models.** They spend output tokens *thinking*
  before the answer; a low cap makes them hit the limit mid-thought and return an **empty
  reply**. It's a ceiling, not a target — efficient models still cost the same.
- **`history_turns` is a sliding window** (context RAM). Only the last N turns enter the
  prompt so cost/latency stay flat no matter how long the conversation runs. Older turns
  aren't lost — they're in the store, distilled into facts, pulled back when relevant.

### Gateways: they only move text

Every gateway is the same shape: read input → call `respond()` → render output. The CLI
and Telegram gateways differ only in `input()` vs. polling. A `source` tag ("cli",
"telegram", "voice", "web") rides with each message so one unified log shows its origin.

```python
def main():
    agent = Agent()
    while True:
        msg = input("you › ").strip()
        result = agent.respond(msg, observer=show_live, source="cli")
        print("agent ›", result.reply)
```

**Scaling gateways:** the seam never changes. Add a FastAPI route, a Slack bolt handler, a
webhook — each is ~40 lines that call the same `respond()`. Nothing about the loop or
memory knows a new channel exists.

---

## 3. Pillar II — The Loop (the whole trick)

Every agent framework is ultimately this, with more indirection:

```python
def run_loop(client, model, system, messages, tools,
             max_iterations=10, max_tokens=2048, observer=None):
    notify = observer or (lambda kind, ev: None)
    result = LoopResult(reply="")

    for iteration in range(1, max_iterations + 1):
        # ---- reason: one LLM call with the current working memory
        response = client.messages.create(
            model=model, system=system, messages=messages,
            tools=tools.schemas(), max_tokens=max_tokens,
        )
        notify("llm", {"iteration": iteration, "usage": {...}})

        # the assistant's turn joins working memory
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        # ---- guardrail 1: no tool calls → the model is talking to the human
        if not tool_uses:
            result.reply = "".join(b.text for b in response.content if b.type == "text")
            return result

        # ---- act + observe: run each tool, feed results back
        tool_results = []
        for call in tool_uses:
            output = tools.execute(call.name, call.input)
            notify("tool", {"tool": call.name, "args": call.input, "output": output})
            tool_results.append({"type": "tool_result",
                                 "tool_use_id": call.id, "content": output})
        messages.append({"role": "user", "content": tool_results})

    # ---- guardrail 2: ran out of iterations — never spin forever
    result.reply = "(hit iteration limit — try smaller steps.)"
    return result
```

That's the entire engine. Key ideas to steal:

- **`messages` is mutated in place.** After the call it holds the full working memory of the
  turn (assistant thoughts, tool calls, tool results) — exactly what you trace.
- **Two guardrails are non-negotiable:** the model stopping (natural end) and a hard
  iteration cap (never spin forever / never bankrupt yourself).
- **Observers decouple everything.** `notify(kind, event)` fans out to the live UI *and* the
  tracer *and* anything else — none of which the loop imports. This one callback is how you
  get live streaming, tracing, and a cost ledger without touching loop logic.
- **Errors are observed, not raised.** A tool that throws returns an error *string* the model
  reads and can retry from — a crash in a tool never crashes the turn.

### Provider portability: one dialect, thin adapters

The loop speaks **one** wire format (here, Anthropic's Messages shape: system/messages/tools
in, content blocks out). Providers plug in two ways:

- **Native wire format** — providers that already speak your dialect: pass through.
- **Adapter** — a ~60-line class that translates the *other* major wire format (OpenAI's
  chat.completions) in and out. That single adapter unlocks a dozen providers.

```python
PROVIDERS = {
    "anthropic": Provider(kind="anthropic", key_env="ANTHROPIC_API_KEY", ...),
    "openai":    Provider(kind="openai",    key_env="OPENAI_API_KEY", ...),
    "gemini":    Provider(kind="openai",    key_env="GEMINI_API_KEY", base_url=...),
    # ... deepseek, openrouter, local models — all just strings + a wire kind
}

def get_client(settings):
    provider = PROVIDERS[settings.provider]
    if provider.kind == "anthropic":
        return anthropic.Anthropic(api_key=..., base_url=provider.base_url)
    return OpenAICompatClient(api_key=..., base_url=provider.base_url)  # the adapter
```

Model IDs are *just strings* — override with env vars when defaults age out. This is the
difference between "locked to one vendor" and "use whatever you already pay for."

> **Gotcha worth copying:** `.strip()` API keys and validate they're latin-1 encodable.
> A smart-quote or trailing newline from a bad paste corrupts the auth header with a
> cryptic error. Catch it at the door with a clear message.

---

## 4. Pillar III — Memory (what makes it feel personal)

Memory is the hero. It is three stores plus two small agents that manage them — all behind
one facade.

| Kind | Question it answers | Default store | Retrieval |
|---|---|---|---|
| **Semantic** | "what is durably true?" (facts) | SQLite FTS5 | keyword top-k (BM25) |
| **Episodic** | "what happened, when?" | SQLite | keyword top-k |
| **Procedural** | "how do I act?" (SKILL.md) | files | keyword overlap on frontmatter |

### One store file, openable

Everything lives in **one SQLite file** with FTS5 for keyword search — no server, and you
can `sqlite3 state.db '.tables'` to read your agent's mind. For a single user's facts,
ranked keyword search is fast, fully local, and legible. Reach for embeddings only when
you actually have a semantic-similarity problem keyword search can't solve.

```sql
CREATE TABLE facts (id INTEGER PRIMARY KEY, subject TEXT, content TEXT, source TEXT, created_at TEXT);
CREATE VIRTUAL TABLE facts_fts USING fts5(subject, content, content=facts, content_rowid=id);
-- + triggers that keep the FTS index in sync on insert/update/delete
CREATE TABLE episodes (id INTEGER PRIMARY KEY, happened_at TEXT, summary TEXT, ...);
CREATE TABLE chat_log (id INTEGER PRIMARY KEY, role TEXT, content TEXT,
                       consolidated INTEGER DEFAULT 0, session_id TEXT, source TEXT, meta TEXT);
```

### Hero moment #1 — the retrieval gate (retrieve *only when it helps*)

Do **not** hit the memory store every turn. Default-on retrieval is (a) slow — an extra
search before every reply — and (b) *worse*: irrelevant memories bias the answer. So before
touching any store, a cheap fast model answers one narrow question:

```
"does THIS message need the user's memory?"
  "what's 2+2"           → no
  "when am I meeting Alex?" → yes, query="meeting Alex"
```

```python
def should_retrieve(client, small_model, message) -> tuple[bool, str, str]:
    """Returns (retrieve?, query, reason). FAILS OPEN: if the gate errors,
    retrieve anyway — a stale memory beats a lost one."""
    resp = client.messages.create(model=small_model, max_tokens=600,
        messages=[{"role": "user", "content": GATE_PROMPT.format(message=message)}])
    # parse {"retrieve": bool, "query": "...", "reason": "..."} — fail open on any error
```

Cost: one small-model call. Payoff: retrieval only when it earns its place. **Fail open** —
if the gate itself breaks, retrieve rather than lose context.

### Hero moment #2 — consolidation (distill chats into memory, batched)

Running a summarizer after *every* message is wasteful and noisy. Instead, batch: after N
new exchanges, a cheap model reads the unconsolidated chat log and distills:

- **facts** → semantic memory ("Alex prefers morning meetings")
- **one episode** → episodic memory ("2026-07-10: planned the Acme demo with Alex")

It is **asynchronous to the reply path and loss-safe**: if the summarizer fails, the chat
log simply stays unconsolidated and is retried next time. You never lose data.

```python
def consolidate_if_due(conn, client, small_model, every_n, facts, episodes) -> int:
    rows = conn.execute("SELECT ... FROM chat_log WHERE consolidated = 0 ORDER BY id").fetchall()
    if len(rows) < every_n * 2:          # each exchange = 2 rows (user + assistant)
        return 0
    try:
        distilled = json.loads(<summarizer call over the log>)
    except Exception:
        return 0                         # never lose the log — retry next time
    for f in distilled["facts"]: facts.add(f["subject"], f["content"], source="consolidation")
    episodes.add(distilled["episode"], happened_at=date.today().isoformat())
    conn.execute("UPDATE chat_log SET consolidated = 1 WHERE id IN (...)")
    return len(distilled["facts"])
```

### Procedural memory — SKILL.md with progressive disclosure

Skills are Markdown files with YAML frontmatter (`name`, `description`). The **description
doubles as the trigger**. Progressive disclosure keeps the prompt lean:

1. Frontmatter of every skill is always scanned (cheap).
2. A skill's **body** loads into the prompt only when it matches the message.
3. Files a skill references are read only if the model asks.

Matching is transparent keyword overlap — no embeddings, you can compute the score by hand.
The loader re-scans on file change, so a skill authored mid-session is live next turn.

### Working memory assembly (the ephemeral run)

Every turn, `session.build_system()` rebuilds the system prompt from scratch:

```
system prompt (persona / SOUL.md)      ← who the agent is
  + "right now it is <local time>"      ← so it can resolve "next Tuesday"
  + "you are running on <model>"        ← the first thing users ask
  + gated memory retrieval (if needed)  ← hero #1
  + matching skill bodies (if any)      ← procedural
  + sliding window of chat history      ← last N turns only
  + the new user message
```

> **The bug that teaches the pattern:** fold tool activity into the history as a compact
> `[tools used: ...]` line. Without it, the model forgets it already acted and re-runs the
> same tool next turn (the classic "triple-booked meeting" bug). Working memory must record
> *what the agent did*, not just what it said.

### Scaling memory

| Layer | Default (zero-setup) | Upgrade path | When to upgrade |
|---|---|---|---|
| Semantic | SQLite FTS5 keyword | pgvector / hosted vector DB | you have a true semantic-similarity need, or multi-user scale |
| Episodic | SQLite | Notion / external DB | events need to live in a system-of-record |
| Store swap | branch in a factory method | — | one env var (`SEMANTIC_STORE=...`), interface unchanged |

The store interface (`.add()`, `.search()`) stays identical, so the upgrade is a factory
branch, not a rewrite:

```python
def _make_fact_store(conn, settings):
    if settings.semantic_store == "supabase":
        return SupabaseFactStore(settings)   # pgvector
    return SqliteFactStore(conn)             # default
```

---

## 5. Pillar IV — Eval & LLM-Ops (know it works, watch it think)

### Tracing: two outputs from the same events

The tracer is *also* a loop observer — pass `tracer.event` anywhere an observer goes and
every step lands in the trace.

1. **JSONL, always on.** Every turn appends readable lines to `traces/<date>.jsonl`. A trace
   is just "what happened, in order." Zero dependencies — open the file and read.
2. **OpenTelemetry spans, when an OTLP endpoint is set.** The same events as a span tree any
   OTel backend renders (Phoenix locally, Langfuse in the cloud). The instrumentation
   doesn't know or care which — you just set `OTEL_EXPORTER_OTLP_ENDPOINT`.

Keep a **separate permanent ledger** (`usage.jsonl`) for token spend — traces can be reset
for a clean demo, but the cost record should never be wiped. Store *tokens* (the ground
truth); derive dollars later, since pricing changes.

### Evals: two kinds that must never mix

This is the discipline most teams skip. Keep them in **separate directories** and never let
one masquerade as the other.

| | Deterministic | LLM-as-judge |
|---|---|---|
| Directory | `evals/deterministic/` | `evals/judge/` |
| Question | "did the right tool fire, with the right args?" | "was the reply good — helpful, used memory, right tone?" |
| Result | **0 or 1** (pass/fail) | **scored %** with a threshold |
| Nature | unit test | scored opinion |
| Runs | always (uses a scripted model — no key needed) | when an API key is present |

**Deterministic evals use the injectable seam.** A `ScriptedClient` plays back a fixed list
of model responses, so you test *your* code (loop, tools, wiring) with no network:

```python
def test_create_event_writes_db_and_ics(tmp_path):
    script = [gate_says_no, model_calls_create_event, model_says_booked]
    app = make_agent(tmp_path, client=ScriptedClient(script))
    app.respond("coffee with alex tuesday 9am")
    row = app.conn.execute("SELECT title, start FROM calendar_events").fetchone()
    assert row["title"] == "Coffee with Alex"
```

A **live tier** in the same file runs the *real* model over a `dataset.jsonl` of cases when a
key is present — that's the actual model-behavior eval. Offline tests your plumbing; live
tests the model+prompt.

> **When you find a live bug, fix it AND add a deterministic regression case.** The
> "triple-booked meeting" bug became `test_create_event_is_idempotent`. That's how the suite
> grows to match reality instead of rotting.

### The release gate: the diamond before "ship"

Changed the prompt? Swapped the model? Tuned retrieval? Run the gate:

```
deterministic evals  → must pass 100%  (they're unit tests; one failure blocks)
judge evals          → must clear threshold (runs when a key is present)
exit 0               → safe to release
```

The gate persists a verdict + appends to a run-history file, so you have a scoreboard of
every release decision. Wire it into CI and into your "ship a milestone" workflow.

---

## 6. Tools: the smallest possible contract

A tool is exactly three things — a name+description the model reads, a JSON schema for its
args, and a Python function. The registry runs them safely.

```python
@dataclass
class Tool:
    name: str
    description: str            # the model reads this to decide when to call
    input_schema: dict          # JSON schema for arguments
    fn: Callable[..., str]      # returns a STRING the model observes

class ToolRegistry:
    def execute(self, name, args) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.fn(**args)
        except Exception as exc:
            return f"Error running {name}: {exc}"   # observed, never crashes the loop
```

Tool design rules that pay off:

- **Return where the artifact landed.** Every tool's output states exactly what it did and
  where (which file, which table). The model relays it, so the agent never over-claims.
- **Be idempotent where it matters.** `create_event` refuses to write the same title+start
  twice — a confused model must not be able to triple-book.
- **Fail gracefully into text.** Empty/partial tool call? Return a helpful message the model
  can recover from, not a raw `TypeError`.
- **Group tools by origin and gate them.** Core tools always on; opt-in adapters (OS
  integrations, MCP servers, experimental sub-agents) behind flags so the default path
  stays lean.

**Scaling tools:** register more, group them, add MCP servers via a bridge (`mcp.json`).
The loop is indifferent to how many tools exist — it just passes `tools.schemas()` to the
model and runs what comes back.

---

## 7. How to scale — small → large, one boring step at a time

You do **not** rewrite to scale. You swap defaults for upgrades, one layer at a time, each
behind a config flag with the interface unchanged.

| Concern | Small (day one) | Medium | Large |
|---|---|---|---|
| **Store** | one SQLite file | SQLite + WAL, busy_timeout | Postgres / pgvector (factory swap) |
| **Semantic search** | FTS5 keyword (BM25) | FTS5 + reranking | vector DB + hybrid search |
| **Model** | one provider, env var | flagship + fast (gate on cheap model) | per-task routing, shootouts |
| **Gateway** | CLI | + web dashboard | + Telegram / Slack / API, all same seam |
| **Tracing** | JSONL file | + local Phoenix (OTel) | Langfuse / hosted OTel backend |
| **Evals** | deterministic (scripted) | + live dataset tier | + judge suite + CI gate |
| **Concurrency** | single connection | `check_same_thread=False` + lock | connection pool / worker processes |
| **Memory writes** | consolidate every N | async worker | queue + background consolidation |

The invariant: **the loop, the tool contract, and the memory facade never change.** They are
the stable core; everything else is a swappable default. That's what lets a repo grow
without getting muddier.

---

## 8. The design decisions worth stealing (checklist)

- [ ] **Loop under ~100 lines.** Complexity lives in tools and memory, not control flow.
- [ ] **Injectable `client` and `conn`.** The whole agent is testable with a fake model and a
      temp DB — no mocking framework.
- [ ] **Gateways move text only.** Every channel calls one `respond()`.
- [ ] **Observer seam.** Live UI, tracing, and cost ledger subscribe to loop events; the loop
      imports none of them.
- [ ] **Gate before retrieval.** A cheap-model judge decides *whether* to hit memory. Fail open.
- [ ] **Consolidation is batched, async, loss-safe.** Distill every N exchanges; never lose the
      log on failure.
- [ ] **Sliding-window working memory.** Bounded history; older turns come back via retrieval.
- [ ] **Fold tool activity into history** (`[tools used: ...]`) so the model knows it acted.
- [ ] **Tools return a string, errors included.** A tool crash is text, not an exception.
- [ ] **Idempotent side-effecting tools.** Same input, same effect, no duplicates.
- [ ] **Tools state where the artifact landed.** No over-claiming.
- [ ] **Two guardrails:** no-tool exit + hard iteration cap.
- [ ] **`max_tokens` headroom** for reasoning models (empty-reply trap otherwise).
- [ ] **Deterministic and judge evals never mix.** Unit test vs. scored opinion, separate dirs.
- [ ] **Every live bug → a deterministic regression case.**
- [ ] **Release gate:** deterministic 100% + judge threshold before ship.
- [ ] **Trace = JSONL always + OTel when configured.** Permanent, separate cost ledger.
- [ ] **One store file you can open.** Local-first; a human-readable mirror is a bonus.
- [ ] **Boring default + documented upgrade** for every layer. Default is zero-signup.
- [ ] **Providers framed neutrally.** Model IDs are strings; swap with an env var.

---

## 9. Minimum viable version (the afternoon build)

If you build only this, you have a real agent:

1. `config.py` — a `Settings` dataclass reading env vars.
2. `db.py` — one SQLite file with a `facts` table (+ FTS5) and a `chat_log`.
3. `loop/agent.py` — the ~95-line loop above.
4. `loop/models.py` — one provider (start with what you pay for).
5. `tools/registry.py` + one real tool that returns a string.
6. `runtime/session.py` — assemble system prompt + windowed history + new message.
7. `memory/` — a fact store with `.add()`/`.search()`, a retrieval gate, consolidation.
8. `ops/tracing.py` — append JSONL per turn.
9. `app.py` — wire it together; expose `respond()`.
10. `gateway/cli.py` — a `while` loop around `respond()`.
11. `evals/deterministic/` — a `ScriptedClient` and three assertions.

Then grow it one boring default at a time — never more than one layer per change, always
with the tests green and the gate open.

---

*Derived from the [waku-agent](https://github.com/ShenSeanChen/waku-agent) architecture —
Harness · Loop · Memory · Eval/LLM-Ops, in code you can read in an afternoon.*
