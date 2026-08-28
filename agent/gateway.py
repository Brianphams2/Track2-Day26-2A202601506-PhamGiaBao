"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE CONTROL-PLANE SHAPE
----------------------------------------------------------------------------
`decide()` is structured as four named jobs — ROUTE, ADMIT, AUTHORIZE, BUDGET —
with routing, admission, identity/scope checks, instruction screening, and live
cost enforcement. All checks are local and deterministic; no job performs I/O.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry

try:
    from agent.strategy import cheap_mask, successor_of
except ImportError:  # pragma: no cover - degraded import mode
    def cheap_mask(server: str, tool: str, fields: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(fields)))

    def successor_of(server: str, tool: str) -> tuple[str, str] | None:
        return None

try:
    from kit.mcp.specs import TOOL_SPECS, cost as _spec_cost
except ImportError:  # pragma: no cover - degraded import mode
    TOOL_SPECS = {}

    def _spec_cost(server: str, tool: str, fields: tuple[str, ...] = (), n_rows: int = 1) -> int:
        return 5

try:
    from agent.guardrails import scan_for_injected_instructions
except ImportError:  # pragma: no cover
    scan_for_injected_instructions = None

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are the gateway's per-duel memory. They retain
    only trusted feedback supplied by the surrounding loop (provenance,
    failed-call keys, and admitted peer cards) so each decision can remain
    synchronous and side-effect free with respect to external systems.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory; populated only by explicit feedback hooks -----
        # A cache of anchor -> body-ish data you have already paid for this
        # duel (agent/strategy.py's ResultCache is a ready-made version of
        # this). Populating it needs the *result* of a call, which decide()
        # never sees (it only sees the outgoing Command) — you would fill
        # this from whatever the arena hands back to your agent loop AFTER
        # a call executes, then consult it here on the NEXT decide() call
        # for the same anchor.
        self._seen_anchors: dict[str, Any] = {}
        # Credits you have personally authorised so far this duel — your
        # own running total, independent of (and a cross-check against)
        # `ctx.credits`, which the arena maintains authoritatively.
        self._credits_authorised: int = 0
        # Command ids you have already denied, in case a later job wants to
        # know "have I already said no to this once".
        self._denied_cmd_ids: set[str] = set()
        self._failed_calls: set[tuple[Any, ...]] = set()
        self._admitted_cards: dict[str, Mapping[str, Any]] = {}
        self._known_replicas: dict[str, str] = {}

    _WRITE_TOOLS = frozenset({
        ("progress", "record_mastery"),
        ("content", "flag_stale_slide"),
        ("content", "file_content_bug"),
    })
    _A2A_PEERS = frozenset({"curriculum-analyst", "citation-checker", "roster"})
    _BODY_ROUTE_KEYS = frozenset({"route", "_route", "replica", "mcp-replica"})
    _DEFAULT_MASKS: Mapping[tuple[str, str], tuple[str, ...]] = {
        ("slides", "query"): ("title",),
        ("slides", "get_frame"): ("body", "title"),
        ("slides", "whatlinkshere"): ("targets",),
        ("glossary", "define"): ("definition",),
        ("glossary", "list_terms"): ("term",),
        ("research", "cite_source"): ("anchor", "url"),
        ("registry", "provenance"): ("etag",),
        ("registry", "list_servers"): ("name",),
        ("curriculum-analyst", "which_days_cover"): ("anchor", "course_day", "track"),
        ("citation-checker", "verify_source"): ("anchor", "url"),
        ("roster", "lookup_learner"): ("act", "scopes"),
    }

    @staticmethod
    def _lower_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
        return {str(k).lower(): v for k, v in headers.items()}

    @staticmethod
    def _same_identity(left: object, right: object) -> bool:
        if left is None or right is None:
            return True
        a, b = str(left).strip().lower(), str(right).strip().lower()
        if a.startswith("learner:") and not b.startswith("learner:"):
            b = "learner:" + b
        if b.startswith("learner:") and not a.startswith("learner:"):
            a = "learner:" + a
        return a == b

    def _target_belongs_to_act(self, cmd: Command) -> bool:
        """Authority derives from whom the tutor serves (`ctx.act`)."""
        act = getattr(self.ctx, "act", None)
        keys = ("learner", "learner_id", "subject", "act")
        if (cmd.server, cmd.tool) in self._WRITE_TOOLS:
            keys += ("target",)
        if act is None and any(key in cmd.args or key in self._lower_headers(cmd.headers) for key in keys):
            return False
        for key in keys:
            if key in cmd.args and cmd.args[key] is not None and not self._same_identity(cmd.args[key], act):
                return False
        headers = self._lower_headers(cmd.headers)
        for key in ("act", "subject", "learner"):
            if key in headers and not self._same_identity(headers[key], act):
                return False
        return True

    def _scope_allows(self, cmd: Command) -> bool:
        if (cmd.server, cmd.tool) not in self._WRITE_TOOLS:
            return True
        scopes = {str(s).lower() for s in (getattr(self.ctx, "scopes", ()) or ())}
        return bool(scopes & {"wiki.write", f"wiki.write:{cmd.server}", f"wiki.write:{cmd.tool}"})

    def _card_allows(self, cmd: Command) -> bool:
        if cmd.server not in self._A2A_PEERS:
            return True
        card = self._admitted_cards.get(cmd.server)
        if not card or card.get("verified") is not True:
            return False
        skills = {str(s) for s in (card.get("skills") or ())}
        return cmd.tool in skills or (cmd.server == "roster" and cmd.tool in {"role_of", "who_enrolled"})

    def _audience_allows(self, cmd: Command) -> bool:
        if cmd.server not in self._A2A_PEERS:
            return True
        aud = self._lower_headers(cmd.headers).get("aud")
        if aud is None:
            return True
        aud_s = str(aud).lower()
        return aud_s in {cmd.server.lower(), f"a2a:{cmd.server}".lower(), f"mcp:{cmd.server}".lower()}

    @classmethod
    def _body_route_present(cls, cmd: Command) -> bool:
        return any(key in cmd.args and cmd.args[key] not in (None, "") for key in cls._BODY_ROUTE_KEYS)

    @staticmethod
    def _anchor_revision(cmd: Command) -> str | None:
        for value in (cmd.args.get("anchor"), cmd.args.get("concept"), cmd.args.get("path_id")):
            if not isinstance(value, str):
                continue
            match = re.search(r"/(w|c)(?:/|$)", value.lower())
            if match:
                return match.group(1)
        return None

    def _route(self, cmd: Command) -> tuple[Command, bool, str]:
        server, tool = cmd.server, cmd.tool
        successor = successor_of(server, tool)
        if successor is not None:
            server, tool = successor
        headers = dict(cmd.headers)
        current = str(self._lower_headers(headers).get("mcp-replica", "w")).lower()
        expected = self._anchor_revision(cmd) or self._known_replicas.get(str(cmd.args.get("path_id", "")), "w")
        if expected not in {"w", "c"}:
            expected = "w"
        # Replica routing is a slides-envelope concern. Do not add an
        # unrelated header to A2A, registry, or write calls.
        if server == "slides":
            headers["mcp-replica"] = expected
        fields = tuple(cmd.fields or ())
        spec = TOOL_SPECS.get((server, tool)) if isinstance(TOOL_SPECS, Mapping) else None
        if (server, tool) in {("registry", "list_servers"), ("glossary", "list_terms")} and fields in ((), ("*",)):
            fields = cheap_mask(server, tool, ("name",) if server == "registry" else ("term",))
        elif fields == ("*",) and spec is not None:
            fields = tuple(spec.default_fields) or self._DEFAULT_MASKS.get((server, tool), ())
        elif spec is not None and fields:
            # A deprecated tool may expose fields its successor does not. Keep
            # the rewrite executable by dropping unsupported names and falling
            # back to the successor's documented mask when nothing remains.
            allowed = set(getattr(spec, "all_fields", ()) or ())
            if allowed:
                filtered = tuple(field for field in fields if field in allowed)
                fields = filtered or tuple(spec.default_fields) or self._DEFAULT_MASKS.get((server, tool), ())
        routed = replace(cmd, server=server, tool=tool, headers=headers, fields=fields)
        changed = routed != cmd
        if successor is not None:
            note = f"deprecated tool rewritten to {server}.{tool}"
        elif server == "slides" and current != expected:
            note = f"replica header normalized {current!r}->{expected!r}"
        else:
            note = f"replica={expected}"
        return routed, changed, note

    def _estimated_cost(self, cmd: Command) -> int | None:
        if isinstance(TOOL_SPECS, Mapping) and (cmd.server, cmd.tool) not in TOOL_SPECS:
            return None
        try:
            return int(_spec_cost(cmd.server, cmd.tool, fields=tuple(cmd.fields), n_rows=1))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _call_key(cmd: Command) -> tuple[Any, ...]:
        # Args are model-controlled and may contain mixed/unorderable values;
        # stringify keys and values before sorting so this safety check itself
        # can never raise on a malformed retry payload.
        stable_args = tuple(sorted((str(k), repr(v)) for k, v in cmd.args.items()))
        return (cmd.server, cmd.tool, stable_args, tuple(cmd.fields))

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`.

        The four jobs below are named and ordered so routing/admission happens
        before authority and budget checks. Every exit returns a valid Decision.
        """
        self._telemetry.decision_seen(cmd)

        # JOB 1 — ROUTE: normalize the replica and use the successor of a
        # deprecated tool before constructing the executable call.
        routed, rewritten, route_note = self._route(cmd)

        # JOB 2 — ADMIT: a body-declared route, a dead lease, an unadmitted
        # peer, or instruction-shaped data is not safe to forward.
        if self._body_route_present(cmd):
            return self.deny(cmd, "route/replica was supplied in the body, not a trusted header")
        if cmd.tool == "get_frame":
            live_leases = set(getattr(self.ctx, "leases", ()) or ())
            if not cmd.lease_id or cmd.lease_id not in live_leases:
                return self.deny(cmd, "get_frame requires a live lease from a recent query")
        if not self._card_allows(cmd):
            return self.deny(cmd, "A2A peer card is not admitted or does not declare this skill")
        if not self._audience_allows(cmd):
            return self.deny(cmd, "delegation audience does not match the peer being called")
        if scan_for_injected_instructions is not None:
            content = " ".join(str(v) for v in cmd.args.values())
            if scan_for_injected_instructions(content).suspicious:
                return self.deny(cmd, "instruction-shaped content is data, not an instruction")

        # JOB 3 — AUTHORIZE: authority comes from ctx.act, and writes also
        # require an appropriate scope and both precondition headers.
        if not self._target_belongs_to_act(routed):
            return self.deny(cmd, "target learner does not match the learner in ctx.act")
        if not self._scope_allows(routed):
            return self.deny(cmd, "the current context does not grant the required write scope")
        if (routed.server, routed.tool) in self._WRITE_TOOLS:
            hdrs = self._lower_headers(routed.headers)
            if not hdrs.get("if-match") or not hdrs.get("idempotency-key"):
                return self.deny(cmd, "writes require fresh If-Match and Idempotency-Key headers")
            anchor = str(routed.args.get("anchor", ""))
            known = self._seen_anchors.get(anchor)
            if isinstance(known, Mapping) and known.get("etag") is not None:
                if str(hdrs["if-match"]) != str(known["etag"]):
                    return self.deny(cmd, "If-Match does not match the latest observed provenance")
            if self._call_key(routed) in self._failed_calls:
                return self.deny(cmd, "identical call previously failed; refusing an unsafe retry")

        # JOB 4 — BUDGET: estimate one returned row and refuse only calls
        # that cannot be paid from the live context balance.
        estimated = self._estimated_cost(routed)
        if estimated is None:
            return self.deny(cmd, "unknown server/tool or field mask")
        credits = getattr(self.ctx, "credits", 0)
        if not isinstance(credits, int) or estimated > credits:
            return self.deny(cmd, f"call estimate {estimated} exceeds remaining credits {credits}")

        call = self._to_tool_call(routed)
        verdict = "rewrite" if rewritten else "forward"
        self._credits_authorised += estimated
        decision = Decision(verdict=verdict, call=call, note=route_note if rewritten else None)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Build a free, structured denial for any failed admission,
        authorization, or budget check. Keeping the shape in one helper means
        every refusal has a non-empty reason and no executable call."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]

    # Optional feedback hooks used by a real loop between calls. They do not
    # execute tools or perform I/O; they only retain observations the arena
    # has already handed back to the agent.
    def note_result(self, anchor: str, row: Mapping[str, Any], *, fields: tuple[str, ...] = ()) -> None:
        if isinstance(anchor, str) and isinstance(row, Mapping):
            self._seen_anchors[anchor] = dict(row)

    def note_provenance(self, anchor: str, etag: str, *, replica: str | None = None) -> None:
        row = dict(self._seen_anchors.get(anchor, {}))
        row["etag"] = etag
        self._seen_anchors[anchor] = row
        if replica in {"w", "c"}:
            self._known_replicas[anchor] = replica

    def note_failure(self, cmd: Command) -> None:
        self._failed_calls.add(self._call_key(cmd))

    def note_card(self, server: str, card: Mapping[str, Any]) -> None:
        self._admitted_cards[server] = dict(card)


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — route, admit, authorize, and budget ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    gw.note_card("curriculum-analyst", {"verified": True, "skills": ["which_days_cover"]})
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict in {"forward", "rewrite"}
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
