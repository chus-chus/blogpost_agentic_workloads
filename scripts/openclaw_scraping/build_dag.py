"""Build a DAG of LLM inference requests from OpenClaw session files."""

from __future__ import annotations

from pathlib import Path

from parse_session import (
    classify_message,
    estimate_message_tokens,
    find_session_file,
    parse_jsonl,
    resolve_session_files,
)


def _find_spawn_child_key_from_result(entries: list[dict], tool_call_id: str) -> str | None:
    """Find the childSessionKey from the tool result matching a sessions_spawn call."""
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") == "toolResult" and msg.get("toolCallId") == tool_call_id:
            details = msg.get("details", {})
            return details.get("childSessionKey")
    return None


def _get_subagent_session_key_from_injection(entry: dict) -> str | None:
    """Extract the child session key from an injected subagent completion message."""
    msg = entry.get("message", {})
    content = msg.get("content", [])
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))

    if "session_key:" not in text:
        return None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("session_key:"):
            return line.split(":", 1)[1].strip()
    return None


def _extract_child_refs(entries: list[dict]) -> list[dict]:
    """Extract child session references from injection messages in a session's entries.

    Returns list of dicts with keys: session_id, session_key.
    """
    refs: list[dict] = []
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        text = ""
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if "[Internal task completion event]" not in text:
            continue
        child_key = None
        child_sid = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("session_key:"):
                child_key = line.split(":", 1)[1].strip()
            elif line.startswith("session_id:"):
                child_sid = line.split(":", 1)[1].strip()
        if child_key and child_sid:
            refs.append({"session_id": child_sid, "session_key": child_key})
    return refs


def _discover_child_sessions(
    sessions_dir: Path,
    session_lookup: dict[str, dict],
    session_entries: dict[str, list[dict]],
) -> None:
    """Discover child sessions by following spawn references in loaded entries.

    Walks the session tree starting from already-loaded sessions, finding
    child session files that sessions.json may no longer reference (e.g.
    cleaned-up subagent sessions). Mutates session_lookup and session_entries.
    """
    # Iterate until no new sessions are discovered
    while True:
        new_found = False
        for sid in list(session_entries.keys()):
            parent_key = session_lookup[sid]["session_key"]
            for ref in _extract_child_refs(session_entries[sid]):
                child_sid = ref["session_id"]
                if child_sid in session_lookup:
                    continue
                # Find the child's file on disk
                fpath = find_session_file(sessions_dir, child_sid)
                if not fpath or not Path(fpath).exists():
                    continue
                entries = parse_jsonl(fpath)
                # Recover model info from the child's entries
                model = None
                provider = None
                for entry in entries:
                    msg = entry.get("message", {})
                    if msg.get("role") == "assistant" and msg.get("model"):
                        model = msg["model"]
                        provider = msg.get("provider")
                        break
                session_lookup[child_sid] = {
                    "session_key": ref["session_key"],
                    "session_id": child_sid,
                    "file": fpath,
                    "spawned_by": parent_key,
                    "model": model,
                    "provider": provider,
                    "label": None,
                    "spawn_depth": None,
                }
                session_entries[child_sid] = entries
                new_found = True
        if not new_found:
            break


def build_dag(sessions_json_path: str | Path) -> dict:
    """Build the full DAG from a sessions.json file.

    Returns dict with keys: sessions, nodes, edges.
    """
    from datetime import datetime, timezone

    def _parse_ts(ts: str | None) -> datetime | None:
        if not ts:
            return None
        ts_clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_clean)

    def _epoch_to_dt(epoch_ms: int | float) -> datetime:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

    def _epoch_to_iso(epoch_ms: int | float) -> str:
        return _epoch_to_dt(epoch_ms).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(epoch_ms % 1000):03d}Z"

    sessions_json_path = Path(sessions_json_path)
    session_lookup = resolve_session_files(sessions_json_path)

    # Load all session JSONL files
    session_entries: dict[str, list[dict]] = {}
    for sid, meta in session_lookup.items():
        fpath = meta["file"]
        if fpath and Path(fpath).exists():
            session_entries[sid] = parse_jsonl(fpath)

    # Discover child sessions not in sessions.json by following spawn
    # references in loaded entries (e.g. cleaned-up subagent sessions).
    _discover_child_sessions(sessions_json_path.parent, session_lookup, session_entries)

    # Build nodes and edges
    nodes: list[dict] = []
    edges: list[dict] = []

    # Index: session_key -> session_id
    key_to_sid: dict[str, str] = {}
    for sid, meta in session_lookup.items():
        key_to_sid[meta["session_key"]] = sid

    # Process each session
    # Track: for each session, the ordered list of assistant nodes
    session_nodes: dict[str, list[dict]] = {}
    # Track: session_key -> first node id, last node id
    session_first_node: dict[str, str] = {}
    session_last_node: dict[str, str] = {}

    for sid, entries in session_entries.items():
        meta = session_lookup[sid]
        session_key = meta["session_key"]

        # Find all assistant messages with non-zero usage
        assistant_indices: list[int] = []
        for i, entry in enumerate(entries):
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage", {})
            if usage.get("totalTokens", 0) == 0:
                continue
            assistant_indices.append(i)

        prev_node_id: str | None = None

        for node_idx, ai in enumerate(assistant_indices):
            entry = entries[ai]
            msg = entry["message"]
            usage = msg.get("usage", {})

            node_id = f"{sid}:{entry['id']}"
            total_input = usage.get("input", 0) + usage.get("cacheRead", 0)
            output = usage.get("output", 0)

            # Find messages between previous assistant node and this one
            if node_idx == 0:
                between_start = 0
                from_previous_assistant = 0
            else:
                between_start = assistant_indices[node_idx - 1] + 1
                prev_assistant_msg = entries[assistant_indices[node_idx - 1]].get(
                    "message", {}
                )
                # Use the visible assistant payload rather than usage.output,
                # which can include hidden reasoning tokens.
                from_previous_assistant = estimate_message_tokens(
                    prev_assistant_msg,
                    model=msg.get("model"),
                    include_thinking=True,
                )
            between_end = ai  # exclusive

            # Estimate tokens directly from the intervening messages instead of
            # relying on input/cache deltas, which vary with prompt caching.
            token_counts = {"tool_result": 0, "user_input": 0, "injected": 0}
            for bi in range(between_start, between_end):
                bentry = entries[bi]
                cat = classify_message(bentry)
                if cat in token_counts:
                    token_counts[cat] += estimate_message_tokens(
                        bentry.get("message", {}),
                        model=msg.get("model"),
                    )

            from_tool_results = token_counts["tool_result"]
            from_user_input = token_counts["user_input"]
            from_injected = token_counts["injected"]
            other_tokens = 0

            if node_idx == 0:
                # Session logs omit the hidden scaffold (system/developer/tool
                # schema prompt). Recover that once from the provider-reported
                # first-turn prompt size.
                visible_first_turn_tokens = (
                    from_tool_results
                    + from_user_input
                    + from_injected
                    + from_previous_assistant
                )
                hidden_scaffold_tokens = max(0, total_input - visible_first_turn_tokens)
                from_injected += hidden_scaffold_tokens

            # Detect sessions_spawn calls
            spawns_subagent = None
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    if block.get("name") == "sessions_spawn":
                        tool_call_id = block.get("id", "")
                        child_key = _find_spawn_child_key_from_result(
                            entries, tool_call_id
                        )
                        if child_key:
                            spawns_subagent = child_key

            # Detect subagent result injection
            receives_subagent_result = None
            # Look at the user message immediately before this assistant message
            for bi in range(ai - 1, max(between_start - 1, -1), -1):
                bentry = entries[bi]
                if classify_message(bentry) == "injected":
                    receives_subagent_result = _get_subagent_session_key_from_injection(
                        bentry
                    )
                    break

            # Determine what triggered this LLM call
            if node_idx == 0:
                # First call in session — look at messages before it
                has_user = token_counts["user_input"] > 0
                trigger = "user" if has_user else "system"
            elif token_counts["user_input"] > 0 and token_counts["injected"] > 0:
                trigger = "subagent_result"
            elif token_counts["user_input"] > 0:
                trigger = "user"
            elif token_counts["tool_result"] > 0:
                trigger = "tool_result"
            elif token_counts["injected"] > 0:
                trigger = "subagent_result"
            else:
                trigger = "continuation"

            # message.timestamp (epoch ms) = dispatch time
            # entry timestamp (ISO) = response time
            dispatch_epoch = msg.get("timestamp")
            response_iso = entry.get("timestamp")
            inference_ms = None
            if dispatch_epoch and response_iso:
                resp_dt = _parse_ts(response_iso)
                disp_dt = _epoch_to_dt(dispatch_epoch)
                if resp_dt and disp_dt:
                    inference_ms = round((resp_dt - disp_dt).total_seconds() * 1000)

            node = {
                "id": node_id,
                "session_id": sid,
                "session_key": session_key,
                "entry_id": entry["id"],
                "timestamp_dispatched": _epoch_to_iso(dispatch_epoch) if dispatch_epoch else response_iso,
                "timestamp_responded": response_iso,
                "inference_ms": inference_ms,
                "model": msg.get("model"),
                "provider": msg.get("provider"),
                "trigger": trigger,
                "stop_reason": msg.get("stopReason"),
                "usage": {
                    "input": usage.get("input", 0),
                    "output": output,
                    "cache_read": usage.get("cacheRead", 0),
                    "cache_write": usage.get("cacheWrite", 0),
                    "total_tokens": usage.get("totalTokens", 0),
                },
                "new_tokens": {
                    "from_tool_results": from_tool_results,
                    "from_user_input": from_user_input,
                    "from_previous_assistant": from_previous_assistant,
                    "from_injected": from_injected,
                    "other": other_tokens,
                    "total": (
                        from_tool_results
                        + from_user_input
                        + from_previous_assistant
                        + from_injected
                        + other_tokens
                    ),
                },
                "spawns_subagent": spawns_subagent,
                "receives_subagent_result": receives_subagent_result,
            }
            nodes.append(node)

            # Sequential edge within session
            if prev_node_id is not None:
                edges.append(
                    {"from": prev_node_id, "to": node_id, "type": "sequential", "is_history_parent": True}
                )

            # Track first/last
            if node_idx == 0:
                session_first_node[session_key] = node_id
            session_last_node[session_key] = node_id

            prev_node_id = node_id

        session_nodes[session_key] = [
            n for n in nodes if n["session_key"] == session_key
        ]

    # Add cross-session edges (subagent_spawn and subagent_result)
    for node in nodes:
        if node["spawns_subagent"]:
            child_key = node["spawns_subagent"]
            if child_key in session_first_node:
                edges.append(
                    {
                        "from": node["id"],
                        "to": session_first_node[child_key],
                        "type": "subagent_spawn",
                        "is_history_parent": False,
                    }
                )

        if node["receives_subagent_result"]:
            child_key = node["receives_subagent_result"]
            if child_key in session_last_node:
                edges.append(
                    {
                        "from": session_last_node[child_key],
                        "to": node["id"],
                        "type": "subagent_result",
                        "is_history_parent": False,
                    }
                )

    # Compute wait_after_ready for each node.
    # ready_time = max(response timestamp of all parent nodes)
    # dispatch_time = this node's dispatch timestamp
    # wait = dispatch_time - ready_time

    node_by_id = {n["id"]: n for n in nodes}
    # Build incoming edge map: only history-parent edges determine dispatch readiness
    incoming: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for edge in edges:
        if edge.get("is_history_parent") and edge["to"] in incoming:
            incoming[edge["to"]].append(edge["from"])

    for node in nodes:
        parents = incoming[node["id"]]
        if not parents:
            node["wait_after_ready_ms"] = None
            continue
        # Dispatch time of this node
        dispatch_ts = _parse_ts(node["timestamp_dispatched"])
        if dispatch_ts is None:
            node["wait_after_ready_ms"] = None
            continue
        # Ready time = max response time of all parent nodes
        parent_response_times = []
        for pid in parents:
            pts = _parse_ts(node_by_id[pid]["timestamp_responded"])
            if pts is not None:
                parent_response_times.append(pts)
        if not parent_response_times:
            node["wait_after_ready_ms"] = None
            continue
        ready_time = max(parent_response_times)
        node["wait_after_ready_ms"] = round((dispatch_ts - ready_time).total_seconds() * 1000)

    # Build sessions summary
    sessions_summary = {}
    # Group nodes by session to compute per-session stats
    nodes_by_session: dict[str, list[dict]] = {}
    for n in nodes:
        nodes_by_session.setdefault(n["session_id"], []).append(n)

    for sid, meta in session_lookup.items():
        snodes = nodes_by_session.get(sid, [])
        sess_max_ctx = max(
            (n["usage"]["input"] + n["usage"]["cache_read"] for n in snodes),
            default=0,
        )
        sessions_summary[sid] = {
            "session_key": meta["session_key"],
            "model": meta["model"],
            "provider": meta["provider"],
            "spawned_by": meta["spawned_by"],
            "label": meta["label"],
            "file": meta["file"],
            "max_context_length": sess_max_ctx,
        }

    # ---- helpers for distribution stats ----
    def _dist(values: "list[int] | list[float] | list[int | float]") -> dict:
        """Return min/max/mean/median/p95/sum for a list of numbers."""
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None, "p95": None, "sum": None, "count": 0}
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
            "median": s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 1),
            "p95": s[min(int(n * 0.95), n - 1)],
            "sum": sum(s),
            "count": n,
        }

    # Compute high-level stats
    total_input_tokens = sum(n["usage"]["input"] for n in nodes)
    total_output_tokens = sum(n["usage"]["output"] for n in nodes)
    total_cache_read = sum(n["usage"]["cache_read"] for n in nodes)
    total_cache_write = sum(n["usage"]["cache_write"] for n in nodes)
    max_context = max((n["usage"]["input"] + n["usage"]["cache_read"] for n in nodes), default=0)

    trigger_counts: dict[str, int] = {}
    stop_counts: dict[str, int] = {}
    for n in nodes:
        trigger_counts[n["trigger"]] = trigger_counts.get(n["trigger"], 0) + 1
        stop_counts[n["stop_reason"]] = stop_counts.get(n["stop_reason"], 0) + 1

    edge_type_counts: dict[str, int] = {}
    for e in edges:
        edge_type_counts[e["type"]] = edge_type_counts.get(e["type"], 0) + 1

    # Wall time: from first dispatch to last response
    dispatch_ts = [n["timestamp_dispatched"] for n in nodes if n["timestamp_dispatched"]]
    response_ts = [n["timestamp_responded"] for n in nodes if n["timestamp_responded"]]
    duration_s = None
    if dispatch_ts and response_ts:
        t_first = _parse_ts(min(dispatch_ts))
        t_last = _parse_ts(max(response_ts))
        if t_first and t_last:
            duration_s = (t_last - t_first).total_seconds()

    inference_times = [n["inference_ms"] for n in nodes if n["inference_ms"] is not None]
    total_inference_ms = sum(inference_times) if inference_times else None

    # Per-session stats
    for sid, meta in sessions_summary.items():
        snodes = nodes_by_session.get(sid, [])
        s_dispatch = [n["timestamp_dispatched"] for n in snodes if n["timestamp_dispatched"]]
        s_response = [n["timestamp_responded"] for n in snodes if n["timestamp_responded"]]
        s_dur = None
        if s_dispatch and s_response:
            t0 = _parse_ts(min(s_dispatch))
            t1 = _parse_ts(max(s_response))
            if t0 and t1:
                s_dur = round((t1 - t0).total_seconds(), 3)
        meta["num_nodes"] = len(snodes)
        meta["wall_time_s"] = s_dur

    # Distribution stats for tokens and timing
    new_token_values = [n["new_tokens"]["total"] for n in nodes]
    output_token_values = [n["usage"]["output"] for n in nodes]
    wait_values = [n["wait_after_ready_ms"] for n in nodes if n["wait_after_ready_ms"] is not None]
    inference_values = [n["inference_ms"] for n in nodes if n["inference_ms"] is not None]

    # Fan-out / fan-in degree per node
    fan_out: dict[str, int] = {}
    fan_in: dict[str, int] = {}
    for e in edges:
        fan_out[e["from"]] = fan_out.get(e["from"], 0) + 1
        fan_in[e["to"]] = fan_in.get(e["to"], 0) + 1
    fan_out_values = [fan_out.get(n["id"], 0) for n in nodes]
    fan_in_values = [fan_in.get(n["id"], 0) for n in nodes]

    stats = {
        "num_sessions": len(sessions_summary),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "triggers": trigger_counts,
        "stop_reasons": stop_counts,
        "edge_types": edge_type_counts,
        "tokens": {
            "total_input": total_input_tokens,
            "total_output": total_output_tokens,
            "total_cache_read": total_cache_read,
            "total_cache_write": total_cache_write,
            "total": total_input_tokens + total_output_tokens + total_cache_read + total_cache_write,
            "max_context_length": max_context,
        },
        "new_tokens": _dist(new_token_values),
        "output_tokens": _dist(output_token_values),
        "wait_after_ready_ms": _dist(wait_values),
        "inference_ms": _dist(inference_values),
        "fan_out": _dist(fan_out_values),
        "fan_in": _dist(fan_in_values),
        "wall_time_s": duration_s,
        "total_inference_ms": total_inference_ms,
    }

    return {"stats": stats, "sessions": sessions_summary, "nodes": nodes, "edges": edges}


def dag_to_dot(dag: dict) -> str:
    """Convert a DAG dict to Graphviz DOT format."""
    lines = ["digraph SessionDAG {", "  rankdir=TB;", '  node [shape=box, fontsize=10];']

    # Color by session
    session_colors = {}
    palette = ["#4A90D9", "#E6994A", "#5CB85C", "#D94A4A", "#9B59B6", "#1ABC9C"]
    for i, sid in enumerate(dag["sessions"]):
        session_colors[sid] = palette[i % len(palette)]

    for node in dag["nodes"]:
        sid = node["session_id"]
        color = session_colors.get(sid, "#999999")
        wait = node.get("wait_after_ready_ms")
        wait_str = f"wait={wait}ms" if wait is not None else "wait=-"
        label_parts = [
            node["entry_id"][:8],
            wait_str,
            f"trigger={node['trigger']}",
            f"out={node['usage']['output']}tok",
            f"new={node['new_tokens']['total']}tok",
            f"stop={node['stop_reason']}",
        ]
        if node["spawns_subagent"]:
            label_parts.append("SPAWN")
        if node["receives_subagent_result"]:
            label_parts.append("RECV")
        label = "\\n".join(label_parts)
        safe_id = node["id"].replace("-", "_").replace(":", "__")
        lines.append(
            f'  "{safe_id}" [label="{label}", style=filled, fillcolor="{color}40"];'
        )

    edge_styles = {
        "sequential": "",
        "subagent_spawn": '[style=dashed, color="#E6994A", label="spawn"]',
        "subagent_result": '[style=dashed, color="#5CB85C", label="result"]',
    }
    for edge in dag["edges"]:
        src = edge["from"].replace("-", "_").replace(":", "__")
        dst = edge["to"].replace("-", "_").replace(":", "__")
        style = edge_styles.get(edge["type"], "")
        lines.append(f'  "{src}" -> "{dst}" {style};')

    lines.append("}")
    return "\n".join(lines)


def dag_summary_table(dag: dict) -> str:
    """Produce a human-readable summary table."""
    lines: list[str] = []

    lines.append("=" * 110)
    lines.append("SESSION DAG SUMMARY")
    lines.append("=" * 110)

    # Sessions
    lines.append("\nSessions:")
    for sid, meta in dag["sessions"].items():
        spawned = f" (spawned by {meta['spawned_by']})" if meta["spawned_by"] else " (root)"
        lines.append(f"  {sid[:12]}... [{meta['session_key']}] model={meta['model']}{spawned}")

    # Nodes
    lines.append(f"\nNodes ({len(dag['nodes'])} total):")
    header = (
        f"  {'ID':<24} {'Dispatched':>12} {'Infer':>7} {'Wait':>7} {'Trigger':>14} {'Stop':>8} "
        f"{'In':>7} {'Out':>6} {'Cache':>7} {'NewTok':>7} "
        f"{'ToolR':>6} {'User':>6} {'Asst':>6} {'Other':>6} {'Flags'}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for node in dag["nodes"]:
        nid = node["id"]
        # Shorten: first 8 of session + entry_id
        short_id = nid[:8] + ":" + node["entry_id"][:8]
        disp = node.get("timestamp_dispatched", "")
        ts = disp[11:23] if disp else "?"
        nt = node["new_tokens"]
        infer = node.get("inference_ms")
        if infer is None:
            infer_str = "-"
        elif infer < 1000:
            infer_str = f"{infer}ms"
        else:
            infer_str = f"{infer / 1000:.1f}s"
        wait = node.get("wait_after_ready_ms")
        if wait is None:
            wait_str = "-"
        elif wait < 1000:
            wait_str = f"{wait}ms"
        else:
            wait_str = f"{wait / 1000:.1f}s"
        flags = []
        if node["spawns_subagent"]:
            flags.append("S")
        if node["receives_subagent_result"]:
            flags.append("R")
        flag_str = ",".join(flags) if flags else ""
        lines.append(
            f"  {short_id:<24} {ts:>12} {infer_str:>7} {wait_str:>7} {node['trigger']:>14} {node['stop_reason']:>8} "
            f"{node['usage']['input']:>7} {node['usage']['output']:>6} "
            f"{node['usage']['cache_read']:>7} {nt['total']:>7} "
            f"{nt['from_tool_results']:>6} {nt['from_user_input']:>6} "
            f"{nt['from_previous_assistant']:>6} {nt['other']:>6} {flag_str}"
        )

    # Edges
    edge_counts = {}
    for edge in dag["edges"]:
        edge_counts[edge["type"]] = edge_counts.get(edge["type"], 0) + 1
    lines.append(f"\nEdges ({len(dag['edges'])} total):")
    for etype, count in sorted(edge_counts.items()):
        lines.append(f"  {etype}: {count}")

    # Cross-session edges
    cross = [e for e in dag["edges"] if e["type"] != "sequential"]
    if cross:
        lines.append("\nCross-session edges:")
        for e in cross:
            lines.append(f"  {e['from'][:20]}... -> {e['to'][:20]}... [{e['type']}]")

    lines.append("")
    return "\n".join(lines)
