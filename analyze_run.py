from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse
import re

EXPERIMENT_ID_RE = re.compile(r"exp_\d{8}_[0-9a-f]+", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_values[low]
    frac = idx - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def _basic_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
        "avg": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(_percentile(ordered, 0.95)),
    }


def _header_ci(headers: dict[str, Any], name: str) -> Any:
    target = name.lower()
    for key, value in (headers or {}).items():
        if str(key).lower() == target:
            return value
    return None


def _normalize_status(value: Any) -> str:
    text = str(value or "unknown")
    if "." in text:
        text = text.split(".")[-1]
    return text.lower()


def _normalize_endpoint(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""

    path = re.sub(
        r"/research/exp_\d{8}_[0-9a-f]+/files/.+",
        "/research/{experiment_id}/files/{file_path}",
        path,
        flags=re.IGNORECASE,
    )
    path = EXPERIMENT_ID_RE.sub("{experiment_id}", path)

    if parsed.query:
        query_keys = sorted({k for k, _ in parse_qsl(parsed.query, keep_blank_values=True)})
        if query_keys:
            return f"{path}?{'&'.join(query_keys)}"
    return path


def _extract_response_data(body: Any) -> Any:
    if isinstance(body, dict) and "data" in body:
        return body.get("data")
    return body


def _walk_request_envelopes(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "url" in value and "status_code" in value and "headers" in value:
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return found


def _catalog_api_endpoints(project_root: Path) -> list[dict[str, str]]:
    route_dir = project_root / "src" / "api" / "routes"
    if not route_dir.exists():
        return []

    endpoints: list[dict[str, str]] = []
    pattern = re.compile(r"@router\.(get|post|put|patch|delete)\(\"([^\"]+)\"")
    for file_path in sorted(route_dir.glob("*.py")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for method, path in pattern.findall(text):
            endpoints.append(
                {
                    "method": method.upper(),
                    "path": f"/api/v1{path}",
                    "source_file": str(file_path),
                }
            )
    return endpoints


def _flatten_numeric_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric_metrics(value, sub))
    elif isinstance(obj, list):
        return out
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def _analyze_test_outputs(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(root),
        "available": root.exists(),
        "run_count": 0,
        "runs": [],
        "endpoint_summary": {},
        "latency_summary_ms": {},
        "scenario_summary": {},
        "scenario_request_examples": {},
        "coverage_notes": [],
    }

    if not root.exists():
        result["coverage_notes"].append("No workspace/test_outputs directory found.")
        return result

    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    result["run_count"] = len(run_dirs)
    if not run_dirs:
        result["coverage_notes"].append("No run folders found under workspace/test_outputs.")
        return result

    all_requests: list[dict[str, Any]] = []
    scenario_endpoint_counter: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run_dir in sorted(run_dirs):
        run_id = run_dir.name
        json_files = sorted(run_dir.rglob("*.json"))
        if not json_files:
            continue

        envelopes: list[dict[str, Any]] = []
        request_examples: list[dict[str, Any]] = []
        has_answer = False
        has_confirm = False
        has_abort = False
        poll_count = 0
        final_status = None

        for file_path in json_files:
            payload = _read_json(file_path)
            if payload is None:
                continue

            lower_name = file_path.name.lower()
            if lower_name.startswith("poll_status_"):
                poll_count += 1
            if "answer" in lower_name:
                has_answer = True
            if "confirm" in lower_name:
                has_confirm = True
            if "abort" in lower_name:
                has_abort = True

            if isinstance(payload, dict) and "final_status" in payload:
                final_status = payload.get("final_status")

            if isinstance(payload, dict) and "steps" in payload:
                for step in payload.get("steps", []):
                    if isinstance(step, dict) and step.get("request"):
                        resp = step.get("response") if isinstance(step.get("response"), dict) else {}
                        request_examples.append(
                            {
                                "step": step.get("name"),
                                "endpoint": _normalize_endpoint(str(resp.get("url", ""))),
                                "request": step.get("request"),
                                "status_code": resp.get("status_code"),
                            }
                        )

            envelopes.extend(_walk_request_envelopes(payload))

        scenario = "standard"
        if has_abort:
            scenario = "abort_flow"
        elif has_answer and has_confirm:
            scenario = "clarification_plus_confirmation"
        elif has_answer:
            scenario = "clarification_only"

        seen: set[tuple[str, str, str]] = set()
        run_requests: list[dict[str, Any]] = []
        for env in envelopes:
            url = str(env.get("url", ""))
            headers = env.get("headers") if isinstance(env.get("headers"), dict) else {}
            request_id = str(_header_ci(headers, "x-request-id") or "")
            status_code = int(env.get("status_code", 0)) if str(env.get("status_code", "")).isdigit() else 0
            dedup_key = (request_id, url, str(status_code))
            if request_id and dedup_key in seen:
                continue
            if request_id:
                seen.add(dedup_key)

            latency_sec = _as_float(_header_ci(headers, "x-process-time"))
            normalized = _normalize_endpoint(url)
            item = {
                "run_id": run_id,
                "scenario": scenario,
                "url": url,
                "endpoint": normalized,
                "status_code": status_code,
                "latency_ms": (latency_sec * 1000.0) if latency_sec is not None else None,
                "request_id": request_id or None,
            }
            run_requests.append(item)
            all_requests.append(item)
            scenario_endpoint_counter[scenario][normalized] += 1

        result["runs"].append(
            {
                "run_id": run_id,
                "path": str(run_dir),
                "scenario": scenario,
                "final_status": final_status,
                "poll_count": poll_count,
                "request_count": len(run_requests),
                "request_examples_count": len(request_examples),
            }
        )

        if request_examples:
            scenario_examples[scenario].extend(request_examples[:8])

    endpoint_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "ok": 0, "errors": 0, "latencies": []})
    global_latencies: list[float] = []

    for req in all_requests:
        bucket = endpoint_groups[req["endpoint"]]
        bucket["count"] += 1
        if 200 <= req["status_code"] < 300:
            bucket["ok"] += 1
        elif req["status_code"] >= 400:
            bucket["errors"] += 1
        if req["latency_ms"] is not None:
            bucket["latencies"].append(float(req["latency_ms"]))
            global_latencies.append(float(req["latency_ms"]))

    endpoint_summary = {}
    for endpoint, bucket in sorted(endpoint_groups.items(), key=lambda x: x[1]["count"], reverse=True):
        endpoint_summary[endpoint] = {
            "requests": bucket["count"],
            "ok": bucket["ok"],
            "errors": bucket["errors"],
            "latency_ms": _basic_stats(bucket["latencies"]),
        }

    result["endpoint_summary"] = endpoint_summary
    result["latency_summary_ms"] = _basic_stats(global_latencies)
    result["scenario_summary"] = {
        name: {
            "run_count": sum(1 for run in result["runs"] if run["scenario"] == name),
            "endpoint_requests": dict(counter),
        }
        for name, counter in scenario_endpoint_counter.items()
    }
    result["scenario_request_examples"] = {k: v[:8] for k, v in scenario_examples.items()}

    if not all_requests:
        result["coverage_notes"].append("No API request envelope JSON records found in run folders.")

    return result


def _analyze_db(db_path: Path, experiment_id: str | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "db_path": str(db_path),
        "available": db_path.exists(),
        "experiments": {},
        "agent_behavior": {},
        "rl_feedback": {},
        "metrics": {},
        "latency": {},
        "coverage_notes": [],
    }

    if not db_path.exists():
        output["coverage_notes"].append("SQLite state DB not found.")
        return output

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = ""
    params: list[Any] = []
    if experiment_id:
        where = "WHERE id = ?"
        params = [experiment_id]

    exp_rows = cur.execute(
        f"""
        SELECT id, status, phase, prompt, requires_quantum, framework, target_metric,
               created_at, updated_at, completed_at, state_json
        FROM experiments
        {where}
        ORDER BY created_at DESC
        """,
        params,
    ).fetchall()

    if not exp_rows:
        output["coverage_notes"].append("No experiments found in DB for selected filter.")

    exp_ids = [row["id"] for row in exp_rows]
    status_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    framework_counts: Counter[str] = Counter()
    duration_values: list[float] = []
    phase_latencies: dict[str, list[float]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    experiments_summary: list[dict[str, Any]] = []

    for row in exp_rows:
        normalized_status = _normalize_status(row["status"])
        status_counts[normalized_status] += 1
        phase_counts[row["phase"]] += 1
        if row["framework"]:
            framework_counts[row["framework"]] += 1

        state: dict[str, Any] = {}
        if row["state_json"]:
            try:
                state = json.loads(row["state_json"])
            except Exception:
                state = {}

        duration_sec = None
        ts_start = _as_float(state.get("timestamp_start")) if state else None
        ts_end = _as_float(state.get("timestamp_end")) if state else None
        total_duration = _as_float(state.get("total_duration_sec")) if state else None
        if total_duration is not None:
            duration_sec = total_duration
        elif ts_start is not None and ts_end is not None and ts_end >= ts_start:
            duration_sec = ts_end - ts_start
        if duration_sec is not None:
            duration_values.append(float(duration_sec))

        timings = state.get("phase_timings") if isinstance(state.get("phase_timings"), dict) else {}
        for phase, value in timings.items():
            val = _as_float(value)
            if val is not None:
                phase_latencies[str(phase)].append(val * 1000.0)

        metrics_obj = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
        flat_metrics = _flatten_numeric_metrics(metrics_obj)
        for name, val in flat_metrics.items():
            metric_values[name].append(float(val))

        experiments_summary.append(
            {
                "experiment_id": row["id"],
                "status": normalized_status,
                "phase": row["phase"],
                "framework": row["framework"],
                "requires_quantum": bool(row["requires_quantum"]),
                "target_metric": row["target_metric"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "duration_sec": duration_sec,
                "has_metrics": bool(metrics_obj),
                "phase_count": len(timings),
            }
        )

    output["experiments"] = {
        "count": len(exp_rows),
        "status_distribution": dict(status_counts),
        "phase_distribution": dict(phase_counts),
        "framework_distribution": dict(framework_counts),
        "duration_sec": _basic_stats(duration_values),
        "recent": experiments_summary[:20],
    }

    if exp_ids:
        placeholders = ",".join(["?"] * len(exp_ids))
        log_rows = cur.execute(
            f"""
            SELECT experiment_id, phase, level, message, details_json, timestamp
            FROM experiment_logs
            WHERE experiment_id IN ({placeholders})
            ORDER BY timestamp DESC
            """,
            exp_ids,
        ).fetchall()
    else:
        log_rows = []

    phase_level_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phase_message_counts: dict[str, Counter[str]] = defaultdict(Counter)
    completion_counts: Counter[str] = Counter()
    waiting_confirmation_events = 0

    for row in log_rows:
        phase = row["phase"] or "unknown"
        level = row["level"] or "unknown"
        message = row["message"] or ""
        phase_level_counts[phase][level] += 1
        phase_message_counts[phase][message] += 1
        if message.startswith("Phase ") and message.endswith(" completed"):
            completion_counts[phase] += 1
        if "Confirmation processed" in message:
            waiting_confirmation_events += 1

    phase_latency_summary = {phase: _basic_stats(vals) for phase, vals in phase_latencies.items()}
    output["latency"] = {
        "phase_latency_ms": phase_latency_summary,
        "phase_latency_top": dict(
            sorted(
                ((p, s.get("avg", 0.0)) for p, s in phase_latency_summary.items()),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ),
    }

    output["agent_behavior"] = {
        "log_entries": len(log_rows),
        "phase_level_distribution": {phase: dict(levels) for phase, levels in phase_level_counts.items()},
        "phase_completion_events": dict(completion_counts),
        "phase_top_messages": {
            phase: [{"message": msg, "count": cnt} for msg, cnt in counter.most_common(5)]
            for phase, counter in phase_message_counts.items()
        },
        "confirmation_events": waiting_confirmation_events,
    }

    if exp_ids:
        placeholders = ",".join(["?"] * len(exp_ids))
        metric_rows = cur.execute(
            f"""
            SELECT experiment_id, metric_name, metric_value, metric_type, recorded_at
            FROM experiment_metrics
            WHERE experiment_id IN ({placeholders})
            ORDER BY recorded_at DESC
            """,
            exp_ids,
        ).fetchall()
        rl_rows = cur.execute(
            f"""
            SELECT phase, signal, reward
            FROM rl_feedback
            WHERE experiment_id IN ({placeholders})
            """,
            exp_ids,
        ).fetchall()
    else:
        metric_rows = []
        rl_rows = []

    metrics_table_values: dict[str, list[float]] = defaultdict(list)
    metrics_table_types: dict[str, Counter[str]] = defaultdict(Counter)
    for row in metric_rows:
        name = row["metric_name"] or "unknown"
        val = _as_float(row["metric_value"])
        if val is not None:
            metrics_table_values[name].append(val)
        if row["metric_type"]:
            metrics_table_types[name][row["metric_type"]] += 1

    metric_state_summary = {
        name: _basic_stats(values) for name, values in sorted(metric_values.items(), key=lambda x: len(x[1]), reverse=True)
    }
    metric_table_summary = {
        name: {
            "stats": _basic_stats(values),
            "types": dict(metrics_table_types.get(name, {})),
        }
        for name, values in sorted(metrics_table_values.items(), key=lambda x: len(x[1]), reverse=True)
    }

    rl_values_by_phase_signal: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rl_rows:
        phase = row["phase"] or "unknown"
        signal = row["signal"] or "unknown"
        val = _as_float(row["reward"])
        if val is not None:
            rl_values_by_phase_signal[(phase, signal)].append(val)

    rl_summary = {
        f"{phase}:{signal}": {
            "count": len(values),
            "avg_reward": float(sum(values) / len(values)) if values else 0.0,
            "positive_rate": float(sum(1 for x in values if x > 0) / len(values)) if values else 0.0,
        }
        for (phase, signal), values in sorted(rl_values_by_phase_signal.items(), key=lambda x: len(x[1]), reverse=True)
    }

    output["metrics"] = {
        "from_state_json": metric_state_summary,
        "from_experiment_metrics_table": metric_table_summary,
        "records_in_metrics_table": len(metric_rows),
    }
    output["rl_feedback"] = {
        "records": len(rl_rows),
        "by_phase_signal": rl_summary,
    }

    if not metric_rows:
        output["coverage_notes"].append("experiment_metrics table is empty; using state_json.metrics only.")

    conn.close()
    return output


def _compose_summary(api_data: dict[str, Any], db_data: dict[str, Any], experiment_filter: str | None) -> dict[str, Any]:
    return {
        "generated_at_utc": _utc_now_iso(),
        "experiment_filter": experiment_filter,
        "overview": {
            "db_available": db_data.get("available", False),
            "api_trace_available": api_data.get("available", False),
            "api_run_count": api_data.get("run_count", 0),
            "configured_endpoint_count": len(api_data.get("configured_endpoints", [])),
            "db_experiment_count": db_data.get("experiments", {}).get("count", 0),
        },
        "api_analysis": api_data,
        "database_analysis": db_data,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    overview = summary.get("overview", {})
    api = summary.get("api_analysis", {})
    db = summary.get("database_analysis", {})

    lines: list[str] = []
    lines.append("# Research Platform Analysis Report")
    lines.append("")
    lines.append(f"Generated (UTC): {summary.get('generated_at_utc')}")
    if summary.get("experiment_filter"):
        lines.append(f"Experiment Filter: `{summary['experiment_filter']}`")
    lines.append("")

    lines.append("## 1) Overview")
    lines.append("")
    lines.append(f"- DB available: `{overview.get('db_available')}`")
    lines.append(f"- API traces available: `{overview.get('api_trace_available')}`")
    lines.append(f"- API run folders: `{overview.get('api_run_count')}`")
    lines.append(f"- Configured endpoints: `{overview.get('configured_endpoint_count', 0)}`")
    lines.append(f"- DB experiments analyzed: `{overview.get('db_experiment_count')}`")
    lines.append("")

    lines.append("## 2) API Endpoints and Latency")
    lines.append("")
    latency = api.get("latency_summary_ms", {})
    lines.append(
        "- Global latency (ms): "
        f"avg={latency.get('avg', 0):.2f}, p95={latency.get('p95', 0):.2f}, "
        f"min={latency.get('min', 0):.2f}, max={latency.get('max', 0):.2f}, count={int(latency.get('count', 0))}"
    )
    lines.append("")
    lines.append("| Endpoint | Requests | OK | Errors | Avg ms | P95 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    endpoint_items = list(api.get("endpoint_summary", {}).items())[:25]
    for endpoint, stats in endpoint_items:
        ls = stats.get("latency_ms", {})
        lines.append(
            f"| `{endpoint}` | {stats.get('requests', 0)} | {stats.get('ok', 0)} | {stats.get('errors', 0)} | "
            f"{ls.get('avg', 0):.2f} | {ls.get('p95', 0):.2f} |"
        )
    if not endpoint_items:
        lines.append("| _No API trace records found_ | 0 | 0 | 0 | 0 | 0 |")
    lines.append("")
    configured = api.get("configured_endpoints", [])
    if configured:
        lines.append("Configured endpoint catalog:")
        lines.append("| Method | Path |")
        lines.append("|---|---|")
        for item in configured:
            lines.append(f"| `{item.get('method')}` | `{item.get('path')}` |")
        lines.append("")

    lines.append("## 3) Agent Behavior")
    lines.append("")
    exp_stats = db.get("experiments", {})
    lines.append(f"- Status distribution: `{json.dumps(exp_stats.get('status_distribution', {}), ensure_ascii=True)}`")
    lines.append(f"- Phase distribution: `{json.dumps(exp_stats.get('phase_distribution', {}), ensure_ascii=True)}`")
    agent = db.get("agent_behavior", {})
    lines.append(f"- Log entries analyzed: `{agent.get('log_entries', 0)}`")
    lines.append(f"- Confirmation events: `{agent.get('confirmation_events', 0)}`")
    lines.append("")
    lines.append("Top phase completion events:")
    completions = agent.get("phase_completion_events", {})
    if completions:
        for phase, count in sorted(completions.items(), key=lambda x: x[1], reverse=True)[:15]:
            lines.append(f"- `{phase}`: {count}")
    else:
        lines.append("- No phase completion logs found.")
    lines.append("")

    lines.append("## 4) Scenario and Request Patterns")
    lines.append("")
    scenarios = api.get("scenario_summary", {})
    if scenarios:
        for scenario, info in scenarios.items():
            lines.append(f"- Scenario `{scenario}` runs: {info.get('run_count', 0)}")
            endpoints = info.get("endpoint_requests", {})
            for endpoint, count in sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:8]:
                lines.append(f"  - `{endpoint}`: {count}")
    else:
        lines.append("- No scenario-level API traces found.")
    lines.append("")

    examples = api.get("scenario_request_examples", {})
    if examples:
        lines.append("Example request payloads captured from transcript files:")
        for scenario, rows in examples.items():
            lines.append(f"- `{scenario}`")
            for item in rows[:3]:
                req_preview = json.dumps(item.get("request", {}), ensure_ascii=True)
                lines.append(
                    f"  - step={item.get('step')} endpoint={item.get('endpoint')} "
                    f"status={item.get('status_code')} request={req_preview[:220]}"
                )
    lines.append("")

    lines.append("## 5) Metrics and RL Signals")
    lines.append("")
    metrics = db.get("metrics", {})
    state_metrics = metrics.get("from_state_json", {})
    lines.append(f"- Metrics table records: `{metrics.get('records_in_metrics_table', 0)}`")
    lines.append("Top numeric metrics from state snapshots:")
    if state_metrics:
        for name, stats in list(sorted(state_metrics.items(), key=lambda x: x[1].get("count", 0), reverse=True))[:15]:
            lines.append(
                f"- `{name}`: avg={stats.get('avg', 0):.4f}, min={stats.get('min', 0):.4f}, "
                f"max={stats.get('max', 0):.4f}, samples={int(stats.get('count', 0))}"
            )
    else:
        lines.append("- No numeric metrics found in state snapshots.")

    rl = db.get("rl_feedback", {})
    lines.append("")
    lines.append(f"- RL feedback records: `{rl.get('records', 0)}`")
    by_phase_signal = rl.get("by_phase_signal", {})
    if by_phase_signal:
        lines.append("Top RL phase/signal averages:")
        for key, stats in list(sorted(by_phase_signal.items(), key=lambda x: x[1].get("count", 0), reverse=True))[:12]:
            lines.append(
                f"- `{key}`: avg_reward={stats.get('avg_reward', 0):.3f}, "
                f"positive_rate={stats.get('positive_rate', 0):.3f}, count={stats.get('count', 0)}"
            )

    lines.append("")
    lines.append("## 6) Coverage Notes")
    lines.append("")
    notes = []
    notes.extend(api.get("coverage_notes", []))
    notes.extend(db.get("coverage_notes", []))
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- None")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze platform runs: metrics, endpoints, agent behavior, scenarios, and latency.")
    parser.add_argument("--db-path", default="workspace/state.db", help="Path to SQLite state DB")
    parser.add_argument("--test-outputs-root", default="workspace/test_outputs", help="Root folder of JSON API run artifacts")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment ID filter")
    parser.add_argument("--output-json", default="workspace/analysis/analysis_summary.json", help="Path for output JSON summary")
    parser.add_argument("--output-md", default="workspace/analysis/analysis_report.md", help="Path for output markdown report")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    test_outputs_root = Path(args.test_outputs_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()

    api_data = _analyze_test_outputs(test_outputs_root)
    api_data["configured_endpoints"] = _catalog_api_endpoints(Path.cwd())
    db_data = _analyze_db(db_path, experiment_id=args.experiment_id)
    summary = _compose_summary(api_data, db_data, args.experiment_id)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    output_md.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"Analysis JSON: {output_json}")
    print(f"Analysis report: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
