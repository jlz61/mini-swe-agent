"""Pure, non-executing parser for mini-swe-agent 1.1 trajectories."""

import json
import math
import re
import shlex
from pathlib import Path

import typer

from minisweagent.run.benchmarks.test_status import classify_execution

app = typer.Typer(add_completion=False)


class TrajectoryError(ValueError):
    """Invalid, inconsistent, or unsupported trajectory."""


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _path_text(value: str) -> str:
    value = value.split("\t", 1)[0].strip()
    if value.startswith('"'):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    return value


def patch_files(patch: str) -> list[str]:
    """Extract final changed paths, including both sides of a rename. Never open them."""
    files = set()
    for block in re.split(r"(?m)^diff --git ", patch)[1:]:
        lines = block.splitlines()
        paths = []
        for line in lines[1:]:
            if line.startswith("@@"):
                break
            if line.startswith(("--- ", "+++ ")):
                value = _path_text(line[4:])
                if value != "/dev/null":
                    paths.append(value[2:] if value.startswith(("a/", "b/")) else value)
            elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
                paths.append(_path_text(line.split(" ", 2)[2]))
        if not paths and lines:
            try:
                parts = shlex.split(lines[0])
                if len(parts) == 2:
                    paths = [p[2:] if p.startswith(("a/", "b/")) else p for p in parts]
            except ValueError:
                pass
        files.update(paths)
    return sorted(files)


def _test_result(call: dict, raw_output: str) -> dict | None:
    result = classify_execution(call.get("command") or "", raw_output, call["returncode"],
                                call["exception_info"] or "", observed=call["execution"] == "observed")
    if result["kind"] != "test":
        return None
    return {
        "message_index": call["message_index"], "tool_call_id": call["tool_call_id"],
        "command": call["command"], "frameworks": result["frameworks"], "status": result["status"],
        "returncode": call["returncode"], "evidence": result["reason"],
        "excerpt": result["excerpt"], "observation_index": call["observation_index"],
        "verification_target": result.get("verification_target"),
        "closure_requires_reliable_rerun": result.get("closure_requires_reliable_rerun", False),
        "reliable_verification": result.get("reliable_verification", False),
    }


def _actions(message: dict, index: int, warnings: list[str]) -> tuple[list, str]:
    actions = _object(message.get("extra")).get("actions")
    if actions is not None:
        if not isinstance(actions, list):
            raise TrajectoryError("extra.actions must be a list")
        return actions, "extra.actions"
    actions = []
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise TrajectoryError("tool_calls must be a list")
    for tool in tool_calls:
        tool = _object(tool)
        function = _object(tool.get("function"))
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
            warnings.append(f"invalid_tool_arguments_at_{index}")
        actions.append({"command": _object(arguments).get("command"),
                        "tool_call_id": tool.get("id"), "name": function.get("name")})
    return actions, "tool_calls"


def parse_trajectory(data: dict, *, instance_id: str | None = None,
                     runtime_seconds: float | None = None, patch: str | None = None) -> dict:
    """Parse in-memory data; absent measurements stay null, not zero."""
    if not isinstance(data, dict) or data.get("trajectory_format") != "mini-swe-agent-1.1":
        raise TrajectoryError("Expected trajectory_format mini-swe-agent-1.1")
    messages = data.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise TrajectoryError("messages must be a list of objects")
    actual_id = data.get("instance_id")
    if actual_id is not None and (not isinstance(actual_id, str) or not actual_id):
        raise TrajectoryError("Invalid instance_id")
    if instance_id and actual_id and instance_id != actual_id:
        raise TrajectoryError("Trajectory instance_id does not match requested instance")
    if runtime_seconds is not None and not _number(runtime_seconds):
        raise TrajectoryError("runtime_seconds must be a finite nonnegative number")
    warnings = [] if actual_id else ["trajectory_instance_id_missing"]
    info = _object(data.get("info"))
    stats = _object(info.get("model_stats"))
    api_calls, cost = stats.get("api_calls"), stats.get("instance_cost")
    if not isinstance(api_calls, int) or isinstance(api_calls, bool) or api_calls < 0:
        api_calls = None
        warnings.append("api_calls_missing_or_invalid")
    if not _number(cost):
        cost = None
        warnings.append("cost_missing_or_invalid")
    terminal = next((m for m in reversed(messages) if m.get("role") == "exit"), {})
    terminal_extra = _object(terminal.get("extra"))
    exit_status = info.get("exit_status") or terminal_extra.get("exit_status") or None
    if exit_status is not None and not isinstance(exit_status, str):
        raise TrajectoryError("exit_status must be a string")
    if info.get("exit_status") and terminal_extra.get("exit_status") not in (None, info["exit_status"]):
        raise TrajectoryError("Conflicting info and terminal exit_status")
    status = "submitted" if exit_status == "Submitted" else "failed" if exit_status else "unknown"
    timestamps = [_object(m.get("extra")).get("timestamp") for m in messages]
    timestamps = [t for t in timestamps if _number(t)]
    observations: dict[str, list[tuple[int, dict]]] = {}
    for i, message in enumerate(messages):
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            observations.setdefault(message["tool_call_id"], []).append((i, message))
    calls, tests = [], []
    used_observations = set()
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    for step, i in enumerate(assistant_indices):
        actions, source = _actions(messages[i], i, warnings)
        boundary = assistant_indices[step + 1] if step + 1 < len(assistant_indices) else len(messages)
        for action in actions:
            if not isinstance(action, dict):
                raise TrajectoryError("Action must be an object")
            call_id = action.get("tool_call_id")
            if call_id is not None and not isinstance(call_id, str):
                raise TrajectoryError("tool_call_id must be a string")
            candidates = [(j, m) for j, m in observations.get(call_id, [])
                          if i < j < boundary and j not in used_observations]
            observation_index, observation = candidates[0] if candidates else (None, {})
            if observation_index is not None:
                used_observations.add(observation_index)
            output = _object(observation.get("extra"))
            command = action.get("command")
            if command is not None and not isinstance(command, str):
                raise TrajectoryError("Action command must be a string")
            execution = "observed" if observation else "unconfirmed"
            if output.get("exception_info") == "action was not executed":
                execution = "not_executed"
            if not observation and status == "submitted" and i == assistant_indices[-1] and len(actions) == 1:
                if command and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
                    execution = "submission_exit"
            code = output.get("returncode")
            if not isinstance(code, int) or isinstance(code, bool):
                code = None
            call = {
                "message_index": i, "tool_call_id": call_id, "name": action.get("name", "bash"),
                "command": command, "source": source, "observation_index": observation_index,
                "returncode": code, "exception_info": output.get("exception_info") or None, "execution": execution,
            }
            calls.append(call)
            if call["name"] == "bash":
                test = _test_result(call, str(output.get("raw_output") or observation.get("content") or ""))
                if test:
                    tests.append(test)
    patch_source = "prediction" if patch is not None else "info.submission_or_terminal"
    if patch is None:
        patch = info.get("submission", terminal_extra.get("submission"))
    if patch is not None and not isinstance(patch, str):
        raise TrajectoryError("Submission patch must be a string or null")
    files = patch_files(patch) if patch is not None else None
    if patch and not files:
        warnings.append("nonempty_patch_has_no_recognized_git_diff_paths")
    guard = _object(info.get("visible_counterexample_guard"))
    return {
        "schema_version": 1, "trajectory_format": data["trajectory_format"], "instance_id": actual_id or instance_id,
        "status": status, "exit_status": exit_status, "total_steps": len(assistant_indices),
        "api_calls": api_calls, "cost": cost, "cost_currency": "USD", "runtime_seconds": runtime_seconds,
        "observed_span_seconds": max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None,
        "tool_calls": calls, "tool_call_count": len(calls),
        "bash_calls": [c for c in calls if c["name"] == "bash"],
        "bash_call_count": sum(c["name"] == "bash" for c in calls),
        "test_runs": tests, "test_run_count": len(tests),
        "test_failures": sum(t["status"] == "failed" for t in tests),
        "test_unknown_count": sum(t["status"] == "unknown" for t in tests),
        "test_timeout_count": sum(t["status"] == "timeout" for t in tests),
        "test_no_tests_count": sum(t["status"] == "no_tests" for t in tests),
        "visible_counterexample_guard": info.get("visible_counterexample_guard"),
        "guard_unresolved_evidence": guard.get("guard_unresolved_evidence"),
        "guard_termination_reason": guard.get("guard_termination_reason"),
        "modified_files": files, "warnings": warnings,
        "sources": {
            "total_steps": "assistant_message_count", "api_calls": "info.model_stats.api_calls",
            "cost": "info.model_stats.instance_cost (upstream USD estimate)",
            "runtime_seconds": "runner_monotonic_clock" if runtime_seconds is not None else None,
            "observed_span_seconds": "message_timestamp_span_not_full_runtime",
            "tests": "heuristic_supported_python_commands_not_official_harness",
            "modified_files": f"{patch_source}: final_net_diff_only",
        },
    }


def parse_file(path: Path, **kwargs) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrajectoryError(f"Cannot read trajectory: {type(exc).__name__}") from exc
    return parse_trajectory(data, **kwargs)


@app.command()
def main(trajectory: Path = typer.Option(..., "--trajectory"),
         output: Path = typer.Option(..., "--output")) -> None:
    if output.exists():
        raise typer.BadParameter("Output already exists; choose a new file")
    try:
        result = parse_file(trajectory)
    except TrajectoryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    app()
