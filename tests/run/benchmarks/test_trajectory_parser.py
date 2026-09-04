import copy
import json
import os
from pathlib import Path

from minisweagent.run.benchmarks.trajectory_parser import TrajectoryError, parse_file, parse_trajectory, patch_files


def document(messages=None, status="Submitted", patch=""):
    return {"trajectory_format": "mini-swe-agent-1.1", "instance_id": "org__repo-1",
            "info": {"exit_status": status, "submission": patch,
                     "model_stats": {"api_calls": 3, "instance_cost": 0.02}}, "messages": messages or []}


def pair(command, code=0, output="", call_id="a"):
    return [{"role": "assistant", "extra": {"actions": [{"tool_call_id": call_id, "command": command}],
                                                "timestamp": 10}},
            {"role": "tool", "tool_call_id": call_id,
             "extra": {"returncode": code, "raw_output": output, "timestamp": 12}}]


def raises_trajectory_error(function):
    try:
        function()
    except TrajectoryError:
        return
    raise AssertionError("Expected TrajectoryError")


def test_multi_call_and_terminal_submission():
    messages = pair("pytest -q", 1, "1 failed")
    messages[0]["extra"]["actions"].append({"tool_call_id": "b", "command": "pwd"})
    messages[0]["extra"]["response"] = {"choices": [{"message": copy.deepcopy(messages[0])}]}
    messages.insert(2, {"role": "tool", "tool_call_id": "b", "extra": {"returncode": 0}})
    messages.extend([{"role": "assistant", "extra": {"actions": [
        {"tool_call_id": "c", "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}]}},
        {"role": "exit", "extra": {"exit_status": "Submitted", "submission": ""}}])
    result = parse_trajectory(document(messages))
    assert result["total_steps"] == 2 and result["tool_call_count"] == 3
    assert result["tool_calls"][-1]["execution"] == "submission_exit"
    assert result["test_failures"] == 1 and result["api_calls"] == 3
    assert result["runtime_seconds"] is None and result["observed_span_seconds"] == 2


def test_test_command_classification():
    examples = [
        ("python -m pytest -q", 0, "", "passed"),
        ("pytest -q", 1, "1 failed", "failed"),
        ("pytest -q", 2, "ERROR collecting module", "failed"),
        ("pytest -q", 2, "KeyboardInterrupt", "unknown"),
        ("pytest -q", 5, "no tests ran", "no_tests"),
        ("false && pytest -q", 1, "", "unknown"),
        ("pytest -q | tee out.txt", 0, "1 failed", "failed"),
        ("python -m unittest", 1, "FAILED (failures=1)", "failed"),
        ("python manage.py test", 1, "FAILED (errors=1)", "failed"),
        ("tox", 1, "installation failed", "unknown"),
        ("grep pytest test.py", 1, "failed", None),
        ("cat tests/test_example.py", 0, "1 failed", None),
        ("echo 'pytest -q'", 0, "", None),
        ("python -c 'print(\"pytest failed\")'", 0, "", None),
    ]
    for command, code, output, expected in examples:
        tests = parse_trajectory(document(pair(command, code, output)))["test_runs"]
        assert (tests[0]["status"] if tests else None) == expected, command


def test_parser_exposes_verification_and_guard_termination():
    messages = pair("pytest tests/test_a.py -q | head -20", 0, "1 failed")
    data = document(messages, status="GuardUnresolvedEvidence")
    data["info"]["visible_counterexample_guard"] = {
        "guard_unresolved_evidence": [{"evidence_id": "evidence-1"}],
        "guard_termination_reason": "max_total_submit_blocks_reached",
    }
    result = parse_trajectory(data)
    assert result["test_runs"][0]["verification_target"] == "pytest::tests/test_a.py"
    assert result["test_runs"][0]["closure_requires_reliable_rerun"] is True
    assert result["guard_unresolved_evidence"] == [{"evidence_id": "evidence-1"}]
    assert result["guard_termination_reason"] == "max_total_submit_blocks_reached"


def test_missing_fields_fallback_and_errors(tmp_path):
    message = {"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]}
    data = {"trajectory_format": "mini-swe-agent-1.1", "messages": [message]}
    result = parse_trajectory(data, instance_id="org__repo-1")
    assert result["status"] == "unknown" and result["api_calls"] is None and result["cost"] is None
    assert result["modified_files"] is None and result["observed_span_seconds"] is None
    assert result["tool_call_count"] == 1 and result["tool_calls"][0]["execution"] == "unconfirmed"
    raises_trajectory_error(lambda: parse_trajectory(document(), instance_id="wrong__id"))
    raises_trajectory_error(lambda: parse_trajectory({"messages": []}))
    raises_trajectory_error(lambda: parse_trajectory(document(), runtime_seconds=float("nan")))
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    raises_trajectory_error(lambda: parse_file(tmp_path / "bad.json"))
    data = document(pair("pytest", 0))
    data["messages"][1]["extra"]["exception_info"] = "action was not executed"
    assert parse_trajectory(data)["test_runs"][0]["status"] == "unknown"


def test_patches_and_reused_ids():
    patch = ('diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n'
             'diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n'
             'diff --git a/before.py b/after.py\nsimilarity index 100%\nrename from before.py\nrename to after.py\n')
    assert patch_files(patch) == ["after.py", "before.py", "new.py", "old.py"]
    assert parse_trajectory(document(patch=""))["modified_files"] == []
    messages = pair("pytest", 1) + pair("pytest", 0)
    result = parse_trajectory(document(messages))
    assert [c["returncode"] for c in result["tool_calls"]] == [1, 0]
    assert result["test_failures"] == 1
    # A command's repeated key in the provider response is never counted a second time.
    messages[0]["tool_calls"] = [{"id": "unused"}]
    assert parse_trajectory(document(messages))["tool_call_count"] == 2


def test_inconsistent_status_and_invalid_values():
    data = document([{"role": "exit", "extra": {"exit_status": "LimitsExceeded"}}])
    raises_trajectory_error(lambda: parse_trajectory(data))
    data = document(pair("pytest"))
    data["info"]["model_stats"] = {"api_calls": True, "instance_cost": float("inf")}
    data["messages"][1]["extra"]["timestamp"] = "not a time"
    result = parse_trajectory(data)
    assert result["api_calls"] is None and result["cost"] is None and result["observed_span_seconds"] is None
    assert len(result["warnings"]) == 2
    data["messages"][0]["extra"]["actions"][0]["command"] = {"bad": "command"}
    raises_trajectory_error(lambda: parse_trajectory(data))


def test_real_trajectory():
    source = os.environ.get("MINI_BATCH_TEST_TRAJECTORY")
    if not source:
        import pytest

        pytest.skip("Set MINI_BATCH_TEST_TRAJECTORY to the previously completed sqlfluff trajectory")
    original = json.loads(Path(source).read_text(encoding="utf-8"))
    # Keep only parser inputs, excluding credentials, prompts, reasoning, and provider metadata.
    sanitized = {key: original[key] for key in ("instance_id", "trajectory_format")}
    sanitized["info"] = {key: original["info"][key] for key in ("model_stats", "exit_status", "submission")}
    sanitized["messages"] = []
    for message in original["messages"]:
        clean = {key: message[key] for key in ("role", "tool_call_id") if key in message}
        clean["extra"] = {key: value for key, value in message.get("extra", {}).items()
                          if key in {"actions", "returncode", "timestamp", "exception_info", "exit_status"}}
        sanitized["messages"].append(clean)
    result = parse_trajectory(sanitized)
    assert result["instance_id"] == "sqlfluff__sqlfluff-1625" and result["status"] == "submitted"
    assert result["total_steps"] == 39 and result["api_calls"] == 39
    assert abs(result["cost"] - 0.026041275) < 1e-12
    assert result["modified_files"] == ["src/sqlfluff/rules/L031.py"]
    assert result["tool_call_count"] == 39 and result["tool_calls"][-1]["execution"] == "submission_exit"
