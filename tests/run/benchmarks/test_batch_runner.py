import json
import os
import subprocess
import sys
from pathlib import Path

from minisweagent.run.benchmarks.batch_runner import BatchConfig, build_command, run_batch, validate_ids

# A real child process producing fixture artifacts. It never imports, patches, or mocks the upstream agent.
CHILD = '''
import json, sys
from pathlib import Path
raw, iid, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if mode == "crash":
    print("fixture process failed")
    sys.exit(7)
status = "LimitsExceeded" if mode == "limited" else "Submitted"
identity = "wrong__id" if mode == "wrong_id" else iid
prediction = {"instance_id": identity, "model_name_or_path": "fixture/model", "model_patch": ""}
if mode != "no_prediction":
    (raw / "preds.json").write_text(json.dumps({iid: prediction}))
trajectory = {"trajectory_format": "mini-swe-agent-1.1", "instance_id": identity,
              "info": {"exit_status": status, "submission": "", "model_stats": {"api_calls": 2, "instance_cost": .1}},
              "messages": [{"role": "assistant", "extra": {"actions": []}}]}
(raw / iid).mkdir()
(raw / iid / (iid + ".traj.json")).write_text("bad json" if mode == "bad_trajectory" else json.dumps(trajectory))
(raw / "exit_statuses_1.yaml").write_text("instances_by_exit_status:\\n  " + status + ":\\n    - " + iid + "\\n")
'''


def preparer(modes, order):
    def prepare(iid, raw):
        order.append(iid)
        return [sys.executable, "-c", CHILD, str(raw), iid, modes.get(iid, "ok")]
    return prepare


def test_sequential_failure_isolation_and_metrics(tmp_path):
    order = []
    output = tmp_path / "batch"
    ids = ["org__one", "org__two", "org__three", "org__one"]
    assert run_batch(ids, output, preparer({"org__two": "crash"}, order)) == 1
    report = json.loads((output / "summary.json").read_text())
    assert order == ids[:3] and report["case_count"] == 3
    assert [c["status"] for c in report["cases"]] == ["submitted", "failed", "submitted"]
    assert report["metrics"]["api_calls"] == {"known_sum": 4, "unknown_cases": 1}
    assert abs(report["metrics"]["cost"]["known_sum"] - .2) < 1e-12
    assert report["missing_prediction_ids"] == ["org__two"]
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2
    assert report["cases"][1]["runtime_seconds"] >= 0 and report["cases"][1]["returncode"] == 7
    assert report["cases"][1]["artifacts"]["trajectory"] is None
    assert (output / "cases/org__one/patch.diff").read_text() == ""


def test_exit_zero_is_not_success_and_artifact_errors(tmp_path):
    modes = {"org__limit": "limited", "org__missing": "no_prediction", "org__wrong": "wrong_id",
             "org__bad": "bad_trajectory", "org__ok": "ok"}
    assert run_batch(list(modes), tmp_path / "batch", preparer(modes, [])) == 1
    report = json.loads((tmp_path / "batch/summary.json").read_text())
    assert [c["status"] for c in report["cases"]] == ["failed"] * 4 + ["submitted"]
    assert all(c["returncode"] == 0 for c in report["cases"])
    assert report["cases"][0]["exit_status"] == "LimitsExceeded"
    assert report["cases"][2]["parse_status"] == "error"
    assert report["cases"][3]["parse_status"] == "error"
    preds = [json.loads(line) for line in (tmp_path / "batch/predictions.jsonl").read_text().splitlines()]
    assert {p["instance_id"] for p in preds} == {"org__limit", "org__bad", "org__ok"}


def test_preflight_failure_and_interrupt(tmp_path):
    def prepare(iid, raw):
        if iid == "org__missing":
            raise ValueError("ImageUnavailable")
        if iid == "org__interrupt":
            raise KeyboardInterrupt
        return [sys.executable, "-c", CHILD, str(raw), iid, "ok"]

    assert run_batch(["org__missing", "org__ok", "org__interrupt", "org__later"],
                     tmp_path / "batch", prepare) == 130
    report = json.loads((tmp_path / "batch/summary.json").read_text())
    assert [c["status"] for c in report["cases"]] == ["failed", "submitted", "interrupted", "not_run"]
    assert report["cases"][0]["runtime_seconds"] is None


def test_input_safety_and_command(tmp_path):
    for ids in ([], ["../bad"], ["org__repo/x"], ["org__repo;touch x"]):
        try:
            validate_ids(ids)
        except ValueError:
            pass
        else:
            raise AssertionError(ids)
    command = build_command(BatchConfig("/data/local", config_specs=("agent.cost_limit=0",)),
                            "org__repo.name-1", tmp_path / "raw", ["--rm", "--pull=never"])
    assert command[:3] == [sys.executable, "-m", "minisweagent.run.benchmarks.swebench"]
    assert command[command.index("--filter") + 1] == r"^org__repo\.name\-1$"
    assert "--pull=never" in command[-1] and "swebench.yaml" in command
    assert command[command.index("--workers") + 1] == "1"
    (tmp_path / "keep.txt").write_text("original")
    try:
        run_batch(["org__one"], tmp_path, preparer({}, []))
    except ValueError:
        pass
    else:
        raise AssertionError("Nonempty output accepted")
    assert (tmp_path / "keep.txt").read_text() == "original"


def test_cli_help_and_invalid_input(tmp_path):
    for module in ("batch_runner", "trajectory_parser"):
        result = subprocess.run([sys.executable, "-m", "minisweagent.run.benchmarks." + module, "--help"],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--output" in result.stdout
    invalid = subprocess.run([sys.executable, "-m", "minisweagent.run.benchmarks.batch_runner",
                              "--instance-id", "../bad", "--subset", "/missing", "--output", str(tmp_path / "unused")],
                             capture_output=True, text=True)
    assert invalid.returncode == 2 and not (tmp_path / "unused").exists()


def test_running_child_interrupt(tmp_path):
    if os.name != "posix":
        import pytest

        pytest.skip("Server process signal test is Linux-only")
    from minisweagent.run.benchmarks.batch_runner import _execute

    # The child signals its own parent once, then exits on the forwarded SIGINT.
    code = "import os,signal,time; time.sleep(.3); os.kill(os.getppid(),signal.SIGINT); time.sleep(10)"
    returncode, interrupted = _execute([sys.executable, "-c", code], tmp_path / "child.log", dict(os.environ))
    assert interrupted and returncode != 0


def test_all_submitted_and_prediction_parser_cli(tmp_path):
    output = tmp_path / "success"
    assert run_batch(["org__one", "org__two", "org__three"], output, preparer({}, [])) == 0
    parsed_path = tmp_path / "parsed.json"
    trajectory = output / "cases/org__one/raw/org__one/org__one.traj.json"
    command = [sys.executable, "-m", "minisweagent.run.benchmarks.trajectory_parser",
               "--trajectory", str(trajectory), "--output", str(parsed_path)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    before = parsed_path.read_bytes()
    result = json.loads(before)
    assert result["runtime_seconds"] is None and result["status"] == "submitted"
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert parsed_path.read_bytes() == before
