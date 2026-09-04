import json
import sys

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.benchmarks.trajectory_parser import parse_trajectory
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.visible_counterexample_guard import (
    EvidenceLedger, GuardedProgressTrackingAgent, VisibleCounterexampleAgent, benchmark_agent_class, create_benchmark_agent,
)

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo fixture-patch"
FAIL = "Traceback (most recent call last):\nAssertionError: visible counterexample"


def model_turn(*commands):
    actions = [{"command": command, "tool_call_id": f"call-{i}"} for i, command in enumerate(commands)]
    tools = [{"id": action["tool_call_id"], "type": "function", "function": {
        "name": "bash", "arguments": json.dumps({"command": action["command"]})}} for action in actions]
    return make_toolcall_output("Deterministic offline fixture", tools, actions)


def make_agent(tmp_path, turns, **kwargs):
    return VisibleCounterexampleAgent(
        DeterministicToolcallModel(outputs=turns, cost_per_call=0),
        LocalEnvironment(cwd=str(tmp_path), timeout=2),
        system_template="Offline fixture", instance_template="{{task}}", cost_limit=0,
        output_path=tmp_path / "dry-run.traj.json", **kwargs,
    )


def test_ledger_failed_retest_unknown_and_recurrence():
    ledger = EvidenceLedger()
    ledger.observe("pytest tests/test_a.py", "1 failed", 1, "", 1)
    ledger.observe("pytest tests/test_a.py", "1 failed", 1, "", 2)
    assert len(ledger.unresolved) == 1 and len(ledger.unresolved[0]["observations"]) == 2
    assert ledger.block_submission(3)["recovery_trigger"] is True
    assert ledger.block_submission(4)["recovery_trigger"] is False
    ledger.observe("pytest tests/test_b.py", "1 passed", 0, "", 5)
    ledger.observe("pytest tests/test_a.py", "no tests ran", 5, "", 6)
    ledger.observe("pytest tests/test_a.py", "", -1, "timed out", 7)
    assert len(ledger.unresolved) == 1
    ledger.observe("pytest tests/test_a.py", "1 passed", 0, "", 8)
    assert ledger.block_submission(9) is None
    ledger.observe("pytest tests/test_a.py", "1 failed", 1, "", 10)
    assert ledger.unresolved[0]["evidence_id"] == "evidence-2"
    assert ledger.block_submission(11)["recovery_trigger"] is True


def test_review_requires_host_citations_and_corroboration():
    ledger = EvidenceLedger()
    ledger.observe("python reproduce.py", FAIL, 1, "", 1)
    ledger.observe("python inspect_dependency.py", "independent dependency check: missing package", 0, "", 2)
    base = {"evidence_id": "evidence-1", "resolution": "environment",
            "reviewer": "host:operator", "reason": "Independent inspection confirms missing dependency before application execution.",
            "citations": [{"observation_id": "obs-1", "quote": "AssertionError: visible counterexample"},
                          {"observation_id": "obs-2", "quote": "independent dependency check: missing package"}]}
    for change in [{"reviewer": "agent"}, {"reason": "unrelated"}, {"citations": base["citations"][:1]},
                   {"citations": [{"observation_id": "obs-2", "quote": "fabricated evidence"}]}]:
        with pytest.raises(ValueError):
            ledger.review(base | change)
        assert ledger.unresolved
    ledger.review(base)
    assert not ledger.unresolved and ledger.evidence[0]["resolution"]["resolution"] == "environment"


def test_dry_run_failed_then_retest_then_submit(tmp_path):
    (tmp_path / "test_counterexample.py").write_text(
        "from pathlib import Path\ndef test_counterexample():\n    assert Path('fixed').exists()\n")
    command = f"{sys.executable} -m pytest -q test_counterexample.py"
    agent = make_agent(tmp_path, [model_turn(command, SUBMIT), model_turn("touch fixed", command), model_turn(SUBMIT)])
    assert agent.run("Synthetic counterexample")["exit_status"] == "Submitted"
    state = agent.serialize()["info"]["visible_counterexample_guard"]
    assert len(state["recovery_events"]) == 1 and state["recovery_events"][0]["trigger_step"] == 1
    assert not state["unresolved_evidence"]
    assert state["evidence"][0]["resolution"]["kind"] == "passed_retest"
    assert "Summarize current state" in "\n".join(m.get("content", "") for m in agent.messages)
    assert parse_trajectory(agent.serialize())["visible_counterexample_guard"]["recovery_events"]


def test_dry_run_repeated_submit_and_computed_marker_blocked(tmp_path):
    repro = f"{sys.executable} -c 'raise AssertionError(\"visible counterexample\")'"
    computed = "printf 'COMPLETE_TASK_AND_SUBMIT_%s\\npatch\\n' FINAL_OUTPUT"
    agent = make_agent(tmp_path, [model_turn(repro, SUBMIT), model_turn(computed)], step_limit=2)
    assert agent.run()["exit_status"] == "LimitsExceeded"
    events = agent.ledger.events
    assert [event["recovery_trigger"] for event in events] == [True, False]
    assert [event["trigger_step"] for event in events] == [1, 2]
    assert all(m.get("extra", {}).get("exit_status") != "Submitted" for m in agent.messages)


def test_dry_run_pipeline_zero_never_closes_failure(tmp_path):
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert False\n")
    command = f"{sys.executable} -m pytest -q test_failure.py | head -40"
    agent = make_agent(tmp_path, [model_turn(command, SUBMIT)], step_limit=1)
    assert agent.run()["exit_status"] == "LimitsExceeded"
    assert agent.ledger.observations[0]["returncode"] == 0
    assert agent.ledger.observations[0]["status"] == "failed"


def test_pipeline_failure_closes_with_reliable_same_target_only(tmp_path):
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert True\n")
    piped = f"{sys.executable} -m pytest -q test_failure.py | head -40"
    direct = f"{sys.executable} -m pytest test_failure.py -q"
    ledger = EvidenceLedger()
    ledger.observe(piped, "1 failed", 0, "", 1)
    assert ledger.unresolved[0]["closure_requires_reliable_rerun"] is True
    ledger.observe(f"{sys.executable} -m pytest other.py -q", "1 passed", 0, "", 2)
    assert ledger.unresolved
    ledger.observe(direct, "1 passed", 0, "", 3)
    assert not ledger.unresolved
    assert ledger.evidence[0]["resolution"]["closure_method"] == "reliable_pytest_target_passed"


def test_dry_run_pipeline_failure_reliable_rerun_then_submit(tmp_path):
    (tmp_path / "test_failure.py").write_text(
        "from pathlib import Path\ndef test_failure():\n    assert Path('fixed').exists()\n")
    piped = f"{sys.executable} -m pytest -q test_failure.py | head -40"
    direct = f"{sys.executable} -m pytest test_failure.py -q"
    agent = make_agent(tmp_path, [model_turn(piped, SUBMIT), model_turn("touch fixed", direct), model_turn(SUBMIT)])
    assert agent.run()["exit_status"] == "Submitted"
    assert agent.ledger.evidence[0]["resolution"]["kind"] == "passed_target_retest"
    assert agent.ledger.evidence[0]["verification_target"] == "pytest::test_failure.py"


def test_guard_terminates_after_bounded_submit_blocks(tmp_path):
    repro = f"{sys.executable} -c 'raise AssertionError()'"
    agent = make_agent(tmp_path, [model_turn(repro, SUBMIT), model_turn(SUBMIT), model_turn(SUBMIT)],
                       guard_max_submit_blocks=3)
    assert agent.run()["exit_status"] == "GuardUnresolvedEvidence"
    state = agent.serialize()["info"]["visible_counterexample_guard"]
    assert len(state["recovery_events"]) == 3 and state["guard_unresolved_evidence"]
    assert state["guard_termination_reason"] == "max_evidence_group_submit_blocks_reached"


def test_guard_terminates_after_bounded_recovery_steps(tmp_path):
    repro = f"{sys.executable} -c 'raise AssertionError()'"
    agent = make_agent(tmp_path, [model_turn(repro, SUBMIT), model_turn("pwd"), model_turn("pwd")],
                       guard_max_submit_blocks=10, guard_max_recovery_steps=2)
    assert agent.run()["exit_status"] == "GuardUnresolvedEvidence"
    state = agent.serialize()["info"]["visible_counterexample_guard"]
    assert state["guard_termination_reason"] == "max_recovery_steps_reached"
    assert state["termination"]["termination_step"] == 3


def test_dry_run_only_visible_observation_is_used(tmp_path):
    agent = make_agent(tmp_path, [model_turn(f"{sys.executable} -c 'raise AssertionError()'"), model_turn(SUBMIT)])
    # A real configured observation template, not a patched formatter or mocked upstream.
    agent.model.config.observation_template = "output omitted by fixture template"
    assert agent.run()["exit_status"] == "Submitted"
    assert not agent.ledger.evidence


def test_hidden_exit_code_does_not_create_evidence(tmp_path):
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert False\n")
    agent = make_agent(tmp_path, [model_turn(f"{sys.executable} -m pytest -q test_failure.py"), model_turn(SUBMIT)])
    agent.model.config.observation_template = "output and returncode omitted"
    assert agent.run()["exit_status"] == "Submitted"
    assert not agent.ledger.evidence and agent.ledger.observations[0]["returncode"] is None


def test_invalid_host_review_file_cannot_unblock(tmp_path):
    review = tmp_path / "host-review.json"
    review.write_text('{"instance_id": "wrong-case", "decisions": []}')
    agent = make_agent(tmp_path, [model_turn(f"{sys.executable} -c 'raise AssertionError()'", SUBMIT)],
                       guard_review_file=str(review), step_limit=1)
    assert agent.run()["exit_status"] == "LimitsExceeded"
    assert agent.review_errors and agent.ledger.unresolved


def test_baseline_factory_and_unrelated_review():
    assert benchmark_agent_class({}) is ProgressTrackingAgent
    assert benchmark_agent_class({"run": {"visible_counterexample_guard": False}}) is ProgressTrackingAgent
    assert benchmark_agent_class({"run": {"visible_counterexample_guard": True}}) is GuardedProgressTrackingAgent
    with pytest.raises(ValueError):
        benchmark_agent_class({"run": {"visible_counterexample_guard": "false"}})
    ledger = EvidenceLedger()
    ledger.observe("python unrelated.py", FAIL, 1, "", 1)
    ledger.observe("cat scope.txt", "This optional integration is explicitly outside the requested task scope.", 0, "", 2)
    ledger.review({"evidence_id": "evidence-1", "resolution": "unrelated", "reviewer": "host:operator",
                   "reason": "Confirmed against the explicit task scope and independent integration inspection.",
                   "citations": [{"observation_id": "obs-1", "quote": "AssertionError: visible counterexample"},
                                 {"observation_id": "obs-2", "quote": "explicitly outside the requested task scope"}]})
    assert not ledger.unresolved


@pytest.mark.parametrize(("enabled", "expected"), [(False, "Submitted"), (True, "LimitsExceeded")])
def test_dry_run_benchmark_factory_switch(tmp_path, enabled, expected):
    progress = RunBatchProgressManager(1)
    progress.on_instance_start("org__repo-1")
    model = DeterministicToolcallModel(outputs=[model_turn(
        f"{sys.executable} -c 'raise AssertionError()'"), model_turn(SUBMIT)], cost_per_call=0)
    config = {"run": {"visible_counterexample_guard": enabled}, "agent": {
        "system_template": "Offline fixture", "instance_template": "{{task}}", "cost_limit": 0, "step_limit": 2}}
    agent = create_benchmark_agent(config, tmp_path, model, LocalEnvironment(cwd=str(tmp_path)),
                                   progress_manager=progress, instance_id="org__repo-1")
    assert agent.run("same fixture task")["exit_status"] == expected
    assert (tmp_path / "org__repo-1.traj.json").exists() == enabled
    assert ("Visible Counterexample Guard" in agent.messages[1]["content"]) == enabled
