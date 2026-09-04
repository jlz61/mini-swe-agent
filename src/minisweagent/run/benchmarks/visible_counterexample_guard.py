"""Host-side visible-evidence policy. No dataset, reference patch, or grading input."""

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, Submitted
from minisweagent.run.benchmarks.test_status import classify_execution
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
POLICY = """
Visible Counterexample Guard is enabled. A visible failing test or Python traceback
creates unresolved evidence. A reliable pytest rerun of the same normalized test
target, or a passed rerun of the same command, can resolve it;
timeout/no-tests/unknown cannot. Do not weaken tests or reproductions to pass.
Unrelated/environment explanations require evidence-backed host review, not an
unsupported assertion. A blocked submission requests one state summary and
re-analysis per unresolved evidence set. Resolve evidence before resubmitting.
The case terminates safely after bounded blocked submissions or recovery steps.
"""


class GuardUnresolvedEvidence(InterruptAgentFlow):
    """The bounded Guard recovery ended with visible evidence still unresolved."""


@dataclass
class EvidenceLedger:
    max_submit_blocks: int = 3
    observations: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    triggered: set[tuple[str, ...]] = field(default_factory=set)
    group_submit_blocks: dict[tuple[str, ...], int] = field(default_factory=dict)
    reviews: list[dict] = field(default_factory=list)
    termination: dict | None = None

    @property
    def unresolved(self) -> list[dict]:
        return [e for e in self.evidence if e["resolution"] is None]

    def observe(self, command: str, visible_output: str, returncode: int | None,
                exception_info: str, step: int) -> dict:
        result = classify_execution(command, visible_output, returncode, exception_info)
        identity = hashlib.sha256(command.strip().encode()).hexdigest()
        observation = {"observation_id": f"obs-{len(self.observations) + 1}", "step": step,
                       "command": command, "command_hash": identity, "visible_output": visible_output,
                       "returncode": returncode, **result}
        self.observations.append(observation)
        if result["status"] == "failed":
            existing = next((e for e in self.unresolved if e["command_hash"] == identity), None)
            if existing:
                existing["observations"].append(observation["observation_id"])
            else:
                self.evidence.append({"evidence_id": f"evidence-{len(self.evidence) + 1}",
                                      "command": command, "command_hash": identity, "first_step": step,
                                      "reason": result["reason"], "excerpt": result["excerpt"],
                                      "verification_target": result.get("verification_target"),
                                      "closure_requires_reliable_rerun": result.get(
                                          "closure_requires_reliable_rerun", False),
                                      "observations": [observation["observation_id"]], "resolution": None})
        elif result["status"] == "passed":
            for evidence in self.unresolved:
                if evidence["command_hash"] == identity:
                    evidence["resolution"] = {"kind": "passed_retest", "step": step,
                                               "observation_id": observation["observation_id"],
                                               "closure_method": "exact_command_passed_retest"}
                elif (result.get("reliable_verification") and result.get("verification_target")
                      and evidence.get("verification_target") == result["verification_target"]):
                    evidence["resolution"] = {"kind": "passed_target_retest", "step": step,
                                               "observation_id": observation["observation_id"],
                                               "verification_target": result["verification_target"],
                                               "closure_method": "reliable_pytest_target_passed"}
        return observation

    def review(self, decision: dict) -> None:
        """Trusted host API only: require failure citation and independent corroboration."""
        evidence = next((e for e in self.unresolved if e["evidence_id"] == decision.get("evidence_id")), None)
        if evidence is None:
            raise ValueError("Review must identify unresolved evidence")
        if decision.get("resolution") not in {"environment", "unrelated"}:
            raise ValueError("Unsupported review resolution")
        if not str(decision.get("reviewer", "")).startswith("host:") or len(decision.get("reason", "")) < 20:
            raise ValueError("Trusted reviewer and substantive reason required")
        by_id = {o["observation_id"]: o for o in self.observations}
        cited = set()
        for citation in decision.get("citations", []):
            oid, quote = citation.get("observation_id"), citation.get("quote", "")
            if oid not in by_id or len(quote.strip()) < 12 or quote not in by_id[oid]["visible_output"]:
                raise ValueError("Review quotes must match visible observations")
            cited.add(oid)
        if not cited.intersection(evidence["observations"]) or not cited.difference(evidence["observations"]):
            raise ValueError("Failure and independent corroborating observations required")
        evidence["resolution"] = copy.deepcopy(decision)
        self.reviews.append(copy.deepcopy(decision))

    def block_submission(self, step: int) -> dict | None:
        if not self.unresolved:
            return None
        key = tuple(e["evidence_id"] for e in self.unresolved)
        first = key not in self.triggered
        self.triggered.add(key)
        self.group_submit_blocks[key] = self.group_submit_blocks.get(key, 0) + 1
        terminate = self.group_submit_blocks[key] >= self.max_submit_blocks or len(self.events) + 1 >= self.max_submit_blocks
        reason = ("max_evidence_group_submit_blocks_reached" if self.group_submit_blocks[key] >= self.max_submit_blocks
                  else "max_total_submit_blocks_reached") if terminate else None
        event = {"recovery_trigger": first, "trigger_step": step,
                 "unresolved_evidence": copy.deepcopy(self.unresolved),
                 "evidence_group": list(key), "evidence_group_submit_blocks": self.group_submit_blocks[key],
                 "total_submit_blocks": len(self.events) + 1, "max_submit_blocks": self.max_submit_blocks,
                 "recovery_reason": "visible_counterexample_unresolved", "submit_blocked": True,
                 "terminate": terminate, "termination_reason": reason}
        self.events.append(event)
        return event

    def terminate(self, reason: str, step: int) -> dict:
        if self.termination is None:
            self.termination = {"guard_termination_reason": reason, "termination_step": step,
                                "guard_unresolved_evidence": copy.deepcopy(self.unresolved),
                                "submit_blocks": len(self.events)}
        return self.termination

    def serialize(self) -> dict:
        return {"schema_version": 1, "enabled": True, "observations": self.observations,
                "evidence": self.evidence, "unresolved_evidence": self.unresolved,
                "recovery_events": self.events, "reviews": self.reviews,
                "max_submit_blocks": self.max_submit_blocks,
                "guard_unresolved_evidence": self.termination["guard_unresolved_evidence"] if self.termination else [],
                "guard_termination_reason": self.termination["guard_termination_reason"] if self.termination else None,
                "termination": self.termination}


class VisibleGuardMixin:
    def __init__(self, *args, guard_review_file: str | None = None, guard_max_submit_blocks: int = 3,
                 guard_max_recovery_steps: int = 24, **kwargs):
        if (not isinstance(guard_max_submit_blocks, int) or isinstance(guard_max_submit_blocks, bool)
                or guard_max_submit_blocks < 1):
            raise ValueError("guard_max_submit_blocks must be a positive integer")
        if (not isinstance(guard_max_recovery_steps, int) or isinstance(guard_max_recovery_steps, bool)
                or guard_max_recovery_steps < 1):
            raise ValueError("guard_max_recovery_steps must be a positive integer")
        super().__init__(*args, **kwargs)
        self.guard_review_file = Path(guard_review_file) if guard_review_file else None
        self.guard_max_submit_blocks = guard_max_submit_blocks
        self.guard_max_recovery_steps = guard_max_recovery_steps
        self.ledger = EvidenceLedger(max_submit_blocks=guard_max_submit_blocks)
        self.review_errors: list[str] = []

    def run(self, task: str = "", **kwargs) -> dict:
        self.ledger = EvidenceLedger(max_submit_blocks=self.guard_max_submit_blocks)
        self.review_errors = []
        return super().run(task + "\n" + POLICY, **kwargs)

    def _terminate_guard(self, reason: str) -> None:
        termination = self.ledger.terminate(reason, self.n_calls)
        raise GuardUnresolvedEvidence({"role": "exit", "content": "GuardUnresolvedEvidence",
                                       "extra": {"exit_status": "GuardUnresolvedEvidence", "submission": "",
                                                 **copy.deepcopy(termination)}})

    def query(self) -> dict:
        triggers = [event["trigger_step"] for event in self.ledger.events if event["recovery_trigger"]]
        if (self.ledger.unresolved and triggers
                and self.n_calls - min(triggers) >= self.guard_max_recovery_steps):
            self._terminate_guard("max_recovery_steps_reached")
        return super().query()

    def _guard_submission(self) -> dict | None:
        if self.guard_review_file and self.guard_review_file.is_file():
            try:
                data = json.loads(self.guard_review_file.read_text(encoding="utf-8"))
                if data.get("instance_id") != getattr(self, "instance_id", ""):
                    raise ValueError("Review instance_id mismatch")
                for decision in data["decisions"]:
                    if decision not in self.ledger.reviews:
                        self.ledger.review(decision)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.review_errors.append(f"ReviewRejected: {exc}")
        event = self.ledger.block_submission(self.n_calls)
        if event is None:
            return None
        if event["terminate"]:
            self._terminate_guard(event["termination_reason"])
        instruction = ("Summarize current state, unresolved counterexamples, and a discriminating retest plan; "
                       "then re-analyze before continuing." if event["recovery_trigger"] else
                       "Submission remains blocked. Resolve the evidence; repeating Submit does not bypass the guard.")
        return {"returncode": 1, "exception_info": "", "output": "VISIBLE_COUNTEREXAMPLE_GUARD\n" +
                instruction + "\n" + json.dumps(event, ensure_ascii=False), "extra": {"visible_guard": event}}

    def execute_actions(self, message: dict) -> list[dict]:
        observations = []
        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            command = action.get("command", "")
            blocked = self._guard_submission() if SUBMIT in command else None
            if blocked is not None:
                output = blocked
            else:
                try:
                    output = self.env.execute(action)
                except Submitted:
                    # Covers computed markers as well as the ordinary literal submission command.
                    output = self._guard_submission()
                    if output is None:
                        raise
                    blocked = output
            single = {**message, "extra": {**message.get("extra", {}), "actions": [action]}}
            formatted = self.model.format_observation_messages(single, [output], self.get_template_vars())
            observations.extend(self.add_messages(*formatted))
            if blocked is None:
                visible = "\n".join(m["content"] for m in formatted if isinstance(m.get("content"), str))
                code_match = re.search(r"<returncode>\s*(-?\d+)\s*</returncode>", visible)
                exception = output.get("exception_info") or ""
                self.ledger.observe(command, visible, int(code_match[1]) if code_match else None,
                                    exception if exception in visible else "", self.n_calls)
        return observations

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"visible_counterexample_guard": {
            **self.ledger.serialize(), "review_errors": self.review_errors}}}, *extra_dicts)


class VisibleCounterexampleAgent(VisibleGuardMixin, DefaultAgent):
    """Standalone policy wrapper for local dry-runs."""


class GuardedProgressTrackingAgent(VisibleGuardMixin, ProgressTrackingAgent):
    """Benchmark adapter; keeps the upstream loop and progress reporting."""


def benchmark_agent_class(config: dict) -> type:
    enabled = config.get("run", {}).get("visible_counterexample_guard", False)
    if not isinstance(enabled, bool):
        raise ValueError("run.visible_counterexample_guard must be a boolean")
    return GuardedProgressTrackingAgent if enabled else ProgressTrackingAgent


def create_benchmark_agent(config: dict, instance_dir: Path, *args, **kwargs):
    agent_class = benchmark_agent_class(config)
    options = dict(config.get("agent", {}))
    if agent_class is GuardedProgressTrackingAgent:
        options.setdefault("output_path", instance_dir / f"{kwargs['instance_id']}.traj.json")
    return agent_class(*args, **kwargs, **options)
