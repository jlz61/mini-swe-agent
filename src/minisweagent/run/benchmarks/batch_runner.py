"""Sequential subprocess wrapper around the existing SWE-bench run script."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from minisweagent.run.benchmarks.trajectory_parser import TrajectoryError, parse_file

app = typer.Typer(add_completion=False)
PrepareCommand = Callable[[str, Path], list[str]]


class CaseError(ValueError):
    """An expected case failure that must not abort the batch."""


@dataclass
class BatchConfig:
    subset: str
    split: str = "dev"
    model: str | None = None
    config_specs: tuple[str, ...] = ()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_ids(instance_ids: list[str]) -> list[str]:
    if not instance_ids:
        raise ValueError("At least one --instance-id is required")
    for instance_id in instance_ids:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*__[A-Za-z0-9][A-Za-z0-9_.-]*", instance_id):
            raise ValueError(f"Invalid instance ID: {instance_id!r}")
    return list(dict.fromkeys(instance_ids))


def build_command(config: BatchConfig, instance_id: str, raw: Path, run_args: list[str]) -> list[str]:
    command = [sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
               "--subset", config.subset, "--split", config.split,
               "--filter", f"^{re.escape(instance_id)}$", "--workers", "1", "--output", str(raw)]
    if config.model:
        command.extend(["--model", config.model])
    for spec in ("swebench.yaml", *config.config_specs):
        command.extend(["-c", spec])
    command.extend(["-c", "environment.environment_class=docker",
                    "-c", "environment.run_args=" + json.dumps(run_args)])
    return command


def make_preparer(config: BatchConfig, env: dict[str, str]) -> PrepareCommand:
    from datasets import load_dataset

    from minisweagent.config import get_config_from_spec
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_swebench_docker_image_name
    from minisweagent.utils.serialize import recursive_merge

    resolved = recursive_merge(*[get_config_from_spec(s) for s in ("swebench.yaml", *config.config_specs)])
    environment = resolved.get("environment", {})
    if environment.get("environment_class", "docker") != "docker":
        raise ValueError("v1 supports only the existing Docker environment")
    docker = environment.get("executable") or env.get("MSWEA_DOCKER_EXECUTABLE", "docker")
    if not isinstance(docker, str):
        raise ValueError("Docker executable must be a string")
    run_args = environment.get("run_args", ["--rm"])
    if not isinstance(run_args, list) or any(not isinstance(a, str) for a in run_args):
        raise ValueError("environment.run_args must be a list of strings")
    # Preserve user arguments except pull policy: neither preflight races nor config may trigger a pull.
    safe_args = []
    skip = False
    for arg in run_args:
        if skip:
            skip = False
        elif arg == "--pull":
            skip = True
        elif not arg.startswith("--pull="):
            safe_args.append(arg)
    safe_args.append("--pull=never")
    dataset = {row["instance_id"]: row for row in load_dataset(DATASET_MAPPING.get(config.subset, config.subset),
                                                              split=config.split)}

    def prepare(instance_id: str, raw: Path) -> list[str]:
        if instance_id not in dataset:
            raise CaseError("InstanceNotFound: requested ID is absent from the selected split")
        image = get_swebench_docker_image_name(dataset[instance_id])
        try:
            result = subprocess.run([docker, "image", "inspect", image], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaseError(f"DockerPreflightError: {type(exc).__name__}") from exc
        if result.returncode:
            raise CaseError(f"ImageUnavailable: {image} (or Docker daemon inaccessible)")
        return build_command(config, instance_id, raw, safe_args)

    return prepare


def _execute(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, bool]:
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=os.name == "posix")
        try:
            return process.wait(), False
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
            while process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                except KeyboardInterrupt:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
            return process.returncode, True


def _collect(case: dict, directory: Path) -> dict | None:
    instance_id, raw = case["instance_id"], directory / "raw"
    prediction = None
    preds_path = raw / "preds.json"
    if preds_path.is_file():
        try:
            predictions = json.loads(preds_path.read_text(encoding="utf-8"))
            candidate = predictions.get(instance_id) if isinstance(predictions, dict) else None
            if not isinstance(candidate, dict) or candidate.get("instance_id") != instance_id:
                raise CaseError("Prediction instance_id mismatch or missing")
            if not isinstance(candidate.get("model_name_or_path"), str) or not candidate["model_name_or_path"]:
                raise CaseError("Missing prediction model_name_or_path")
            if not isinstance(candidate.get("model_patch"), str):
                raise CaseError("Prediction model_patch must be a string")
            prediction = {key: candidate[key] for key in ("instance_id", "model_name_or_path", "model_patch")}
            _write_json(directory / "prediction.json", prediction)
            (directory / "patch.diff").write_text(prediction["model_patch"], encoding="utf-8")
            case["artifacts"].update(prediction=str(directory / "prediction.json"), patch=str(directory / "patch.diff"))
        except (OSError, ValueError) as exc:
            case["errors"].append(f"InvalidPrediction: {exc}")
    else:
        case["errors"].append("MissingPrediction")

    trajectory = raw / instance_id / f"{instance_id}.traj.json"
    if trajectory.is_file():
        case["artifacts"]["trajectory"] = str(trajectory)
        try:
            data = json.loads(trajectory.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("instance_id") != instance_id:
                raise TrajectoryError("Trajectory instance_id mismatch or missing")
            parsed = parse_file(trajectory, instance_id=instance_id, runtime_seconds=case["runtime_seconds"],
                                patch=prediction["model_patch"] if prediction else None)
            _write_json(directory / "parsed.json", parsed)
            case["artifacts"]["parsed"] = str(directory / "parsed.json")
            case["parse_status"] = "parsed"
            case["agent_status"], case["exit_status"] = parsed["status"], parsed["exit_status"]
            case.update({key: parsed[key] for key in ("total_steps", "api_calls", "cost")})
            info = data.get("info")
            submission = info.get("submission") if isinstance(info, dict) else None
            if prediction is not None and isinstance(submission, str) and submission != prediction["model_patch"]:
                case["errors"].append("ArtifactMismatch: trajectory and prediction patches differ")
        except (OSError, ValueError) as exc:
            case["parse_status"] = "error"
            case["errors"].append(f"TrajectoryParseError: {exc}")
    else:
        case["parse_status"] = "missing"
        case["errors"].append("MissingTrajectory")

    statuses = set()
    for path in raw.glob("exit_statuses_*.yaml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            for status, ids in data["instances_by_exit_status"].items():
                if isinstance(ids, list) and instance_id in ids:
                    statuses.add(str(status))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
            case["errors"].append("InvalidStatusYaml")
    if len(statuses) > 1 or (statuses and case["exit_status"] and case["exit_status"] not in statuses):
        case["errors"].append("ArtifactMismatch: conflicting exit statuses")
    elif statuses and not case["exit_status"]:
        case["exit_status"] = next(iter(statuses))
        case["agent_status"] = "submitted" if case["exit_status"] == "Submitted" else "failed"
    return prediction


def _summary(output: Path, started_at: str, cases: list[dict], predictions: list[dict]) -> None:
    status_counts = dict(Counter(case["status"] for case in cases))
    metrics = {}
    for key in ("api_calls", "cost"):
        known = [case[key] for case in cases if case[key] is not None]
        metrics[key] = {"known_sum": sum(known), "unknown_cases": len(cases) - len(known)}
    _write_json(output / "summary.json", {
        "schema_version": 1, "started_at": started_at, "updated_at": _utc(),
        "requested_instance_ids": [case["instance_id"] for case in cases], "case_count": len(cases),
        "status_counts": status_counts, "metrics": metrics, "cost_currency": "USD",
        "evaluation": "not_run", "cases": cases,
        "missing_prediction_ids": [case["instance_id"] for case in cases if not case["artifacts"]["prediction"]],
    })
    temporary = output / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
    temporary.replace(output / "predictions.jsonl")


def run_batch(instance_ids: list[str], output: Path, prepare: PrepareCommand, *, env: dict[str, str] | None = None) -> int:
    """Run prepared commands serially; the preparation seam allows real, non-model test subprocesses."""
    instance_ids = validate_ids(instance_ids)
    output = output.absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = dict(os.environ if env is None else env)
    cases = [{
        "instance_id": iid, "status": "not_run", "agent_status": "unknown", "parse_status": "not_run",
        "exit_status": None, "returncode": None, "started_at": None, "finished_at": None,
        "runtime_seconds": None, "total_steps": None, "api_calls": None, "cost": None, "errors": [],
        "artifacts": {key: None for key in ("trajectory", "prediction", "patch", "parsed", "log", "run")},
    } for iid in instance_ids]
    started_at, predictions = _utc(), []
    _summary(output, started_at, cases, predictions)
    for case in cases:
        directory = output / "cases" / case["instance_id"]
        raw = directory / "raw"
        raw.mkdir(parents=True)
        log_path = directory / "runner.log"
        log_path.touch()
        case["artifacts"].update(log=str(log_path), run=str(directory / "run.json"))
        case["status"], case["started_at"] = "running", _utc()
        _summary(output, started_at, cases, predictions)
        interrupted = False
        try:
            command = prepare(case["instance_id"], raw)
            began = time.monotonic()
            try:
                case["returncode"], interrupted = _execute(command, log_path, env)
            finally:
                case["runtime_seconds"] = time.monotonic() - began
            if case["returncode"] != 0:
                case["errors"].append(f"ProcessExit: {case['returncode']}")
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            # Per-case containment is intentional; global output filesystem failures remain fatal.
            case["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            case["finished_at"] = _utc()
        try:
            prediction = _collect(case, directory)
            if prediction is not None:
                predictions.append(prediction)
        except KeyboardInterrupt:
            interrupted = True
            case["parse_status"] = "interrupted"
        except Exception as exc:
            case["parse_status"] = "error"
            case["errors"].append(f"ArtifactCollectionError: {type(exc).__name__}")
        case["status"] = "submitted" if case["agent_status"] == "submitted" and not case["errors"] else "failed"
        if interrupted:
            case["status"] = "interrupted"
        _write_json(directory / "run.json", case)
        _summary(output, started_at, cases, predictions)
        typer.echo(f"{case['instance_id']}: {case['status']} (parse={case['parse_status']})")
        if interrupted:
            return 130
    return int(any(case["status"] != "submitted" for case in cases))


@app.command()
def main(instance_ids: list[str] = typer.Option(..., "--instance-id"),
         subset: str = typer.Option(..., "--subset"), split: str = typer.Option("dev", "--split"),
         output: Path = typer.Option(..., "--output"), model: str | None = typer.Option(None, "--model"),
         configs: list[str] | None = typer.Option(None, "-c", "--config")) -> None:
    try:
        validate_ids(instance_ids)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output must be a new or empty directory")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Process-local only; dotenv, Conda config and the user's shell remain unchanged.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    env = dict(os.environ)
    config = BatchConfig(subset, split, model, tuple(configs or ()))
    try:
        prepare = make_preparer(config, env)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        error = f"BatchPreflightError: {type(exc).__name__} (check local dataset and CLI dependencies)"

        def prepare(instance_id: str, raw: Path) -> list[str]:
            raise CaseError(error)

    try:
        code = run_batch(instance_ids, output, prepare, env=env)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
