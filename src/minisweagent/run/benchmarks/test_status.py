"""Conservative classification of executed Python tests/reproducers; never executes input."""

import re
import shlex


def _pytest_target(segments: list[list[str]]) -> str | None:
    invocations = []
    harmless = re.compile(r"^(?:-q+|-v+|-s|--quiet|--verbose|--capture=.*|--tb=.*|--color=.*|--disable-warnings|--strict-markers)$")
    value_options = {"--tb", "--capture", "--color", "--basetemp", "--rootdir", "--confcutdir",
                     "--junitxml", "--maxfail", "-p"}
    for original in segments:
        segment = list(original)
        while segment and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", segment[0]):
            segment.pop(0)
        if not segment:
            continue
        exe, start = segment[0].rsplit("/", 1)[-1], None
        if exe in {"pytest", "py.test"}:
            start = 1
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", exe) and segment[1:3] == ["-m", "pytest"]:
            start = 3
        if start is None or any(arg in {"--version", "--help", "--collect-only"} for arg in segment[start:]):
            continue
        targets, i = [], start
        while i < len(segment):
            arg = segment[i]
            if arg == "--":
                targets.extend(segment[i + 1:])
                break
            if harmless.match(arg) or re.match(r"^-[qvs]+$", arg):
                i += 1
                continue
            if arg in value_options:
                if i + 1 >= len(segment):
                    return None
                i += 2
                continue
            if arg.startswith("-"):
                return None
            if arg.isdigit() or any(char in arg for char in "*?[]{}$`"):
                return None
            targets.append(re.sub(r"^(?:\./)+", "", arg))
            i += 1
        if targets:
            invocations.append(tuple(targets))
    if len(invocations) != 1:
        return None
    return "pytest::" + "|".join(invocations[0])


def command_profile(command: str) -> dict:
    here_match = re.search(r"(?:^|\s)<<\s*['\"]?(\w+)", command.splitlines()[0] if command else "")
    heredoc = here_match is not None
    header = command.splitlines()[0] if heredoc else command
    if here_match:
        lines = command.splitlines()
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == here_match[1]), None)
        if end is not None and any(line.strip() for line in lines[end + 1:]):
            header += " ; " + "\n".join(lines[end + 1:])
    try:
        lexer = shlex.shlex(header, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return {"frameworks": [], "python": False, "safe_success": False,
                "verification_target": None, "pipeline": False}
    segments: list[list[str]] = [[]]
    unsafe = "`" in header or "$(" in header or ("\n" in command and not heredoc)
    pipeline = False
    skip_redirection_target = False
    if here_match and command.rstrip().splitlines()[-1].strip() != here_match[1]:
        unsafe = True
    for token in tokens:
        if token and all(c in ";&|()<>" for c in token):
            if any(c in "<>" for c in token):
                if segments[-1][-1:] and segments[-1][-1].isdigit():
                    segments[-1].pop()
                skip_redirection_target = True
                continue
            pipeline |= "|" in token
            unsafe |= token != "&&"
            segments.append([])
        elif skip_redirection_target:
            skip_redirection_target = False
        else:
            segments[-1].append(token)
    frameworks, python, executables = [], False, 0
    for segment in segments:
        while segment and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", segment[0]):
            segment = segment[1:]
        if not segment:
            continue
        exe = segment[0].rsplit("/", 1)[-1]
        if exe == "cd":
            continue
        executables += 1
        if exe in {"pytest", "py.test", "tox", "unittest"}:
            frameworks.append(exe)
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", exe):
            python = True
            if len(segment) >= 3 and segment[1] == "-m" and segment[2] in {"pytest", "unittest", "tox"}:
                frameworks.append(segment[2])
            elif len(segment) >= 3 and segment[1].endswith("manage.py") and segment[2] == "test":
                frameworks.append("django")
        elif exe == "manage.py" and segment[1:2] == ["test"]:
            frameworks.append("django")
    maintenance = bool(re.search(r"\.write_text\s*\(|\.write_bytes\s*\(|open\([^\n]+,\s*['\"](?:w|a)", command))
    return {"frameworks": frameworks, "python": python and not maintenance,
            "safe_success": not unsafe and executables == 1 and "--collect-only" not in tokens,
            "direct": "&&" not in tokens, "verification_target": _pytest_target(segments),
            "pipeline": pipeline}


def classify_execution(command: str, output: str, returncode: int | None,
                       exception_info: str = "", *, observed: bool = True) -> dict:
    """Output must be the observation visible to the agent when used at runtime."""
    profile = command_profile(command)
    result = {"status": "unknown", "kind": "test" if profile["frameworks"] else
              "reproducer" if profile["python"] else "other", "frameworks": profile["frameworks"],
              "reason": "no_confirmed_execution_result", "excerpt": "", "test_count": None,
              "verification_target": profile.get("verification_target"),
              "closure_requires_reliable_rerun": False, "reliable_verification": False}
    if not observed or exception_info == "action was not executed":
        return result
    if result["kind"] == "other":
        return result
    if re.search(r"timed out|TimeoutExpired|timeout", exception_info, re.I):
        return result | {"status": "timeout", "reason": "execution_timeout", "excerpt": exception_info[:1500]}
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    summary = [line for line in text.splitlines() if re.search(
        r"^\s*(?:=+\s*)?(?:\d+\s+(?:failed|passed|errors?|skipped|deselected)|FAILED\s*\(|ERROR collecting)", line)]
    failed = [line for line in summary if re.search(r"[1-9]\d*\s+(?:failed|errors?)\b|FAILED\s*\(|ERROR collecting", line)]
    if profile["frameworks"] and failed:
        return result | {"status": "failed", "reason": "visible_test_failure", "excerpt": "\n".join(failed)[-1500:],
                         "closure_requires_reliable_rerun": bool(profile["pipeline"])}
    if profile["python"] and "Traceback (most recent call last):" in text:
        errors = re.findall(r"(?m)^\s*(?:[\w.]+(?:Error|Exception)|AssertionError|SystemExit)(?::[^\n]*)?\s*$", text)
        if errors:
            return result | {"status": "failed", "reason": "visible_python_traceback", "excerpt": errors[-1].strip()[:1500]}
    if profile["frameworks"] and (re.search(r"no tests ran|collected 0 items|Ran 0 tests|Empty suite|ERROR: (?:not found|file or directory not found)", text, re.I)
                                    or (returncode == 5 and profile["safe_success"])):
        return result | {"status": "no_tests", "reason": "no_tests_or_invalid_test_target", "excerpt": text[-1500:]}
    if exception_info or "<elided_chars>" in text or not profile["safe_success"]:
        return result | {"reason": "exception_truncation_or_compound_exit_ambiguity"}
    if returncode == 0:
        return result | {"status": "passed", "reason": "direct_execution_success",
                         "excerpt": "\n".join(summary)[-1500:],
                         "reliable_verification": bool(profile.get("verification_target"))}
    if returncode == 1 and profile.get("direct") and any(f in {"pytest", "py.test"} for f in profile["frameworks"]):
        return result | {"status": "failed", "reason": "direct_pytest_failure", "excerpt": text[-1500:]}
    return result
