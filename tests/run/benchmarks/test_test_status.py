import pytest

from minisweagent.run.benchmarks.test_status import classify_execution


@pytest.mark.parametrize(("command", "output", "code", "exception", "expected"), [
    ("pytest -q", "3 passed in .1s", 0, "", "passed"),
    ("cd /testbed && python -m pytest -q", "3 passed", 0, "", "passed"),
    ("pytest | head", "1 failed, 2 passed", 0, "", "failed"),
    ("pytest | head", "3 passed", 0, "", "unknown"),
    ("pytest | head", "..", 0, "", "unknown"),
    ("false && pytest", "", 1, "", "unknown"),
    ("cd missing && pytest", "cd: missing: No such file or directory", 1, "", "unknown"),
    ("pytest", "ERROR collecting test_file.py", 2, "", "failed"),
    ("pytest", "no tests ran", 5, "", "no_tests"),
    ("pytest wrong.py", "ERROR: file or directory not found: wrong.py", 4, "", "no_tests"),
    ("pytest", "", -1, "Command timed out after 1 second", "timeout"),
    ("pytest", "1 passed", -1, "Command timed out", "timeout"),
    ("pytest --collect-only", "3 tests collected", 0, "", "unknown"),
    ("cat tests/test_x.py", "1 failed", 0, "", "unknown"),
    ("grep pytest test_x.py", "FAILED (errors=1)", 0, "", "unknown"),
    ("echo 'pytest failed'", "1 failed", 0, "", "unknown"),
    ("python - <<'PY'\nraise AssertionError('counterexample')\nPY", "Traceback (most recent call last):\nAssertionError: counterexample", 1, "", "failed"),
    ("python - <<'PY'\nassert True\nPY", "", 0, "", "passed"),
    ("python - <<'PY'\nassert True\nPY\necho done", "done", 0, "", "unknown"),
    ("python -c 'print(1)'", "Traceback (most recent call last):\nmarshmallow.exceptions.ValidationError: invalid", 0, "", "failed"),
    ("python -m unittest", "Ran 1 test\nFAILED (failures=1)", 1, "", "failed"),
    ("python manage.py test", "Ran 0 tests\nOK", 0, "", "no_tests"),
    ("pytest", "1 passed\n<elided_chars>hidden</elided_chars>", 0, "", "unknown"),
    ("python - <<'PY'\nassert old in s\np.write_text(s)\nPY", "Traceback (most recent call last):\nAssertionError", 1, "", "unknown"),
    ("python - <<'PY'\np.write_text(s)\nPY\npytest -q", "1 failed, 2 passed", 1, "", "failed"),
])
def test_status(command, output, code, exception, expected):
    assert classify_execution(command, output, code, exception)["status"] == expected


def test_unexecuted_and_quoted_test_names():
    assert classify_execution("pytest", "1 failed", 1, observed=False)["status"] == "unknown"
    assert classify_execution("python -c 'print(\"pytest failed\")'", "pytest failed", 0)["kind"] == "reproducer"
    assert classify_execution("cat test.py", "Traceback (most recent call last):\nAssertionError", 1)["kind"] == "other"


def test_pytest_verification_targets_are_conservative():
    piped = classify_execution(
        "cd /testbed && python -m pytest -q tests/test_a.py::TestA::test_x 2>&1 | head -40",
        "1 failed", 0,
    )
    direct = classify_execution(
        "cd /testbed && python -m pytest tests/test_a.py::TestA::test_x -q -s 2>&1",
        "1 passed", 0,
    )
    other = classify_execution("pytest tests/test_b.py -q", "1 passed", 0)
    ambiguous = classify_execution("pytest -k expression tests/test_a.py", "1 passed", 0)
    assert piped["verification_target"] == direct["verification_target"] == "pytest::tests/test_a.py::TestA::test_x"
    assert piped["closure_requires_reliable_rerun"] is True and piped["status"] == "failed"
    assert direct["reliable_verification"] is True and direct["status"] == "passed"
    assert other["verification_target"] != direct["verification_target"]
    assert ambiguous["verification_target"] is None and ambiguous["reliable_verification"] is False
