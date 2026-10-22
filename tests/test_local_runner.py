import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pseudo_local", Path(__file__).resolve().parents[1] / "scripts/pseudo_local.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    state = tmp_path / "ui state"
    data = tmp_path / "engine data"
    source = state / "inputs" / "sample document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("Synthetic example", encoding="utf-8")
    result = data / "outputs/pseudonymization/demo/sample document_ollama_test_20261022_120000"
    result.mkdir(parents=True)
    (result / "findings.json").write_text("{}", encoding="utf-8")
    (result / "pseudonymized.txt").write_text("PERSON_A", encoding="utf-8")
    monkeypatch.setenv("PSEUDO_UI_STATE_DIR", str(state))
    monkeypatch.setenv("DATA_DIR", str(data))
    return state, source, result


def fake_process(result, rc=0):
    class Process:
        stdout = iter([f"\x1b[32m{runner.MARKER}{result}/\x1b[0m\n"])
        def wait(self):
            return rc
    return Process()


def test_copies_artifacts_for_ui(local_run, monkeypatch):
    state, source, result = local_run
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return fake_process(result)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.run(source, "demo") == 0
    assert (state / "outputs" / result.name / "pseudonymized.txt").read_text() == "PERSON_A"
    assert calls[0][0][1:4] == ["-m", "app", "run"]
    assert calls[0][1]["env"]["DATA_DIR"] == str(result.parents[3])


def test_rejects_external_input(local_run, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("example")
    with pytest.raises(ValueError, match="UI inputs"):
        runner.run(outside, "demo")


def test_rejects_external_output(local_run, monkeypatch, tmp_path):
    _, source, _ = local_run
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: fake_process(tmp_path / "outside"))
    with pytest.raises(RuntimeError, match="Unexpected engine output"):
        runner.run(source, "demo")


def test_propagates_engine_failure(local_run, monkeypatch):
    state, source, result = local_run
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: fake_process(result, 7))
    assert runner.run(source, "demo") == 7
    assert not (state / "outputs").exists()
