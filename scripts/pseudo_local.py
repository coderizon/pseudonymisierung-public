"""Connect the local review UI to the engine without sudo, Docker or systemd."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MARKER = "Verbose-Ausgabe geschrieben: "


def run(file, project):
    state = Path(os.environ.get("PSEUDO_UI_STATE_DIR", str(ROOT / "data/ui"))).resolve()
    inputs = (state / "inputs").resolve()
    source = Path(file).resolve()
    if not source.is_file() or not source.is_relative_to(inputs):
        raise ValueError("Input must be a file in the UI inputs directory.")
    if source.suffix.lower() not in {".pdf", ".docx", ".xlsx", ".md", ".txt"}:
        raise ValueError("Unsupported file type.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", project):
        raise ValueError("Invalid project identifier.")
    env = os.environ.copy()
    data = Path(env.get("DATA_DIR", str(ROOT / "data/engine"))).resolve()
    env["DATA_DIR"] = str(data)
    env.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "app", "run", str(source), "--project", project, "--verbose"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    result = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        clean = ANSI.sub("", line).strip()
        if MARKER in clean:
            result = Path(clean.split(MARKER, 1)[1].rstrip("/"))
            if not result.is_absolute():
                result = ROOT / result
    rc = proc.wait()
    if rc:
        return rc
    if result is None:
        raise RuntimeError("Engine did not report its output directory.")
    result = result.resolve()
    if not result.is_relative_to((data / "outputs").resolve()):
        raise RuntimeError("Unexpected engine output path.")
    if not (result / "findings.json").is_file():
        raise RuntimeError("Engine output is incomplete.")
    output_root = state / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / result.name
    shutil.copytree(result, target)
    print(f"Ergebnisse fuer die lokale Review-UI: {target}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    try:
        return run(args.file, args.project)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
