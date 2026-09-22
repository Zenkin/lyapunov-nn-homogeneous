"""Run experiments in separate timestamped directories."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "example1-original": "example_1.original.example1",
    "example1-improved": "example_1.improved.example1",
    "example2-article": "example_2.article_version.example2",
    "example2-corrected": "example_2.corrected_matrix.example2",
    "example2-improved": "example_2.improved.example2",
}


def requirements(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r "):
            pins.update(requirements(path.parent / line[3:].strip()))
        else:
            name, version = line.split("==")
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting dependency pins: {name}")
            pins[name] = version
    return pins


def probe() -> int:
    try:
        if sys.version_info[:2] != (3, 12):
            return 1
        for name, version in requirements(ROOT / "requirements.txt").items():
            if importlib.metadata.version(name) != version:
                return 1
            importlib.import_module(name)
        return 0
    except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError):
        return 1


def execute(arguments: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as output:
        output.write(
            "Command: " + subprocess.list2cmdline([sys.executable, *arguments]) + "\n"
        )
        output.flush()
        with subprocess.Popen(
            [sys.executable, *arguments],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
            return process.wait()


def expected_audit_failure(folder: Path, code: int, log: Path) -> bool:
    """Recognize the baseline's reported numerical failure, not arbitrary errors."""
    if code != 1 or not all(
        (folder / f).is_file() for f in ("metrics.json", "config.json", "models.pt")
    ):
        return False
    try:
        result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        return result["automated_audit"][
            "all_finite_grid_checks_passed"
        ] is False and "RuntimeError: Finite-grid audit failed:" in log.read_text(
            encoding="utf-8"
        )
    except (OSError, ValueError, KeyError):
        return False


def publish_gallery(folder: Path) -> None:
    """Index the latest successful reproduction without changing tracked files."""
    results = ROOT / "results"
    prefix = folder.resolve().relative_to(results.resolve()).as_posix() + "/"
    html = (folder / "RESULTS.html").read_text(encoding="utf-8")
    html = html.replace('href="', f'href="{prefix}').replace('src="', f'src="{prefix}')
    html = html.replace(
        "</html>",
        '<p><a href="../REPOSITORY_RESULTS.html">Other repository variants and the later checkpoint</a> · <a href="../reproduction/README.md">Reproduction protocol</a></p></html>',
    )
    (results / "LATEST.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--action",
        choices=("check", "quick", "full", "tables", "reproduce"),
        default="reproduce",
    )
    parser.add_argument("--experiment", choices=("all", *EXPERIMENTS), default="all")
    args = parser.parse_args()
    if args.probe:
        return probe()
    if args.action == "reproduce" and args.experiment != "all":
        parser.error(
            "Checkpoint reproduction runs both paper examples; use experiment all."
        )
    run_root = ROOT / "results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_root.mkdir(parents=True, exist_ok=False)
    summary = {"action": args.action, "python": sys.version, "runs": []}

    def record(name: str, code: int, status: str) -> None:
        summary["runs"].append({"name": name, "exit_code": code, "status": status})
        (run_root / "run-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Results: {run_root}", flush=True)
    if args.action == "reproduce":
        folder = run_root / "reproduction"
        code = execute(
            ["reproduction/reproduce.py", "--outdir", str(folder)],
            run_root / "reproduction.log",
        )
        record("checkpoint-reproduction", code, "verified" if code == 0 else "failed")
        if code == 0:
            publish_gallery(folder)
            import os

            if sys.platform == "win32":
                try:
                    os.startfile(str(folder / "RESULTS.html"))
                except OSError:
                    print(f'Open the gallery manually: {folder / "RESULTS.html"}')
        return 1 if code else 0
    if args.action == "quick":
        warning = "INSTALLATION TEST ONLY. Models are undertrained. These plots are NOT paper results. Use START.cmd reproduce."
        print(warning, flush=True)
        (run_root / "READ_FIRST.txt").write_text(warning + "\n", encoding="utf-8")
    if args.action in ("check", "quick", "full"):
        code = execute(["-m", "unittest", "discover", "-v"], run_root / "tests.log")
        record("tests", code, "passed" if code == 0 else "failed")
        if code or args.action == "check":
            return 1 if code else 0
    if args.action in ("quick", "full"):
        chosen = (
            EXPERIMENTS
            if args.experiment == "all"
            else {args.experiment: EXPERIMENTS[args.experiment]}
        )
        for name, module in chosen.items():
            folder = run_root / name
            command = ["-m", module, "--outdir", str(folder)]
            if args.action == "quick":
                command.append("--quick")
            print(f"\nRunning {name} ({args.action})...", flush=True)
            code = execute(command, run_root / f"{name}.log")
            audit_failure = (
                args.action == "full"
                and name == "example1-original"
                and expected_audit_failure(folder, code, run_root / f"{name}.log")
            )
            status = (
                ("smoke_only_not_validated" if args.action == "quick" else "completed")
                if code == 0
                else ("audit_failed" if audit_failure else "failed")
            )
            record(name, code, status)
    if args.action == "tables":
        code = execute(
            ["publication/make_summary.py", "--outdir", str(run_root / "tables")],
            run_root / "tables.log",
        )
        record("tables", code, "completed" if code == 0 else "failed")
    statuses = [item["status"] for item in summary["runs"]]
    print(f"\nSaved: {run_root / 'run-summary.json'}")
    if "failed" in statuses:
        print("One or more commands failed. See the logs.")
        return 1
    if "audit_failed" in statuses:
        print(
            "Runs finished; the original Example 1 reported numerical audit violations. See its metrics and log."
        )
        return 2
    print("All requested commands completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
