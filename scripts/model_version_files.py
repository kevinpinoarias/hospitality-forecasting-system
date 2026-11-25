"""
The forecast log's registered model files, kept as assets of one GitHub release so a build can fetch them.

The files are too large for git (about 205 MB) and must never change, so they live as release assets: one zip
per version (its model pieces and its Saturday and Sunday correction history, at the paths the API reads) and
the versions' backtest file. `verify` checks every installed file against the fingerprints in
config/model_versions.json; a build stops if anything is missing or differs, so nothing unchecked goes live.

  python scripts/model_version_files.py pack --out <folder>        zips to upload (maintainer)
  python scripts/model_version_files.py fetch                      download and unpack into place (CI; needs gh)
  python scripts/model_version_files.py verify                     check the installed files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHOICES = ROOT / "config" / "model_versions.json"
MODELS = ROOT / "models"
CORRECTIONS = ROOT / "data" / "training"
BACKTEST = MODELS / "version_backtests.csv"
RELEASE_TAG = "model-versions-1"


def config() -> dict:
    return json.loads(CHOICES.read_text(encoding="utf-8"))


def versions() -> list[str]:
    return sorted({c["version"] for c in config()["choices"] if c["id"] != "standard"})


def fingerprint(directory: Path) -> str:
    """sha256 of every file in a folder, sorted by name and concatenated (the forecast log's convention; as src/forecast_core/bags.py)."""
    h = hashlib.sha256()
    for f in sorted(p for p in directory.iterdir() if p.is_file()):
        h.update(f.read_bytes())
    return h.hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for version in versions():
        target = out / f"{version}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted((MODELS / version).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT).as_posix())
            history = CORRECTIONS / f"{version}_correction_history.csv"
            z.write(history, history.relative_to(ROOT).as_posix())
        print(f"packed {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    if BACKTEST.exists():
        (out / BACKTEST.name).write_bytes(BACKTEST.read_bytes())
        print(f"copied {BACKTEST.name}")


def fetch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["gh", "release", "download", RELEASE_TAG, "--dir", tmp, "--pattern", "*.zip", "--pattern", BACKTEST.name], check=True)
        for z in sorted(Path(tmp).glob("*.zip")):
            with zipfile.ZipFile(z) as archive:
                archive.extractall(ROOT)
            print(f"unpacked {z.name}")
        downloaded = Path(tmp) / BACKTEST.name
        if downloaded.exists():
            MODELS.mkdir(exist_ok=True)
            BACKTEST.write_bytes(downloaded.read_bytes())
            print(f"installed {BACKTEST.name}")


def verify() -> bool:
    cfg, problems = config(), []
    for c in cfg["choices"]:
        if c["id"] == "standard":
            continue
        for name, expected in c["pieces"].items():
            folder = MODELS / c["version"] / name
            if not folder.is_dir():
                problems.append(f"{c['id']}: {folder.relative_to(ROOT)} is missing")
            elif fingerprint(folder) != expected:
                problems.append(f"{c['id']}: {folder.relative_to(ROOT)} does not match its registered fingerprint")
        history = CORRECTIONS / f"{c['version']}_correction_history.csv"
        if not history.exists() or sha256(history) != c["correction_history_sha256"]:
            problems.append(f"{c['id']}: {history.relative_to(ROOT)} is missing or does not match")
    if cfg.get("backtest_sha256") and (not BACKTEST.exists() or sha256(BACKTEST) != cfg["backtest_sha256"]):
        problems.append(f"{BACKTEST.relative_to(ROOT)} is missing or does not match")
    for p in sorted(set(problems)):
        print("PROBLEM:", p)
    print("all model-version files present and matching" if not problems else f"{len(set(problems))} problem(s)")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pack").add_argument("--out", type=Path, required=True)
    sub.add_parser("fetch")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.out)
    elif args.command == "fetch":
        fetch()
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
