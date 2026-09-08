#!/usr/bin/env python3
"""Validate circuit syntax and submission folders without running SPICE or generators."""
import argparse
from pathlib import Path

try:
    from .runner import validate_submission
except ImportError:
    from runner import validate_submission


ROOT = Path(__file__).resolve().parents[1]


def validate_folder(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"submission must be a real directory: {path}")
    for name in ("circuit.cir", "README.md", "LICENSE"):
        item = path / name
        if item.is_symlink() or not item.is_file() or not item.stat().st_size:
            raise ValueError(f"submission needs a nonempty, non-symlink {name}: {path}")
    # Legal permission/attribution requires human review; this only checks presence.
    return validate_submission(path / "circuit.cir")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="netlist files or submission directories")
    parser.add_argument("--all", action="store_true", help="check bundled examples and all submission directories")
    args = parser.parse_args(argv)
    paths = list(args.paths)
    if args.all:
        paths += sorted((ROOT / "competition/examples").glob("*.cir"))
        paths += sorted(path for path in (ROOT / "submissions").iterdir() if path.is_dir() or path.is_symlink())
    if not paths:
        parser.error("supply a circuit/submission path or --all")
    failed = False
    for path in paths:
        try:
            if path.is_symlink():
                raise ValueError("symlink submissions are not accepted")
            _, counts, digest = validate_folder(path) if path.is_dir() else validate_submission(path)
            print(f"OK {path}: {sum(counts.values())} components; SHA-256 {digest}")
        except (OSError, ValueError) as error:
            print(f"ERROR {path}: {error}")
            failed = True
    if failed:
        parser.exit(1)


if __name__ == "__main__":
    main()
