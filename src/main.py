"""Run the complete staged Residual Paving pipeline from a JSON recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def recipe_arguments(path: Path) -> list[str]:
    recipe = json.loads(path.read_text())
    arguments = recipe.get("cli_args")
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError("Recipe cli_args must be a list of strings.")
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER,
                        help="Additional training arguments after -- override the recipe.")
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    command = [sys.executable, str(ROOT / "experiments/moe_gate_policy_smoke.py"),
               *recipe_arguments(args.config), *overrides]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
