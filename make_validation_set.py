#!/usr/bin/env python3
"""Sample deception logs into balanced grader folders.

By default this selects 20 turn_*.json files from each configured experiment
and distributes them across validation/grader1, grader2, and grader3 as 7/7/6
per experiment.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


EXPERIMENTS = (
    "qwen27b-qwen235",
    "qwen3.5-2b_qwen3.5-2b",
    "kimi26-kimi26",
)
MONITOR = "qwen3.5-2b"
GRADERS = ("grader1", "grader2", "grader3")
PER_EXPERIMENT = 20


@dataclass(frozen=True)
class Sample:
    experiment: str
    source: Path

    @property
    def run_id(self) -> str:
        return self.source.parent.name

    @property
    def turn_file(self) -> str:
        return self.source.name

    @property
    def output_name(self) -> str:
        return f"{self.experiment}__{self.run_id}__{self.turn_file}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validation/grader* folders with sampled deception logs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation",
        help="Destination folder to create. Default: validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def collect_logs(experiment: str) -> list[Sample]:
    root = Path("deceptionlogs") / experiment / MONITOR
    if not root.exists():
        raise FileNotFoundError(f"Missing deception log directory: {root}")

    samples = [
        Sample(experiment=experiment, source=path)
        for path in sorted(root.glob("*/turn_*.json"))
        if path.is_file()
    ]
    if len(samples) < PER_EXPERIMENT:
        raise RuntimeError(
            f"{experiment} only has {len(samples)} deception logs; "
            f"need {PER_EXPERIMENT}."
        )
    return samples


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    for grader in GRADERS:
        (output_dir / grader).mkdir(parents=True, exist_ok=True)


def assign_to_graders(selected: list[Sample], rng: random.Random) -> dict[str, list[Sample]]:
    shuffled = selected[:]
    rng.shuffle(shuffled)
    return {
        "grader1": shuffled[:7],
        "grader2": shuffled[7:14],
        "grader3": shuffled[14:20],
    }


def copy_samples(output_dir: Path, assignments: dict[str, list[Sample]]) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []

    for grader, samples in assignments.items():
        for sample in samples:
            dest = output_dir / grader / sample.output_name
            shutil.copy2(sample.source, dest)
            manifest_rows.append(
                {
                    "grader": grader,
                    "experiment": sample.experiment,
                    "run_id": sample.run_id,
                    "turn_file": sample.turn_file,
                    "source": str(sample.source),
                    "copied_to": str(dest),
                }
            )

    return manifest_rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("grader", "experiment", "run_id", "turn_file", "source", "copied_to"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    selections_by_experiment = {
        experiment: rng.sample(collect_logs(experiment), PER_EXPERIMENT)
        for experiment in EXPERIMENTS
    }

    prepare_output_dir(output_dir, args.overwrite)

    all_rows: list[dict[str, str]] = []
    for experiment, selected in selections_by_experiment.items():
        all_rows.extend(copy_samples(output_dir, assign_to_graders(selected, rng)))

    write_manifest(output_dir, all_rows)

    print(f"Created {output_dir} with {len(all_rows)} sampled deception logs.")
    for grader in GRADERS:
        rows = [row for row in all_rows if row["grader"] == grader]
        counts = {
            experiment: sum(1 for row in rows if row["experiment"] == experiment)
            for experiment in EXPERIMENTS
        }
        print(f"{grader}: {len(rows)} logs {counts}")


if __name__ == "__main__":
    main()
