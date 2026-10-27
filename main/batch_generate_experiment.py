#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-generate the structures listed in main/experiment.csv.

What it does
------------
1. Reads experiment.csv (ID, 化学式, base[, 推荐化学式]).
2. Plans one or more single-substitution stages for every target:
   parent -> intermediate -> ... -> target.
   Each stage is a valid host -> dopant substitution with Li as the
   charge-compensating element.
3. Calls generate_stru.py for every stage with:
       --n-candidates 200
       --n-relax 10
       --fmax 0.05 --relax-steps 1000 --relax-optimizer FIRE
       --device auto
   The final stage is written to output/experiment/<target formula>/.
   Intermediate stages are written under output/experiment/_stages/.
4. Writes a real-time log and a machine-readable progress.csv.

The log is intentionally built from the same lines that generate_stru.py
prints (and that are visible in main/print-out):
    [BaseStru]       base structure source
    [Stoichiometry]  actual substitution / composition
    [Candidates]     number of generated unique candidates
    [Screen]         single-point progress (done/total)
    [Relax]          selected candidates, per-candidate convergence,
                     relaxation batch progress
    summary.json     final counts and best energy (read after each stage)

Typical use
-----------
    # inspect the planned stages without running anything
    /home/ziyiguo/project/mattergen/.venv/bin/python \
        batch_generate_experiment.py --dry-run

    # run all planned tasks
    /home/ziyiguo/project/mattergen/.venv/bin/python \
        batch_generate_experiment.py

    # run selected tasks, resuming previous checkpoints
    /home/ziyiguo/project/mattergen/.venv/bin/python \
        batch_generate_experiment.py --only exp_001 exp_011 --resume

    # follow the log in another terminal
    tail -f logs/experiment_*.log

Notes
-----
* The planner uses the "base" column as a family label.  A family can have
  several concrete parents (for example LPSC -> Li6PS5Cl, Li5.4PS4.4Cl1.6,
  Li5.5AsS4.5Br1.5, ...).  It prefers a parent that needs fewer stages and
  that already has a local CIF.
* A local parent CIF is looked up in basestru/, output/, output_local/,
  output/experiment/ and output/experiment/_stages/.  If none is found,
  generate_stru.py is allowed to fall back to Materials Project unless
  --no-mp is given.
* Some compositions cannot be reached by host -> dopant + Li compensation
  (for example the complex oxyhalides exp_073-exp_076, or an unbalanced
  target such as exp_056).  Those tasks are reported as SKIPPED with the
  reason instead of being silently changed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE = SCRIPT_DIR / "generate_stru.py"
DEFAULT_CSV = SCRIPT_DIR / "experiment.csv"
BASESTRU_DIR = SCRIPT_DIR / "basestru"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "experiment"
LOG_DIR = SCRIPT_DIR / "logs"

MODEL_PATH = Path(
    os.environ.get(
        "MATTERSIM_MODEL",
        str(SCRIPT_DIR / "models" / "mattersim-v1.0.0-1M.pth"),
    )
)

N_CANDIDATES = 200
N_RELAX = 10
FMAX = 0.05
RELAX_STEPS = 1000
RELAX_OPTIMIZER = "FIRE"
DEVICE = "auto"

# Parent candidates for each family label.  The planner deduplicates these by
# composition and tries them in this order.
PARENT_LIBRARY: Dict[str, List[str]] = {
    "LPSC": [
        "Li6PS5Cl",
        "Li5.4PS4.4Cl1.6",
        "Li5.4PS4.4Cl1Br0.6",
        "Li5.5PS4.5Cl1.5",
        "Li5.5AsS4.5Br1.5",
        "Li6AsS5I",
        "Li6SbS5I",
        "Li6.7Sb0.3Si0.7S5I",
        "Li6.7Si0.7Sb0.3S5I",
    ],
    "LPS": [
        "Li3PS4",
        "Li3AsS4",
        "Li4SnS4",
    ],
    "LGePS": [
        "Li10GeP2S12",
        "Li10SnP2S12",
        "Li10SiP2S12",
        "Li9.54Si1.74P1.44S11.7Br0.3",
        "Li9.54Si1.74P1.44S11.1Br0.3O0.6",
        "Li9.54Si1.566Sn0.174P1.44S11.1Br0.3O0.6",
    ],
    "LiZrCl": [
        "Li2ZrCl6",
    ],
    "LiInCl": [
        "Li3InCl6",
    ],
    "LiYCl": [
        "Li3YCl6",
        "Li3YBr6",
    ],
}

# When True, a task may only start from a base that already has a local CIF, or
# from another task in the same CSV that will be generated first.  Used by the
# LPSC-only wrapper to keep everything derivable from the local LPSC bases.
REQUIRE_LOCAL_BASE = False

# Stable parents that may be used as a "target == parent" task even when they
# are not another row in experiment.csv.
KNOWN_BASES = {
    "Li6PS5Cl",
    "Li3PS4",
    "Li3AsS4",
    "Li4SnS4",
    "Li10GeP2S12",
    "Li10SnP2S12",
    "Li10SiP2S12",
    "Li2ZrCl6",
    "Li3InCl6",
    "Li3YCl6",
    "Li3YBr6",
    # Frequently used doped parents in this project.  Treating them as bases
    # lets the planner stop at the parent instead of building a circular
    # parent -> target -> parent chain.
    "Li5.5AsS4.5Br1.5",
    "Li6.7Si0.7Sb0.3S5I",
    "Li9.54Si1.74P1.44S11.7Br0.3",
    "Li9.54Si1.74P1.44S11.1Br0.3O0.6",
}

# Oxidation states used only for planning / charge-balanced intermediate Li.
BASE_OX = {
    "Li": 1,
    "P": 5,
    "As": 5,
    "Sb": 5,
    "Si": 4,
    "Sn": 4,
    "Ge": 4,
    "Zr": 4,
    "Al": 3,
    "Fe": 3,
    "In": 3,
    "Y": 3,
    "Zn": 2,
    "Ca": 2,
    "W": 5,
    "S": -2,
    "O": -2,
    "Se": -2,
    "Cl": -1,
    "Br": -1,
    "I": -1,
    "F": -1,
    "H": 1,
}

AMBIGUOUS_OX = {
    "S": (-2, 6),
    "W": (5, 6, 4),
    "As": (5, 3),
    "Sb": (5, 3),
    "Sn": (4, 2),
    "Fe": (3, 2),
}

CATIONS = {
    "Li", "P", "As", "Sb", "Si", "Sn", "Ge", "Zr",
    "Al", "Fe", "In", "Y", "Zn", "Ca", "W",
}
ANIONS = {"S", "O", "Se", "Cl", "Br", "I", "F"}

FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
SCREEN_RE = re.compile(
    r"^\[Screen\]\s+(\d+)/(\d+)\s+elapsed=([0-9.]+)\s+min\s+ETA=([0-9.]+)"
)
RELAX_SELECT_RE = re.compile(r"^\[Relax\]\s+筛选前\s+(\d+)\s+名候选")
RELAX_CAND_RE = re.compile(
    r"^\[Relax\]\s+candidate=(\d+)\s+converged=(\w+)\s+"
    r"steps=(\d+)\s+energy_per_atom=([-+0-9.eE]+)(?:\s+error=(.*))?"
)
RELAX_PROGRESS_RE = re.compile(
    r"^\[Relax\]\s+(\d+)/(\d+)\s+batch_seconds=([0-9.]+)"
)


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def _expand_parentheses(match: re.Match) -> str:
    inner = match.group(1)
    multiplier = float(match.group(2) or 1.0)
    return "".join(
        f"{element}{float(number or 1.0) * multiplier:g}"
        for element, number in FORMULA_TOKEN.findall(inner)
    )


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a formula with decimal subscripts and simple parentheses."""
    text = str(formula).replace(" ", "")
    while "(" in text:
        expanded = re.sub(
            r"\(([^()]*)\)([0-9]*\.?[0-9]*)",
            _expand_parentheses,
            text,
        )
        if expanded == text:
            raise ValueError(f"无法解析化学式: {formula}")
        text = expanded

    counts: Dict[str, float] = defaultdict(float)
    for element, number in FORMULA_TOKEN.findall(text):
        counts[element] += float(number) if number else 1.0
    return dict(counts)


def format_formula(counts: Dict[str, float]) -> str:
    parts: List[str] = []
    for element in sorted(counts, key=lambda item: (item != "Li", item)):
        amount = counts[element]
        if abs(amount) < 1e-9:
            continue
        if abs(amount - round(amount)) < 1e-6:
            text = str(int(round(amount)))
        else:
            text = f"{amount:.4f}".rstrip("0").rstrip(".")
        parts.append(element if text == "1" else f"{element}{text}")
    return "".join(parts)


def formula_key(formula: str) -> Tuple[Tuple[str, float], ...]:
    counts = parse_formula(formula)
    return tuple(sorted((element, round(amount, 4)) for element, amount in counts.items()))


def is_known_base(formula: str) -> bool:
    return formula_key(formula) in {formula_key(item) for item in KNOWN_BASES}


def normalize_formula_key(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text)).lower()


def formula_dirname(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.()+\-]+", "_", str(formula).strip())


def charge(counts: Dict[str, float], oxidation: Dict[str, int]) -> float:
    return sum(oxidation.get(element, 0.0) * amount for element, amount in counts.items())


def find_valence_model(*formulas: str) -> Optional[Dict[str, int]]:
    """Find a consistent oxidation-state assignment for all formulas."""
    elements = set()
    for formula in formulas:
        elements.update(parse_formula(formula))
    if not elements.issubset(BASE_OX):
        return None

    ambiguous = sorted(element for element in elements if element in AMBIGUOUS_OX)
    for values in itertools.product(*(AMBIGUOUS_OX[element] for element in ambiguous)):
        oxidation = dict(BASE_OX)
        oxidation.update(dict(zip(ambiguous, values)))
        if all(abs(charge(parse_formula(formula), oxidation)) < 1e-4 for formula in formulas):
            return oxidation
    return None


def charge_balanced_li(counts: Dict[str, float], oxidation: Dict[str, int]) -> float:
    non_li_charge = sum(
        oxidation.get(element, 0.0) * amount
        for element, amount in counts.items()
        if element != "Li"
    )
    return -non_li_charge / oxidation["Li"]


def same_site_class(host: str, dopant: str) -> bool:
    return (host in CATIONS and dopant in CATIONS) or (
        host in ANIONS and dopant in ANIONS
    )


def unique_formulas(formulas: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for formula in formulas:
        try:
            key = formula_key(formula)
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            result.append(formula)
    return result


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    base_formula: str
    target_formula: str
    host: str
    dopant: str
    amount: float
    stage_charge: float = 0.0
    base_only: bool = False


@dataclass
class Task:
    index: int
    task_id: str
    formula: str
    label: str
    parent: Optional[str] = None
    stages: List[Stage] = field(default_factory=list)
    status: str = "pending"
    error: str = ""
    stage_results: List[Dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    candidate_count: int = 0
    screened_count: int = 0
    selected_for_relaxation: int = 0
    relaxed_count: int = 0
    converged_count: int = 0
    best_energy_per_atom_ev: Optional[float] = None
    best_converged: Optional[bool] = None
    output_dir: Optional[Path] = None

    def stage_details(self) -> str:
        if not self.stages:
            return ""
        details = []
        for stage in self.stages:
            if stage.base_only:
                details.append("base-only")
            else:
                details.append(
                    f"{stage.host}->{stage.dopant} {stage.amount:g}"
                    f" ({stage.base_formula} -> {stage.target_formula})"
                )
        return "; ".join(details)


def plan_path(
    parent_counts: Dict[str, float],
    target_counts: Dict[str, float],
    oxidation: Dict[str, int],
    max_depth: int = 4,
) -> Optional[List[Tuple[str, str, float, str, float]]]:
    """Return [(host, dopant, amount, intermediate_formula, charge), ...]."""

    def recurse(
        state: Dict[str, float],
        depth: int,
        path: List[Tuple[str, str, float, str, float]],
    ) -> Optional[List[Tuple[str, str, float, str, float]]]:
        elements = set(state) | set(target_counts)
        if all(
            abs(state.get(element, 0.0) - target_counts.get(element, 0.0)) < 1e-6
            for element in elements
            if element != "Li"
        ):
            return path
        if depth >= max_depth:
            return None

        hosts = [
            element
            for element in elements
            if element != "Li"
            and state.get(element, 0.0) > target_counts.get(element, 0.0) + 1e-6
        ]
        dopants = [
            element
            for element in elements
            if element != "Li"
            and state.get(element, 0.0) < target_counts.get(element, 0.0) - 1e-6
        ]

        pairs: List[Tuple[float, str, str]] = []
        for host in hosts:
            for dopant in dopants:
                if not same_site_class(host, dopant):
                    continue
                amount = min(
                    state.get(host, 0.0) - target_counts.get(host, 0.0),
                    target_counts.get(dopant, 0.0) - state.get(dopant, 0.0),
                )
                if amount > 1e-6:
                    pairs.append((amount, host, dopant))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        for amount, host, dopant in pairs:
            next_state = dict(state)
            next_state[host] = next_state.get(host, 0.0) - amount
            next_state[dopant] = next_state.get(dopant, 0.0) + amount
            next_li = charge_balanced_li(next_state, oxidation)
            if next_li < -1e-6:
                continue
            next_state["Li"] = next_li
            formula = format_formula(next_state)
            result = recurse(
                next_state,
                depth + 1,
                path
                + [
                    (
                        host,
                        dopant,
                        amount,
                        formula,
                        charge(next_state, oxidation),
                    )
                ],
            )
            if result is not None:
                return result
        return None

    return recurse(dict(parent_counts), 0, [])


def choose_dummy_pair(formula: str) -> Optional[Tuple[str, str]]:
    counts = parse_formula(formula)
    host = next(
        (
            element
            for element in counts
            if element != "Li" and counts[element] > 1e-6
        ),
        None,
    )
    if host is None:
        return None
    pool = ANIONS if host in ANIONS else CATIONS
    for dopant in sorted(pool):
        if dopant != "Li" and dopant != host and dopant not in counts:
            return host, dopant
    return None


def plan_task(task: Task, known_targets: set) -> Tuple[Optional[str], List[Stage], str]:
    """Return (parent, stages, error)."""
    target_counts = parse_formula(task.formula)
    parents = unique_formulas(PARENT_LIBRARY.get(task.label, []))
    if not parents:
        return None, [], f"未知或未配置的 base 标签: {task.label}"

    best_score = None
    best_parent = None
    best_stages: List[Stage] = []
    reasons: List[str] = []

    for parent in parents:
        parent_counts = parse_formula(parent)
        oxidation = find_valence_model(parent, task.formula)
        if oxidation is None:
            reasons.append(f"{parent}: 价态模型无法同时平衡 parent/target")
            continue
        if abs(charge(target_counts, oxidation)) > 1e-4:
            reasons.append(f"{parent}: target 电荷不守恒")
            continue

        same_composition = all(
            abs(parent_counts.get(element, 0.0) - target_counts.get(element, 0.0)) < 1e-6
            for element in set(parent_counts) | set(target_counts)
        )
        if same_composition:
            allowed = find_base_path(parent) is not None
            if not allowed and not REQUIRE_LOCAL_BASE:
                allowed = (
                    formula_key(parent) in known_targets
                    or is_known_base(parent)
                )
            if not allowed:
                reasons.append(f"{parent}: target 与 parent 相同但无本地/CIF 来源")
                continue
            dummy = choose_dummy_pair(parent)
            if dummy is None:
                reasons.append(f"{parent}: 无法为 base-only 任务选择占位取代元素")
                continue
            stage = Stage(
                base_formula=parent,
                target_formula=task.formula,
                host=dummy[0],
                dopant=dummy[1],
                amount=0.0,
                stage_charge=0.0,
                base_only=True,
            )
            local = 2 if find_base_path(parent) is not None else 0
            if local or is_known_base(parent):
                # A real base (local CIF or a known stable parent) is ideal.
                score = (0, -local, 0.0)
            else:
                # The target is only "its own parent" because it appears as
                # another row.  Prefer a real substitution path if one exists,
                # but keep this as a fallback.
                score = (1.5, 0.0, 0.0)
            if best_score is None or score < best_score:
                best_score = score
                best_parent = parent
                best_stages = [stage]
            continue

        path = plan_path(parent_counts, target_counts, oxidation)
        if path is None:
            reasons.append(f"{parent}: 找不到 host->dopant 取代路径")
            continue

        last_counts = parse_formula(path[-1][3])
        if abs(last_counts.get("Li", 0.0) - target_counts.get("Li", 0.0)) > 1e-3:
            reasons.append(f"{parent}: 最终 Li 与 target 不一致")
            continue

        stages: List[Stage] = []
        base_formula = parent
        for index, (host, dopant, amount, intermediate, stage_charge) in enumerate(path):
            stage_target = task.formula if index == len(path) - 1 else intermediate
            stages.append(
                Stage(
                    base_formula=base_formula,
                    target_formula=stage_target,
                    host=host,
                    dopant=dopant,
                    amount=amount,
                    stage_charge=stage_charge,
                )
            )
            base_formula = stage_target

        local = 2 if find_base_path(parent) is not None else 0
        if REQUIRE_LOCAL_BASE:
            available = 0 if (local or formula_key(parent) in known_targets) else 1
            score = (
                available,
                len(stages),
                -local,
                sum(abs(stage.stage_charge) for stage in stages),
            )
        else:
            score = (
                len(stages),
                -local,
                sum(abs(stage.stage_charge) for stage in stages),
            )
        if best_score is None or score < best_score:
            best_score = score
            best_parent = parent
            best_stages = stages

    if best_parent is None:
        return None, [], "; ".join(reasons) or "没有可用 parent"
    return best_parent, best_stages, ""


# ---------------------------------------------------------------------------
# Base lookup
# ---------------------------------------------------------------------------

def find_base_path(formula: str) -> Optional[Path]:
    key = normalize_formula_key(formula)
    name = formula_dirname(formula)
    search_dirs = [
        BASESTRU_DIR,
        SCRIPT_DIR / "output",
        SCRIPT_DIR / "output_local",
        OUTPUT_ROOT,
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        direct = [
            directory / f"{formula}.cif",
            directory / name / "best_structure.cif",
        ]
        for path in direct:
            if path.is_file():
                return path
        for path in directory.glob("*.cif"):
            if normalize_formula_key(path.stem) == key:
                return path
        for path in directory.glob("*/best_structure.cif"):
            if normalize_formula_key(path.parent.name) == key:
                return path

    stages_root = OUTPUT_ROOT / "_stages"
    if stages_root.exists():
        for path in stages_root.glob("*/*/best_structure.cif"):
            stage_name = re.sub(r"^\d+_", "", path.parent.name)
            if normalize_formula_key(stage_name) == key:
                return path
    return None


# ---------------------------------------------------------------------------
# Logging / progress
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.log_path.parent / f"{self.log_path.stem}_raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def write_progress(path: Path, tasks: Sequence[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    fields = [
        "task_index",
        "id",
        "formula",
        "base",
        "parent",
        "stages",
        "status",
        "candidate_count",
        "screened_count",
        "selected_for_relaxation",
        "relaxed_count",
        "converged_count",
        "best_energy_per_atom_ev",
        "best_converged",
        "output_dir",
        "error",
        "elapsed_s",
    ]
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "task_index": task.index,
                    "id": task.task_id,
                    "formula": task.formula,
                    "base": task.label,
                    "parent": task.parent or "",
                    "stages": len(task.stages),
                    "status": task.status,
                    "candidate_count": task.candidate_count,
                    "screened_count": task.screened_count,
                    "selected_for_relaxation": task.selected_for_relaxation,
                    "relaxed_count": task.relaxed_count,
                    "converged_count": task.converged_count,
                    "best_energy_per_atom_ev": (
                        "" if task.best_energy_per_atom_ev is None
                        else f"{task.best_energy_per_atom_ev:.6f}"
                    ),
                    "best_converged": "" if task.best_converged is None else task.best_converged,
                    "output_dir": "" if task.output_dir is None else str(task.output_dir),
                    "error": task.error,
                    "elapsed_s": f"{task.elapsed_s:.1f}",
                }
            )
    temporary.replace(path)


def write_plan(path: Path, tasks: Sequence[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_index",
        "id",
        "formula",
        "base",
        "parent",
        "stages",
        "stage_details",
        "status",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "task_index": task.index,
                    "id": task.task_id,
                    "formula": task.formula,
                    "base": task.label,
                    "parent": task.parent or "",
                    "stages": len(task.stages),
                    "stage_details": task.stage_details(),
                    "status": task.status,
                    "error": task.error,
                }
            )


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def build_command(
    task: Task,
    stage: Stage,
    stage_index: int,
    output_dir: Path,
    base_path: Optional[Path],
    args: argparse.Namespace,
) -> List[str]:
    base_only = bool(stage.base_only)
    n_candidates = 1 if base_only else args.n_candidates
    n_relax = 1 if base_only else args.n_relax

    command = [
        sys.executable,
        str(GENERATE),
        "--base-formula",
        stage.base_formula,
        "--target-formula",
        stage.target_formula,
        "--host",
        stage.host,
        "--dopant",
        stage.dopant,
        "--compensating-element",
        "Li",
        "--output",
        str(output_dir),
        "--n-candidates",
        str(n_candidates),
        "--n-relax",
        str(n_relax),
        "--fmax",
        str(args.fmax),
        "--relax-steps",
        str(args.relax_steps),
        "--relax-optimizer",
        args.relax_optimizer,
        "--device",
        args.device,
    ]
    if base_path is not None:
        command += ["--base-structure", str(base_path)]
    if MODEL_PATH.is_file():
        command += ["--model", str(MODEL_PATH)]
    if stage_index > 0:
        command += ["--scaling", "1,1,1"]
    if not base_only and stage.host == "S" and stage.dopant in {"O", "Se", "Cl", "Br", "I", "F"}:
        command += [
            "--host-site-filter",
            "free-anion",
            "--host-framework-element",
            "P,Si,Ge,Sn,As,Sb",
        ]
    if not base_only and stage.amount < 0.05:
        command += ["--max-supercell-atoms", str(max(args.max_supercell_atoms, 1500))]
    command += ["--substitution-sampler", "random"]
    command += ["DUMMY"]
    return command


def parse_child_line(line: str, logger: RunLogger) -> None:
    line = line.rstrip("\n")
    if line.startswith("[BaseStru]"):
        logger.log(f"  base: {line}")
        return
    if line.startswith("[Candidates]"):
        logger.log(f"  gen: {line}")
        return
    if line.startswith("[Stoichiometry]"):
        logger.log(f"  stoich: {line}")
        return
    if line.startswith("[Substitution]"):
        logger.log(f"  sub: {line}")
        return

    match = SCREEN_RE.match(line)
    if match:
        logger.log(
            f"  screen: {match.group(1)}/{match.group(2)} "
            f"elapsed={match.group(3)} min ETA={match.group(4)} min"
        )
        return

    match = RELAX_SELECT_RE.match(line)
    if match:
        logger.log(f"  relax select: 前 {match.group(1)} 名候选")
        return

    match = RELAX_CAND_RE.match(line)
    if match:
        error = (match.group(5) or "").strip()
        suffix = f" error={error}" if error else ""
        logger.log(
            f"  relax candidate={match.group(1)} "
            f"converged={match.group(2)} steps={match.group(3)} "
            f"E/atom={match.group(4)}{suffix}"
        )
        return

    match = RELAX_PROGRESS_RE.match(line)
    if match:
        logger.log(
            f"  relax progress: {match.group(1)}/{match.group(2)} "
            f"(batch {match.group(3)} s)"
        )
        return

    if any(token in line for token in ("Traceback", "Error", "错误", "失败", "⚠")):
        logger.log(f"  !! {line}")


def read_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_base_only_stage(
    task: Task,
    stage: Stage,
    output_dir: Path,
    base_path: Optional[Path],
    logger: RunLogger,
) -> Dict:
    """A base-only task just publishes the existing base structure.

    generate_stru.py cannot express "zero substitutions", so calling it with
    amount=0 fails.  The base CIF is already the requested composition, so copy
    it into the task output directory and write the summary expected by the
    rest of the pipeline.
    """
    if base_path is None:
        raise RuntimeError(f"base-only 任务缺少 base 结构: {stage.base_formula}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_path, output_dir / "base_input.cif")
    shutil.copy2(base_path, output_dir / "best_structure.cif")
    summary = {
        "base_only": True,
        "base_formula": stage.base_formula,
        "base_structure": str(base_path),
        "candidate_count": 1,
        "screened_count": 0,
        "selected_for_relaxation": 0,
        "relaxed_count": 0,
        "converged_count": 0,
        "best_energy_per_atom_ev": None,
        "best_converged": None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.log(f"  base-only: copied {base_path} -> {output_dir / 'best_structure.cif'}")
    return summary


def run_stage(
    task: Task,
    stage: Stage,
    stage_index: int,
    output_dir: Path,
    base_path: Optional[Path],
    args: argparse.Namespace,
    logger: RunLogger,
) -> Dict:
    if stage.base_only:
        return run_base_only_stage(task, stage, output_dir, base_path, logger)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = logger.raw_dir / f"{task.task_id}_stage{stage_index + 1:02d}.log"
    command = build_command(task, stage, stage_index, output_dir, base_path, args)

    logger.log(
        f"  stage {stage_index + 1}/{len(task.stages)}: "
        f"{'base-only' if stage.base_only else f'{stage.host}->{stage.dopant} {stage.amount:g}'} "
        f"base={stage.base_formula} -> target={stage.target_formula}"
    )
    logger.log(f"  output: {output_dir}")
    logger.log(f"  command: {' '.join(command)}")

    started = time.time()
    with open(raw_path, "w", encoding="utf-8") as raw:
        process = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            raw.write(line)
            raw.flush()
            parse_child_line(line, logger)
        returncode = process.wait()

    elapsed = time.time() - started
    if returncode != 0:
        raise RuntimeError(
            f"generate_stru.py 退出码 {returncode}，原始输出见 {raw_path}"
        )

    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"没有生成 summary.json: {summary_path}")
    summary = read_json(summary_path)
    logger.log(
        f"  stage done in {elapsed:.1f}s: "
        f"candidates={summary.get('candidate_count', 0)} "
        f"screened={summary.get('screened_count', 0)} "
        f"selected={summary.get('selected_for_relaxation', 0)} "
        f"relaxed={summary.get('relaxed_count', 0)} "
        f"converged={summary.get('converged_count', 0)} "
        f"best_E/atom={summary.get('best_energy_per_atom_ev')}"
    )
    if summary.get("relaxed_count", 0) and not summary.get("converged_count", 0):
        logger.log("  WARNING: 本阶段没有任何结构达到力收敛标准")
    return summary


def task_output_dir(task: Task) -> Path:
    return OUTPUT_ROOT / formula_dirname(task.formula)


def stage_output_dir(task: Task, stage_index: int) -> Path:
    if stage_index == len(task.stages) - 1:
        return task_output_dir(task)
    return (
        OUTPUT_ROOT
        / "_stages"
        / task.task_id
        / f"{stage_index + 1:02d}_{formula_dirname(task.stages[stage_index].target_formula)}"
    )


def run_task(task: Task, args: argparse.Namespace, logger: RunLogger) -> None:
    task.output_dir = task_output_dir(task)
    task.status = "running"
    started = time.time()
    previous_best: Optional[Path] = None

    try:
        for stage_index, stage in enumerate(task.stages):
            output_dir = stage_output_dir(task, stage_index)
            if stage_index == 0:
                base_path = find_base_path(stage.base_formula)
            else:
                base_path = previous_best
            if base_path is None and args.no_mp:
                raise RuntimeError(
                    f"找不到本地 base 结构 {stage.base_formula}，且指定了 --no-mp"
                )
            if base_path is None:
                logger.log(
                    f"  base {stage.base_formula} 本地不存在，将回退 Materials Project"
                )

            summary = run_stage(
                task,
                stage,
                stage_index,
                output_dir,
                base_path,
                args,
                logger,
            )
            task.stage_results.append(summary)

            if stage_index < len(task.stages) - 1:
                previous_best = output_dir / "best_structure.cif"
                if not previous_best.is_file():
                    raise RuntimeError(
                        f"中间阶段没有 best_structure.cif: {previous_best}"
                    )

        final = task.stage_results[-1] if task.stage_results else {}
        task.candidate_count = int(final.get("candidate_count", 0))
        task.screened_count = int(final.get("screened_count", 0))
        task.selected_for_relaxation = int(final.get("selected_for_relaxation", 0))
        task.relaxed_count = int(final.get("relaxed_count", 0))
        task.converged_count = int(final.get("converged_count", 0))
        task.best_energy_per_atom_ev = final.get("best_energy_per_atom_ev")
        task.best_converged = final.get("best_converged")
        task.status = "base_done" if any(stage.base_only for stage in task.stages) else "done"
        task.elapsed_s = time.time() - started
        logger.log(
            f"  TASK DONE {task.task_id} {task.formula}: "
            f"candidates={task.candidate_count} screened={task.screened_count} "
            f"relaxed={task.relaxed_count} converged={task.converged_count} "
            f"best_E/atom={task.best_energy_per_atom_ev} "
            f"output={task.output_dir}"
        )
        if task.relaxed_count and task.converged_count == 0:
            logger.log(f"  WARNING {task.task_id}: 任务完成，但没有收敛结构")
    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        task.elapsed_s = time.time() - started
        logger.log(f"  TASK FAILED {task.task_id}: {exc}")
        logger.log("  traceback:")
        for line in traceback.format_exc().rstrip().splitlines():
            logger.log(f"    {line}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量生成 experiment.csv 中的结构",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="输入 TSV/CSV")
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="最终结构输出根目录",
    )
    parser.add_argument("--log", default=None, help="实时日志文件路径")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--n-relax", type=int, default=N_RELAX)
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--relax-steps", type=int, default=RELAX_STEPS)
    parser.add_argument(
        "--relax-optimizer",
        default=RELAX_OPTIMIZER,
        choices=["FIRE", "BFGS"],
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--max-supercell-atoms",
        type=int,
        default=500,
        help="超胞原子数预算；小掺杂量会自动提高到至少 1500",
    )
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="禁止回退 Materials Project；本地没有 base CIF 时跳过任务",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只规划并打印/写出 plan.csv，不调用 generate_stru.py",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出任务和规划，不运行",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="读取 progress.csv，跳过已完成任务",
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="配合 --resume 重试之前跳过的任务",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="只运行指定 ID 或化学式",
    )
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 个任务")
    parser.add_argument(
        "--use-recommended",
        action="store_true",
        help="如果存在推荐化学式列且非空，优先使用推荐化学式",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留化学式完全相同的重复行；默认只运行第一次出现的那一行",
    )
    return parser


def load_tasks(path: Path, use_recommended: bool = False) -> List[Task]:
    tasks: List[Task] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"空文件: {path}")
        # Normalize the possible leading space in the "base" header.
        field_map = {name.strip(): name for name in reader.fieldnames}
        id_field = field_map.get("ID")
        formula_field = field_map.get("化学式")
        base_field = field_map.get("base")
        recommended_field = field_map.get("推荐化学式")
        if not id_field or not formula_field:
            raise ValueError(f"缺少 ID/化学式 列: {reader.fieldnames}")

        for index, row in enumerate(reader, start=1):
            task_id = (row.get(id_field) or "").strip()
            formula = (row.get(formula_field) or "").strip()
            if not task_id or not formula:
                continue
            if use_recommended and recommended_field:
                recommended = (row.get(recommended_field) or "").strip()
                if recommended:
                    formula = recommended
            label = (row.get(base_field) or "").strip() if base_field else ""
            tasks.append(
                Task(
                    index=index,
                    task_id=task_id,
                    formula=formula,
                    label=label,
                )
            )
    return tasks


def filter_tasks(
    tasks: Sequence[Task],
    only: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[Task]:
    selected = list(tasks)
    if only:
        wanted = {item.lower() for item in only}
        selected = [
            task
            for task in selected
            if task.task_id.lower() in wanted or task.formula.lower() in wanted
        ]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def mark_duplicate_formulas(tasks: Sequence[Task]) -> Sequence[Task]:
    """Keep the first task for each composition; mark later ones as duplicates."""
    seen: Dict[Tuple[Tuple[str, float], ...], Task] = {}
    for task in tasks:
        key = formula_key(task.formula)
        if key in seen:
            task.status = "skipped"
            task.error = f"重复化学式，与 {seen[key].task_id} 相同（输出目录会冲突）"
        else:
            seen[key] = task
    return tasks


def order_tasks(tasks: Sequence[Task]) -> List[Task]:
    """Run tasks whose target is another task's parent first when possible."""
    by_formula = {}
    for task in tasks:
        by_formula.setdefault(formula_key(task.formula), task)
    by_id = {task.task_id: task for task in tasks}
    dependencies: Dict[str, List[str]] = {task.task_id: [] for task in tasks}
    for task in tasks:
        if not task.parent:
            continue
        parent_task = by_formula.get(formula_key(task.parent))
        if parent_task is not None and parent_task is not task:
            dependencies[task.task_id].append(parent_task.task_id)

    ordered: List[Task] = []
    visited = set()
    visiting = set()

    def visit(task: Task) -> None:
        if task.task_id in visited:
            return
        if task.task_id in visiting:
            return
        visiting.add(task.task_id)
        for dependency_id in dependencies[task.task_id]:
            visit(by_id[dependency_id])
        visiting.remove(task.task_id)
        visited.add(task.task_id)
        ordered.append(task)

    for task in tasks:
        visit(task)
    return ordered


def apply_previous_progress(
    tasks: Sequence[Task],
    progress_path: Path,
    retry_skipped: bool,
) -> None:
    if not progress_path.is_file():
        return
    with open(progress_path, newline="", encoding="utf-8") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    for task in tasks:
        previous = rows.get(task.task_id)
        if not previous:
            continue
        status = previous.get("status", "")
        if status in {"done", "base_done"} and task.output_dir and task.output_dir.exists():
            task.status = "done_resumed"
        elif status == "skipped" and not retry_skipped:
            task.status = "skipped_resumed"
            task.error = previous.get("error", "")


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    global OUTPUT_ROOT
    OUTPUT_ROOT = output_root

    tasks = load_tasks(csv_path, use_recommended=args.use_recommended)
    tasks = filter_tasks(tasks, args.only, args.limit)
    if not tasks:
        print("没有匹配的任务", file=sys.stderr)
        return 2

    if not args.keep_duplicates:
        mark_duplicate_formulas(tasks)

    known_targets = {formula_key(task.formula) for task in tasks}
    for task in tasks:
        if task.status == "skipped" and task.error.startswith("重复化学式"):
            task.output_dir = task_output_dir(task)
            continue
        parent, stages, error = plan_task(task, known_targets)
        task.parent = parent
        task.stages = stages
        task.output_dir = task_output_dir(task)
        if stages:
            task.status = "planned"
        else:
            task.status = "skipped"
            task.error = error

    tasks = order_tasks(tasks)

    if args.log:
        log_path = Path(args.log).expanduser()
    else:
        log_path = LOG_DIR / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = RunLogger(log_path)

    planned = sum(1 for task in tasks if task.stages)
    skipped = len(tasks) - planned
    logger.log("=" * 78)
    logger.log("Batch experiment generation")
    logger.log(f"csv:         {csv_path}")
    logger.log(f"tasks:       {len(tasks)} (planned={planned}, skipped={skipped})")
    logger.log(f"output root: {OUTPUT_ROOT}")
    logger.log(
        "settings:    "
        f"n_candidates={args.n_candidates} n_relax={args.n_relax} "
        f"fmax={args.fmax} relax_steps={args.relax_steps} "
        f"optimizer={args.relax_optimizer} device={args.device}"
    )
    logger.log(f"log:         {log_path}")
    logger.log("=" * 78)

    if args.dry_run or args.list:
        for position, task in enumerate(tasks, start=1):
            if task.stages:
                logger.log(
                    f"[{position:03d}/{len(tasks)}] {task.task_id} {task.formula} "
                    f"base={task.label} parent={task.parent} "
                    f"stages={len(task.stages)} :: {task.stage_details()}"
                )
            else:
                logger.log(
                    f"[{position:03d}/{len(tasks)}] {task.task_id} {task.formula} "
                    f"base={task.label} SKIPPED: {task.error}"
                )
        plan_path = OUTPUT_ROOT / "plan.csv"
        write_plan(plan_path, tasks)
        logger.log(f"plan written to {plan_path}")
        logger.close()
        return 0

    progress_path = OUTPUT_ROOT / "progress.csv"
    if args.resume:
        apply_previous_progress(tasks, progress_path, args.retry_skipped)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_progress(progress_path, tasks)

    for position, task in enumerate(tasks, start=1):
        if task.status in {"done_resumed", "skipped_resumed"}:
            logger.log(
                f"[{position:03d}/{len(tasks)}] {task.task_id} "
                f"{task.formula} -> {task.status}"
            )
            continue
        if not task.stages:
            logger.log(
                f"[{position:03d}/{len(tasks)}] {task.task_id} "
                f"{task.formula} -> SKIPPED: {task.error}"
            )
            write_progress(progress_path, tasks)
            continue

        logger.log("")
        logger.log(
            f"[{position:03d}/{len(tasks)}] {task.task_id} {task.formula} "
            f"base={task.label} parent={task.parent} stages={len(task.stages)}"
        )
        logger.log(f"  plan: {task.stage_details()}")
        run_task(task, args, logger)
        write_progress(progress_path, tasks)

    done = sum(1 for task in tasks if task.status in {"done", "base_done", "done_resumed"})
    failed = sum(1 for task in tasks if task.status == "failed")
    no_convergence = sum(
        1
        for task in tasks
        if task.status in {"done", "base_done", "done_resumed"}
        and task.relaxed_count > 0
        and task.converged_count == 0
    )
    logger.log("")
    logger.log("=" * 78)
    logger.log("RUN SUMMARY")
    logger.log(
        f"total={len(tasks)} done={done} failed={failed} "
        f"skipped={sum(1 for task in tasks if task.status.startswith('skipped'))} "
        f"done_without_convergence={no_convergence}"
    )
    logger.log(f"progress: {progress_path}")
    logger.log(f"log:      {log_path}")
    logger.log("=" * 78)
    logger.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
