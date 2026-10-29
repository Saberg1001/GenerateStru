#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize the per-target results of a generation run.

For every target it reports:
  - whether candidate generation reached the expected count;
  - how many candidates were screened / selected / relaxed;
  - how many relaxations converged (from ranking.csv and summary.json);
  - the best energy and whether the best structure is converged;
  - a verdict plus any flags worth looking at.

It reads, in order of preference:
  <root>/progress.csv            (task status, including failures)
  <root>/<formula>/summary.json  (counts after the run)
  <root>/<formula>/ranking.csv   (per-candidate converged flags)

Usage:
    python summarize_results.py                       # output/lpsc
    python summarize_results.py --root output/experiment
    python summarize_results.py --root output/lpsc --log print-out-summary.log
    python summarize_results.py --root A --root B     # several roots, one file

The log is ASCII-only so it can be viewed/transferred anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_ROOT / "output" / "lpsc"
DEFAULT_LOG = SCRIPT_DIR / "print-out-summary.log"


def read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def ranking_stats(path: Path) -> Dict[str, int]:
    rows = read_csv(path)
    selected = sum(1 for row in rows if as_bool(row.get("selected_for_relaxation")))
    converged = sum(1 for row in rows if as_bool(row.get("converged")))
    max_steps = max((as_int(row.get("relaxation_steps")) for row in rows), default=0)
    errors = sum(1 for row in rows if (row.get("error") or "").strip())
    return {
        "rows": len(rows),
        "selected": selected,
        "converged": converged,
        "max_steps": max_steps,
        "errors": errors,
    }


def summarize_target(
    output_dir: Path,
    expected_candidates: int,
    expected_relax: int,
    progress_row: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    summary = read_json(output_dir / "summary.json") if output_dir.is_dir() else None
    ranks = ranking_stats(output_dir / "ranking.csv") if output_dir.is_dir() else {}

    entry: Dict[str, object] = {
        "id": (progress_row or {}).get("id") or (progress_row or {}).get("task_index", ""),
        "formula": (progress_row or {}).get("formula") or output_dir.name,
        "status": (progress_row or {}).get("status", ""),
        "error": (progress_row or {}).get("error", ""),
        "output_dir": str(output_dir),
        "candidates": 0,
        "screened": 0,
        "selected": 0,
        "relaxed": 0,
        "converged": 0,
        "best_energy": None,
        "best_converged": None,
        "max_steps": ranks.get("max_steps", 0),
        "rank_errors": ranks.get("errors", 0),
        "has_summary": summary is not None,
        "base_only": False,
    }

    if summary:
        entry["base_only"] = bool(summary.get("base_only"))
        entry["candidates"] = as_int(summary.get("candidate_count"))
        entry["screened"] = as_int(summary.get("screened_count"))
        entry["selected"] = as_int(summary.get("selected_for_relaxation"))
        entry["relaxed"] = as_int(summary.get("relaxed_count"))
        entry["converged"] = as_int(summary.get("converged_count"))
        entry["best_energy"] = as_float(summary.get("best_energy_per_atom_ev"))
        entry["best_converged"] = summary.get("best_converged")
    elif progress_row:
        entry["candidates"] = as_int(progress_row.get("candidate_count"))
        entry["screened"] = as_int(progress_row.get("screened_count"))
        entry["selected"] = as_int(progress_row.get("selected_for_relaxation"))
        entry["relaxed"] = as_int(progress_row.get("relaxed_count"))
        entry["converged"] = as_int(progress_row.get("converged_count"))
        entry["best_energy"] = as_float(progress_row.get("best_energy_per_atom_ev"))
        entry["best_converged"] = progress_row.get("best_converged")

    if entry["best_converged"] in ("", None):
        entry["best_converged"] = None

    if not entry["selected"] and ranks.get("selected"):
        entry["selected"] = ranks["selected"]
    if not entry["converged"] and ranks.get("converged"):
        entry["converged"] = ranks["converged"]

    flags: List[str] = []
    if summary and entry["base_only"]:
        verdict = "BASE"
    elif str(entry["status"]) == "failed":
        verdict = "FAILED"
    elif not summary:
        verdict = "NO-OUTPUT"
    else:
        if entry["candidates"] < expected_candidates:
            flags.append("CAND<{}".format(expected_candidates))
        if entry["screened"] < entry["candidates"]:
            flags.append("SCREEN-PARTIAL")
        if entry["relaxed"] < entry["selected"]:
            flags.append("RELAX-PARTIAL")
        if entry["converged"] == 0:
            flags.append("NO-CONVERGE")
        elif entry["best_converged"] is False:
            flags.append("BEST-UNCONV")
        if entry["relaxed"] == 0:
            flags.append("NO-RELAX")

        pipeline_complete = (
            entry["screened"] == entry["candidates"]
            and entry["relaxed"] == entry["selected"]
            and entry["selected"] >= min(expected_relax, entry["candidates"])
        )
        full_pool = entry["candidates"] >= expected_candidates
        if pipeline_complete and entry["converged"] > 0 and entry["best_converged"] is not False:
            verdict = "OK" if full_pool else "OK-LIMITED"
        elif pipeline_complete:
            verdict = "OK-NOCONV"
        else:
            verdict = "PARTIAL"

    entry["flags"] = flags
    entry["verdict"] = verdict
    return entry


def build_stage_entries(
    root: Path, expected_candidates: int, expected_relax: int
) -> List[Dict]:
    stages_root = root / "_stages"
    if not stages_root.is_dir():
        return []
    entries = []
    for summary_path in sorted(stages_root.glob("*/*/summary.json")):
        entry = summarize_target(
            summary_path.parent, expected_candidates, expected_relax
        )
        entry["id"] = summary_path.parent.parent.name
        entry["formula"] = summary_path.parent.name
        entries.append(entry)
    return entries


def build_entries(root: Path, expected_candidates: int, expected_relax: int) -> List[Dict]:
    progress = read_csv(root / "progress.csv")
    if progress:
        entries = []
        for row in progress:
            out = row.get("output_dir") or ""
            if out:
                output_dir = Path(out)
                if not output_dir.is_absolute():
                    output_dir = (SCRIPT_DIR / output_dir).resolve()
            else:
                output_dir = root / row.get("formula", "")
            entries.append(
                summarize_target(
                    output_dir, expected_candidates, expected_relax, row
                )
            )
        return entries

    entries = []
    for summary_path in sorted(root.glob("*/summary.json")):
        entries.append(
            summarize_target(
                summary_path.parent, expected_candidates, expected_relax
            )
        )
    return entries


def format_row(entry: Dict, expected_candidates: int) -> str:
    candidates = entry["candidates"]
    cand = "{}/{}".format(candidates, expected_candidates)
    screened = str(entry["screened"])
    relax = "{}/{}".format(entry["relaxed"], entry["selected"])
    conv = "{}/{}".format(entry["converged"], entry["relaxed"])

    energy = entry["best_energy"]
    energy_text = "{:.4f}".format(energy) if energy is not None else "-"
    best = entry["best_converged"]
    best_text = "-" if best is None else ("yes" if as_bool(best) else "no")

    flags = list(entry["flags"])
    error = (entry["error"] or "").strip()
    if error:
        flags.append(error[:70])

    return (
        "{id:<9} {verdict:<11} {formula:<34} {cand:>9} {screened:>5} "
        "{relax:>8} {conv:>8} {energy_text:>9} {best_text:>4}  {flags}"
    ).format(
        id=str(entry["id"])[:9],
        verdict=entry["verdict"],
        formula=str(entry["formula"])[:34],
        cand=cand,
        screened=screened,
        relax=relax,
        conv=conv,
        energy_text=energy_text,
        best_text=best_text,
        flags=", ".join(flags),
    )


def summarize_root(
    root: Path,
    expected_candidates: int,
    expected_relax: int,
    lines: List[str],
    include_stages: bool = False,
) -> None:
    lines.append("=" * 118)
    lines.append("root: {}".format(root))
    if not root.is_dir():
        lines.append("  (missing)")
        lines.append("")
        return

    entries = build_entries(root, expected_candidates, expected_relax)
    if not entries:
        lines.append("  (no targets found)")
        lines.append("")
        return

    lines.append(
        "targets: {}   expected: candidates={} relax={}".format(
            len(entries), expected_candidates, expected_relax
        )
    )
    lines.append("-" * 118)
    lines.append(
        "{:<9} {:<11} {:<34} {:>9} {:>5} {:>8} {:>8} {:>9} {:>4}  flags".format(
            "id", "verdict", "formula", "cand", "scr", "relax", "conv",
            "E/atom", "best",
        )
    )
    lines.append("-" * 118)

    order = {
        "OK": 0, "OK-LIMITED": 1, "OK-NOCONV": 2,
        "PARTIAL": 3, "BASE": 4, "NO-OUTPUT": 5, "FAILED": 6,
    }
    for entry in sorted(entries, key=lambda e: (order.get(e["verdict"], 9), str(e["formula"]))):
        lines.append(format_row(entry, expected_candidates))

    counts: Dict[str, int] = {}
    for entry in entries:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1

    total_candidates = sum(e["candidates"] for e in entries)
    total_screened = sum(e["screened"] for e in entries)
    total_relaxed = sum(e["relaxed"] for e in entries)
    total_converged = sum(e["converged"] for e in entries)

    lines.append("-" * 118)
    lines.append(
        "verdicts: "
        + ", ".join("{}={}".format(k, counts[k]) for k in sorted(counts))
    )
    lines.append(
        "totals: candidates={} screened={} relaxed={} converged={}".format(
            total_candidates, total_screened, total_relaxed, total_converged
        )
    )

    short_candidates = [
        e for e in entries
        if e["has_summary"] and not e["base_only"]
        and e["candidates"] < expected_candidates
    ]
    no_converge = [e for e in entries if e["has_summary"] and e["relaxed"] > 0 and e["converged"] == 0]
    failed = [e for e in entries if e["verdict"] == "FAILED"]

    if short_candidates:
        lines.append("")
        lines.append("candidates below {}:".format(expected_candidates))
        for e in short_candidates:
            lines.append(
                "  {:<9} {:<34} {}/{}".format(
                    str(e["id"]), str(e["formula"])[:34], e["candidates"], expected_candidates
                )
            )
    if no_converge:
        lines.append("")
        lines.append("relaxed but nothing converged (max_steps={}):".format(
            no_converge[0]["max_steps"] if no_converge else 0
        ))
        for e in no_converge:
            lines.append(
                "  {:<9} {:<34} relaxed={} max_steps={}".format(
                    str(e["id"]), str(e["formula"])[:34], e["relaxed"], e["max_steps"]
                )
            )
    if failed:
        lines.append("")
        lines.append("failed tasks:")
        for e in failed:
            lines.append(
                "  {:<9} {:<34} {}".format(
                    str(e["id"]), str(e["formula"])[:34], (e["error"] or "")[:150]
                )
            )
    lines.append("")

    if include_stages:
        stages = build_stage_entries(root, expected_candidates, expected_relax)
        if stages:
            lines.append("-" * 118)
            lines.append("intermediate stages under {}/_stages".format(root.name))
            lines.append("-" * 118)
            lines.append(
                "{:<9} {:<11} {:<34} {:>9} {:>5} {:>8} {:>8} {:>9} {:>4}  flags".format(
                    "task", "verdict", "stage dir", "cand", "scr", "relax",
                    "conv", "E/atom", "best",
                )
            )
            for entry in stages:
                lines.append(format_row(entry, expected_candidates))
            lines.append("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize per-target generation results into a log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Output root to scan; repeat for several roots (default: output/lpsc)",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help="Log file to write (use '-' for stdout only)",
    )
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=200,
        help="Candidate count a complete run should reach",
    )
    parser.add_argument(
        "--expected-relax",
        type=int,
        default=10,
        help="Relaxation count a complete run should reach per target",
    )
    parser.add_argument(
        "--include-stages",
        action="store_true",
        help="Also summarize intermediate stages under <root>/_stages/<task>/<NN>_<formula>",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the log instead of overwriting it",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = [Path(r).expanduser() for r in (args.root or [str(DEFAULT_ROOT)])]
    roots = [r if r.is_absolute() else (PROJECT_ROOT / r).resolve() for r in roots]

    lines: List[str] = [
        "=" * 118,
        "MatterSim generation summary",
        "generated: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")),
        "=" * 118,
        "",
    ]
    for root in roots:
        summarize_root(
            root,
            args.expected_candidates,
            args.expected_relax,
            lines,
            include_stages=args.include_stages,
        )

    text = "\n".join(lines)
    print(text)
    if args.log and args.log != "-":
        log_path = Path(args.log).expanduser()
        if not log_path.is_absolute():
            log_path = (SCRIPT_DIR / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(log_path, mode, encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("log written to {}".format(log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
