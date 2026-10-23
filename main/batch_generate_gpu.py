#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU batch doping generator.

Differences from batch_generate.py:
1. Runs MatterSim screening/relaxation on GPU (--device cuda).
2. Output directories are named after the target formula (the default
   behavior of generate_stru.py), e.g. output/Li5.4PS4ClBr0.6O0.4,
   instead of batch_targetN style names.
3. Relaxation step cap (RELAX_STEPS) is raised to 1000 because the earlier
   300-step runs never converged.
4. Uses local MatterSim weights so it runs offline. It looks for
   models/mattersim-v1.0.0-1M.pth next to this script first, then falls
   back to the mattersim default cache (~/.local/mattersim/pretrained_models).
   Override with the MATTERSIM_MODEL env variable.
5. All output (this script and every child process) is tee'd to a log file,
   and the run can be detached with --background.

Substitution-site conventions (matching the target compositions):
- O and Se replace S sites, restricted to the FREE-ANION sub-lattice
  (--host-site-filter free-anion) so the PS4 tetrahedra are not broken;
- F replaces Cl sites;
- metals (W/In/Ga/Bi/As/Si) replace P sites.

A single run can only do one host->dopant substitution, so targets with two
substitutions (In@P + F@Cl, O@S + F@Cl, Si@P + Se@S) are split into two
stages: stage A does the metal (or O) substitution, stage B starts from the
best structure of stage A and does the F/Se substitution with
--scaling 1,1,1 (no further supercell expansion).

Usage:
    python batch_generate_gpu.py                     # run all tasks in the foreground
    python batch_generate_gpu.py --background        # run detached, output to a log
    python batch_generate_gpu.py --list              # list task numbers
    python batch_generate_gpu.py 1 3 8               # run selected tasks only
    python batch_generate_gpu.py --log run.log       # use a specific log file

Logs go to logs/batch_gpu_<timestamp>.log. Follow with:
    tail -f logs/batch_gpu_*.log
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE = SCRIPT_DIR / "generate_stru.py"
BASE_CIF = SCRIPT_DIR / "basestru" / "Li5.4PS4.4Cl1Br0.6.cif"
BASE_FORMULA = "Li5.4PS4.4Cl1Br0.6"

LOG_DIR = SCRIPT_DIR / "logs"

# Local MatterSim weights shipped with the project, so no network is needed.
# Override with the MATTERSIM_MODEL environment variable if needed.
MODEL_PATH = Path(
    os.environ.get(
        "MATTERSIM_MODEL",
        str(SCRIPT_DIR / "models" / "mattersim-v1.0.0-1M.pth"),
    )
)
MATTERSIM_DEFAULT = (
    Path.home() / ".local" / "mattersim" / "pretrained_models"
    / "mattersim-v1.0.0-1M.pth"
)

# Output root (the default output location of generate_stru.py). Actual
# results go to OUTPUT_ROOT / <target formula>.
OUTPUT_ROOT = SCRIPT_DIR / "output"

# Run settings: GPU plus a 1000-step relaxation cap. DEVICE defaults to
# "auto" (CUDA when available, otherwise CPU) so a CPU-only torch build does
# not abort the run; force it with BATCH_DEVICE=cpu or BATCH_DEVICE=cuda.
DEVICE = os.environ.get("BATCH_DEVICE", "auto")
RELAX_STEPS = 1000


def resolve_model_args():
    """Pick the MatterSim weights to pass as --model (empty = mattersim default)."""
    if MODEL_PATH.is_file():
        return ["--model", str(MODEL_PATH)]
    if MATTERSIM_DEFAULT.is_file():
        # mattersim finds this path itself; no download attempt.
        return []
    print(
        "WARNING: no local MatterSim weights found at "
        f"{MODEL_PATH} or {MATTERSIM_DEFAULT}; mattersim will try to "
        "download the model, which needs network access.",
        file=sys.stderr,
    )
    return []


COMMON_ARGS = [
    "--n-candidates", "200",
    "--relax-optimizer", "FIRE",
    "--fmax", "0.05",
    "--relax-steps", str(RELAX_STEPS),
    "--device", DEVICE,
] + resolve_model_args() + [
    "DUMMY",  # placeholder for the required MP_API positional argument
]

# A 0.01-level dopant needs a supercell of about 1240 atoms (100 f.u.),
# so the atom budget is raised.
BIG_SUPERCELL = ["--max-supercell-atoms", "1500"]

# Stage B starts from the best supercell of stage A, no further expansion.
STAGE_B_SCALING = ["--scaling", "1,1,1"]

# Restrict the host site to the free-anion sub-lattice: the host atom must not
# be bonded to the framework element (P). This keeps O/Se on the free S site
# instead of breaking the PS4 tetrahedra. Needed because the base cell is P1
# (all sites are 1a), so --host-wyckoff cannot be used.
FREE_ANION = ["--host-site-filter", "free-anion"]

# Stage A of task 3 puts Si on a P site, so its SiS4 sulfur has no P neighbour
# and would be mistaken for free sulfur. Treat Si as a framework element too
# when filtering stage B.
FRAMEWORK_P_SI = ["--host-framework-element", "P,Si"]

# Uniformly random substitution subsets. The default greedy sampler maximises
# the dopant-dopant distance, which on a symmetric lattice produces only a
# handful of distinct arrangements (e.g. 8 free S choose 6 gave just 2 of 28).
# Used wherever more than one site is substituted out of a limited pool.
SUB_RANDOM = ["--substitution-sampler", "random"]

# Grow the auto supercell to at least 20 formula units. Balanced choice:
# 20 f.u. (248 atoms) gives a useful configurational ensemble without the cost
# of 40 f.u. (496 atoms):
#   T8A O0.3@freeS: 28 arrangements; T3A Si0.05@P: 20 arrangements.
MIN_FU_20 = ["--min-formula-units", "20"]

PREV = "<prev>"  # means: use the best structure of the previous stage

N_RELAX_A = 30
N_RELAX_B = 20
N_RELAX_SI_SE = 28


def stage(base, base_formula, target, host, dopant, comp, extra=(), relax=N_RELAX_A):
    return {
        "base": base,
        "base_formula": base_formula,
        "target": target,
        "host": host,
        "dopant": dopant,
        "comp": comp,
        "extra": list(extra),
        "relax": relax,
    }


# ------------------------------------------------------------
# Target compositions and task definitions
# ------------------------------------------------------------

T_O = "Li5.4PS4ClBr0.6O0.4"
T_W = "Li5.41P0.99W0.01S4.4ClBr0.6"
T_SI_SE = "Li5.45P0.95S4.3ClBr0.6Si0.05Se0.1"
T_IN = "Li5.42P0.99In0.01S4.4Cl0.75F0.25Br0.6"
T_GA = "Li5.42P0.99Ga0.01S4.4Cl0.75F0.25Br0.6"
T_BI = "Li5.42P0.99Bi0.01S4.4Cl0.75F0.25Br0.6"
T_AS = "Li5.4P0.99As0.01S4.4Cl0.75F0.25Br0.6"
T_O_F = "Li5.4PS4.1O0.3Cl0.9F0.1Br0.6"

# Intermediate compositions for the two-substitution targets (stage A).
T_SI_A = "Li5.45P0.95S4.4ClBr0.6Si0.05"
T_IN_A = "Li5.42P0.99In0.01S4.4Cl1Br0.6"
T_GA_A = "Li5.42P0.99Ga0.01S4.4Cl1Br0.6"
T_BI_A = "Li5.42P0.99Bi0.01S4.4Cl1Br0.6"
T_AS_A = "Li5.4P0.99As0.01S4.4Cl1Br0.6"
T_O_A = "Li5.4PS4.1O0.3Cl1Br0.6"


def metal_then_f(metal, intermediate, final):
    """Metal on P (stage A) then F on Cl (stage B)."""
    return [
        stage(BASE_CIF, BASE_FORMULA, intermediate,
              "P", metal, "Li", extra=BIG_SUPERCELL, relax=N_RELAX_A),
        stage(PREV, intermediate, final,
              "Cl", "F", "Li", extra=STAGE_B_SCALING + SUB_RANDOM,
              relax=N_RELAX_B),
    ]


TASKS = [
    {
        # O 0.4 equals the free-S 0.4/f.u.: exactly one ordered arrangement at
        # ANY cell size, so keep the smallest cell (62 atoms).
        "name": f"O on S -> {T_O}",
        "stages": [
            stage(BASE_CIF, BASE_FORMULA, T_O, "S", "O", "Li",
                  extra=FREE_ANION + SUB_RANDOM),
        ],
    },
    {
        "name": f"W on P -> {T_W}",
        "stages": [stage(BASE_CIF, BASE_FORMULA, T_W, "P", "W", "Li",
                         extra=BIG_SUPERCELL)],
    },
    {
        "name": f"Si on P + Se on S -> {T_SI_SE}",
        "stages": [
            stage(BASE_CIF, BASE_FORMULA, T_SI_A, "P", "Si", "Li",
                  extra=MIN_FU_20),
            stage(PREV, T_SI_A, T_SI_SE, "S", "Se", "Li",
                  extra=STAGE_B_SCALING + FREE_ANION + FRAMEWORK_P_SI
                  + SUB_RANDOM, relax=N_RELAX_SI_SE),
        ],
    },
    {
        "name": f"In on P + F on Cl -> {T_IN}",
        "stages": metal_then_f("In", T_IN_A, T_IN),
    },
    {
        "name": f"Ga on P + F on Cl -> {T_GA}",
        "stages": metal_then_f("Ga", T_GA_A, T_GA),
    },
    {
        "name": f"Bi on P + F on Cl -> {T_BI}",
        "stages": metal_then_f("Bi", T_BI_A, T_BI),
    },
    {
        "name": f"As on P + F on Cl -> {T_AS}",
        "stages": metal_then_f("As", T_AS_A, T_AS),
    },
    {
        "name": f"O on S + F on Cl -> {T_O_F}",
        "stages": [
            stage(BASE_CIF, BASE_FORMULA, T_O_A, "S", "O", "Li",
                  extra=FREE_ANION + MIN_FU_20 + SUB_RANDOM),
            stage(PREV, T_O_A, T_O_F, "Cl", "F", "Li",
                  extra=STAGE_B_SCALING + SUB_RANDOM, relax=N_RELAX_B),
        ],
    },
]


# ------------------------------------------------------------
# Logging: tee stdout/stderr to a file
# ------------------------------------------------------------

class Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except (ValueError, OSError):
                # stream already closed (e.g. during interpreter shutdown)
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return False


def default_log_path():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"batch_gpu_{stamp}.log"


def setup_logging(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = Tee(original_out, handle)
    sys.stderr = Tee(original_err, handle)
    return handle, original_out, original_err


# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------

def output_dir_for(target_formula):
    return OUTPUT_ROOT / target_formula


def build_cmd(st, prev_target):
    """Translate a stage descriptor into a generate_stru.py command.

    No --output is passed, so the output directory is named after the
    target formula automatically: output/<target formula>.
    """
    if st["base"] == PREV:
        base_path = output_dir_for(prev_target) / "best_structure.cif"
    else:
        base_path = Path(st["base"])

    return [
        sys.executable, str(GENERATE),
        "--base-structure", str(base_path),
        "--base-formula", st["base_formula"],
        "--target-formula", st["target"],
        "--host", st["host"],
        "--dopant", st["dopant"],
        "--compensating-element", st["comp"],
    ] + st["extra"] + [
        "--n-relax", str(st["relax"]),
    ] + COMMON_ARGS


def run_cmd(title, cmd):
    out_dir = output_dir_for(cmd[cmd.index("--target-formula") + 1])
    print(f"\n{'=' * 70}")
    print(f"Task: {title}")
    print(f"{'=' * 70}")
    print("Command: " + " ".join(cmd))
    print("Output dir: " + str(out_dir))
    print(flush=True)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        sys.stdout.write(line)

    process.wait()
    if process.returncode != 0:
        print(f"FAILED (returncode={process.returncode}): {title}")
    else:
        print(f"DONE: {title}")
    return process.returncode


def describe_model():
    if MODEL_PATH.is_file():
        return f"bundled {MODEL_PATH}"
    if MATTERSIM_DEFAULT.is_file():
        return f"mattersim default {MATTERSIM_DEFAULT}"
    return "NOT FOUND (mattersim will try to download it; requires network)"


def describe_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda else 0
        return (
            f"torch {torch.__version__} cuda_available={cuda} "
            f"device_count={count}"
        )
    except Exception as exc:
        return f"torch import failed: {exc}"


def run_task(task):
    print(f"\n\n######## {task['name']} ########", flush=True)
    prev_target = None
    for index, st in enumerate(task["stages"]):
        if st["base"] == PREV:
            base_path = output_dir_for(prev_target) / "best_structure.cif"
            if not base_path.exists():
                print(f"Skip stage {index + 1}: missing previous result {base_path}")
                return 1
        cmd = build_cmd(st, prev_target)
        rc = run_cmd(task["name"], cmd)
        if rc != 0:
            print("Stage failed; aborting this task")
            return rc
        prev_target = st["target"]
    return 0


def print_final_report(indices):
    """Compact per-task summary so the end result is visible at a glance."""
    print(f"\n{'=' * 70}")
    print("Final results")
    print("=" * 70)
    for index in indices:
        task = TASKS[index - 1]
        target = task["stages"][-1]["target"]
        out_dir = output_dir_for(target)
        best = out_dir / "best_structure.cif"
        summary = out_dir / "summary.json"

        status = "OK" if best.is_file() else "MISSING"
        detail = ""
        if summary.is_file():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                detail = (
                    f"  candidates={data.get('candidate_count')}"
                    f" screened={data.get('screened_count')}"
                    f" relaxed={data.get('relaxed_count')}"
                    f" converged={data.get('converged_count')}"
                    f" best_E/atom={data.get('best_energy_per_atom_ev')}"
                )
            except Exception as exc:  # keep going even if a summary is broken
                detail = f"  (summary unreadable: {exc})"

        print(f"[{index}] {status:<7} {target}")
        print(f"      dir: {out_dir}{detail}")


# ------------------------------------------------------------
# Argument handling and main
# ------------------------------------------------------------

def parse_args(argv):
    flags = {"background": False, "list": False, "log": None, "indices": []}
    args = argv[1:]
    position = 0
    while position < len(args):
        token = args[position]
        if token == "--background":
            flags["background"] = True
        elif token == "--list":
            flags["list"] = True
        elif token == "--log":
            position += 1
            if position >= len(args):
                raise SystemExit("--log needs a path")
            flags["log"] = args[position]
        elif token.startswith("--log="):
            flags["log"] = token.split("=", 1)[1]
        elif token.startswith("-"):
            print(f"Ignore unknown option: {token}")
        else:
            flags["indices"].append(int(token))
        position += 1
    return flags


def launch_background(flags, log_path):
    """Re-launch this script detached, writing everything to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--log", str(log_path)]
    cmd += [str(index) for index in flags["indices"]]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(SCRIPT_DIR),
    )
    print(f"Started in background: PID {process.pid}")
    print(f"Log file: {log_path}")
    print(f"Follow with: tail -f {log_path}")


def main(argv):
    flags = parse_args(argv)

    if flags["list"]:
        for index, task in enumerate(TASKS, start=1):
            print(f"{index}: {task['name']}")
        return 0

    log_path = Path(flags["log"]) if flags["log"] else default_log_path()

    if flags["background"]:
        launch_background(flags, log_path)
        return 0

    handle, original_out, original_err = setup_logging(log_path)
    try:
        indices = flags["indices"] or list(range(1, len(TASKS) + 1))

        print(f"Log file: {log_path}")
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tasks: {indices}")
        print(f"Device: {DEVICE}, relax-steps: {RELAX_STEPS}")
        print(f"Model: {describe_model()}")
        print(f"Runtime: {describe_torch()}", flush=True)

        failures = []
        for index in indices:
            if not 1 <= index <= len(TASKS):
                print(f"Ignore invalid index: {index}")
                continue
            if run_task(TASKS[index - 1]) != 0:
                failures.append(index)

        valid = [i for i in indices if 1 <= i <= len(TASKS)]
        print_final_report(valid)

        print(f"\n{'=' * 70}")
        if failures:
            print(f"Failed tasks: {failures}")
        else:
            print("All tasks completed")
        print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 1 if failures else 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_out, original_err
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
