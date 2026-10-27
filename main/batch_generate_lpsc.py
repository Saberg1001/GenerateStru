#!/usr/bin/env python3
"""只生成 main/LPSC-exp.csv 中的结构。

基于 batch_generate_experiment.py，复用其规划/生成/续跑逻辑，仅覆盖以下配置：

- 默认输入：main/LPSC-exp.csv，且只接受 base=LPSC 的行；
- 默认输出：main/output/lpsc；
- 默认只使用本地 base 结构（main/basestru），不回退 Materials Project；
- base 结构查找顺序：main/basestru -> output*/<化学式>/best_structure.cif。

如果某个 parent 的 base CIF 缺失，可以把之前生成结果中的
``output/<化学式>/best_structure.cif`` 复制为 ``main/basestru/<化学式>.cif``；
也可以直接运行本脚本的 ``--sync-bases`` 自动完成这一步。

常用命令：
    python batch_generate_lpsc.py --check-bases        # 检查还缺哪些 base
    python batch_generate_lpsc.py --sync-bases         # 从 output 收集 base
    python batch_generate_lpsc.py --list               # 只规划并写 plan.csv
    python batch_generate_lpsc.py                      # 正式生成
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import batch_generate_experiment as core

SCRIPT_DIR = Path(__file__).resolve().parent
LPSC_CSV = SCRIPT_DIR / "LPSC-exp.csv"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "lpsc"
LOG_DIR = SCRIPT_DIR / "logs" / "lpsc"

LPSC_PARENTS = list(core.PARENT_LIBRARY["LPSC"])
LPSC_KNOWN_BASES = {
    "Li6PS5Cl",
    "Li5.4PS4.4Cl1.6",
    "Li5.4PS4.4Cl1Br0.6",
    "Li5.5PS4.5Cl1.5",
    "Li5.5AsS4.5Br1.5",
    "Li6AsS5I",
    "Li6SbS5I",
    "Li6.7Sb0.3Si0.7S5I",
    "Li6.7Si0.7Sb0.3S5I",
}

core.DEFAULT_CSV = LPSC_CSV
core.OUTPUT_ROOT = OUTPUT_ROOT
core.LOG_DIR = LOG_DIR
core.PARENT_LIBRARY = {"LPSC": LPSC_PARENTS}
core.KNOWN_BASES = LPSC_KNOWN_BASES
core.REQUIRE_LOCAL_BASE = True

_ALLOW_MP = "--allow-mp" in sys.argv[1:]
_ORIGINAL_BUILD_PARSER = core.build_parser


def build_parser():
    parser = _ORIGINAL_BUILD_PARSER()
    parser.description = "批量生成 LPSC-exp.csv 中的结构（默认仅使用本地 base）"
    parser.set_defaults(
        csv=str(LPSC_CSV),
        output_root=str(OUTPUT_ROOT),
        no_mp=not _ALLOW_MP,
    )
    parser.add_argument(
        "--allow-mp",
        action="store_true",
        help="本地找不到 base 时允许回退 Materials Project（默认关闭）",
    )
    parser.add_argument(
        "--sync-bases",
        action="store_true",
        help="运行前把 output*/<化学式>/best_structure.cif 复制到 main/basestru",
    )
    parser.add_argument(
        "--check-bases",
        action="store_true",
        help="只检查各任务所需的 base 是否存在，然后退出",
    )
    return parser


core.build_parser = build_parser


def sync_bases() -> int:
    dest_dir = core.BASESTRU_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    roots = [
        core.SCRIPT_DIR / "output" / "output",
        core.SCRIPT_DIR / "output_local",
        core.SCRIPT_DIR / "output" / "experiment",
        OUTPUT_ROOT,
    ]
    copied = 0
    for root in roots:
        if not root.is_dir():
            continue
        for cif in sorted(root.glob("*/best_structure.cif")):
            dest = dest_dir / f"{cif.parent.name}.cif"
            if dest.exists():
                continue
            shutil.copy2(cif, dest)
            copied += 1
            print(f"[basestru] {cif} -> {dest}")
    print(f"[basestru] copied {copied} base structure(s) to {dest_dir}")
    return 0


def _load_lpsc_tasks():
    tasks = core.load_tasks(LPSC_CSV, use_recommended=False)
    tasks = [task for task in tasks if task.label.strip() == "LPSC"]
    core.mark_duplicate_formulas(tasks)
    return tasks


def check_bases() -> int:
    tasks = _load_lpsc_tasks()
    known_targets = {core.formula_key(task.formula) for task in tasks}
    by_formula = {}
    for task in tasks:
        by_formula.setdefault(core.formula_key(task.formula), task)

    missing: dict[str, list[str]] = {}
    checked: set[tuple[tuple[str, float], ...]] = set()

    def resolve(task: core.Task, stack: frozenset) -> None:
        key = core.formula_key(task.formula)
        if key in checked or task.status.startswith("skipped"):
            return
        checked.add(key)
        _parent, stages, _error = core.plan_task(task, known_targets)
        if not stages:
            return
        base = stages[0].base_formula
        if core.find_base_path(base) is not None:
            return
        dependency = by_formula.get(core.formula_key(base))
        if (
            dependency is not None
            and dependency is not task
            and dependency.formula not in stack
        ):
            resolve(dependency, stack | {task.formula})
            return
        missing.setdefault(base, []).append(task.task_id)

    for task in tasks:
        resolve(task, frozenset())

    if not missing:
        print(f"所有 base 均已就绪（{len(tasks)} 个 LPSC 任务）")
        return 0

    print(f"缺少 {len(missing)} 个 base 结构：")
    for formula, task_ids in sorted(missing.items()):
        print(f"  {formula:30s} <- {', '.join(task_ids)}")
    print()
    print(f"请将对应结构复制到 {core.BASESTRU_DIR}/<化学式>.cif，")
    print("或运行 python batch_generate_lpsc.py --sync-bases 从 output 收集。")
    return 1


def main() -> int:
    argv = sys.argv[1:]
    do_sync = "--sync-bases" in argv
    do_check = "--check-bases" in argv
    sys.argv = [item for item in argv if item not in {"--sync-bases", "--check-bases"}]
    sys.argv.insert(0, "batch_generate_lpsc.py")

    if do_sync:
        sync_bases()
    if do_check:
        return check_bases()
    return core.main()


if __name__ == "__main__":
    sys.exit(main())
