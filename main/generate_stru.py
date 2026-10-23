import os
import re
import csv
import gc
import json
import math
import time
import random
import hashlib
import argparse
import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np

from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

def parse_scaling(text):
    values = [int(x) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError("scaling 必须是 a,b,c，例如 1,2,5")
    if any(v <= 0 for v in values):
        raise ValueError(f"scaling 必须为正整数，收到 {values}")
    return values


def round_count(value, tol=1e-4, strict=True):
    """
    将化学计量浮点数转换为整数计数。

    不直接使用 round()，因为 Python 使用「银行家舍入」，
    例如 round(2.5) == 2，会给出错误计数。

    strict=True：数值离整数较远时直接报错（默认，避免静默取整）；
    strict=False：按最接近的整数取值，偏差由调用方显式报告
    （对应「有理逼近 + 显式报告偏差」策略）。
    """
    nearest = int(math.floor(value + 0.5))
    if strict and abs(value - nearest) > tol:
        raise ValueError(
            f"期望整数原子数，但计算得到 {value}（最接近 {nearest}）。"
            f"请检查 target-formula 与 scaling 是否自洽，"
            f"或用 --scaling auto 让脚本自动定超胞，"
            f"或加 --allow-stoichiometry-approximation 接受近似并查看偏差。"
        )
    return nearest


def best_diagonal_scaling(structure, multiplier):
    """
    在行列式等于 multiplier 的对角超胞中挑「形状最好」的一个。

    注意：放大倍数 = 三个分量的乘积，不是每个轴各放大这么多倍。
    例如 multiplier=10 → 1x2x5（体积 ×10），而不是 10x10x10（体积 ×1000）。
    """
    triples = set()
    for a in range(1, multiplier + 1):
        if multiplier % a:
            continue
        quotient = multiplier // a
        for b in range(1, quotient + 1):
            if quotient % b:
                continue
            c = quotient // b
            triples.update(itertools.permutations((a, b, c)))

    lengths = np.asarray(structure.lattice.abc, dtype=float)

    def shape_score(triple):
        scaled = lengths * np.asarray(triple)
        return float(scaled.max() / scaled.min()), float(scaled.max())

    return tuple(min(triples, key=shape_score))


def required_volume_multiplier(
    base_formula,
    target_formula,
    formula_units,
    max_multiplier,
):
    """
    计算「让目标化学式的所有元素计数同时成为整数」所需的最小体积倍数。

    做法：目标式每个元素在一个基胞里的原子数为
        x_el = target[el] * formula_units
    把 x_el 还原成最简分数 p/q（q 为下标分母），则体积倍数必须同时是
    所有 q 的倍数，取最小公倍数即可（对应「扩胞放大到最小公倍数」策略）。

    返回 (multiplier, denominators, exact)。
    multiplier 超过 max_multiplier 时 exact=False，表示无法在可接受的
    超胞规模内精确整数化，需要走近似路径。
    """
    target = formula_dict(target_formula)
    denominators = {}
    multiplier = 1
    exact = True

    for element, amount in sorted(target.items()):
        value = amount * formula_units
        if abs(value - round(value)) < 1e-9:
            denominators[element] = 1
            continue

        # 把目标原子数还原成「分母尽量小」的最简分数
        fraction = Fraction(value).limit_denominator(max_multiplier)
        denominator = fraction.denominator
        denominators[element] = denominator

        # 还原误差过大说明真实分母超过 max_multiplier：
        # 这是「有理逼近」而不是精确整数化，偏差稍后要显式报告
        if abs(float(fraction) - value) > 1e-9:
            exact = False

        multiplier = (
            multiplier * denominator // math.gcd(multiplier, denominator)
        )

    if multiplier > max_multiplier:
        exact = False
    return multiplier, denominators, exact


def composition_deviation(base_formula, target_formula, counts, formula_units):
    """
    比较「实际可达组成」与「目标组成」。

    counts 是超胞内各元素的原子数；换算成每 formula unit 的原子数后，
    再与目标化学式比较，给出原子分数偏差（即表中「实际比例 / 偏差」）。
    """
    target = formula_dict(target_formula)
    base = formula_dict(base_formula)

    actual_per_fu = {
        element: counts.get(element, 0.0) / formula_units
        for element in sorted(set(target) | set(base))
    }
    target_fractions = atomic_fractions_from_dict(target)
    actual_fractions = atomic_fractions_from_dict(actual_per_fu)

    rows = []
    for element in sorted(set(target) | set(base) | set(counts)):
        target_fraction = target_fractions.get(element, 0.0)
        actual_fraction = actual_fractions.get(element, 0.0)
        rows.append(
            {
                "element": element,
                "target_per_fu": target.get(element, 0.0),
                "actual_per_fu": actual_per_fu.get(element, 0.0),
                "target_fraction": target_fraction,
                "actual_fraction": actual_fraction,
                "deviation": actual_fraction - target_fraction,
            }
        )

    max_deviation = max(
        (abs(row["deviation"]) for row in rows), default=0.0
    )
    return {
        "rows": rows,
        "max_deviation": max_deviation,
        "max_deviation_percent": max_deviation * 100.0,
        "target_per_fu": target,
        "actual_per_fu": actual_per_fu,
    }


def atomic_fractions_from_dict(amounts):
    positive = {
        element: float(amount)
        for element, amount in amounts.items()
        if float(amount) > 0
    }
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("无法比较空组成")
    return {
        element: amount / total for element, amount in positive.items()
    }


def plan_scaling_for_target(
    structure,
    base_formula,
    target_formula,
    max_multiplier,
    explicit=None,
    min_formula_units=0,
):
    """
    决定超胞矩阵（对应「策略 1：扩胞放大到最小公倍数」）。

    explicit 为 None 时自动计算体积倍数并挑形状最好的对角矩阵；
    否则校验用户给定的矩阵乘积是否真的能让计数整数化。

    min_formula_units > 0 时，把自动算出的体积倍数再乘一个整数因子，
    使超胞至少包含这么多 f.u.（计数仍是整数）。用于掺杂位点很少、
    小胞里可取代的位点数目不足以产生构型多样性的情况。
    """
    formula_units = get_formula_units(structure, base_formula)

    if explicit is None:
        multiplier, denominators, exact = required_volume_multiplier(
            base_formula,
            target_formula,
            formula_units,
            max_multiplier,
        )
        if multiplier is None or multiplier > max_multiplier:
            return {
                "mode": "auto",
                "scaling": None,
                "multiplier": multiplier,
                "denominators": denominators,
                "exact": False,
                "note": (
                    f"精确整数化需要体积倍数 {multiplier}"
                    f"（≈{len(structure) * multiplier} 原子），"
                    f"超过原子预算"
                ),
            }
        growth_note = ""
        if min_formula_units > 0:
            current_units = formula_units * multiplier
            if current_units < min_formula_units:
                factor = -(-min_formula_units // current_units)
                multiplier *= factor
                growth_note = (
                    f"，再 ×{factor} 放大到 {formula_units * multiplier} f.u."
                )
        if multiplier > max_multiplier:
            return {
                "mode": "auto",
                "scaling": None,
                "multiplier": multiplier,
                "denominators": denominators,
                "exact": False,
                "note": (
                    f"精确整数化 + 至少 {min_formula_units} f.u. 需要体积倍数 "
                    f"{multiplier}（≈{len(structure) * multiplier} 原子），"
                    f"超过原子预算"
                ),
            }
        scaling = best_diagonal_scaling(structure, multiplier)
        return {
            "mode": "auto",
            "scaling": scaling,
            "multiplier": multiplier,
            "denominators": denominators,
            "exact": exact,
            "note": (
                f"按下标分母 LCM 定体积倍数，"
                f"矩阵 {'x'.join(str(v) for v in scaling)}"
                + growth_note
                + ("" if exact else "（有理逼近）")
            ),
        }

    scaling = list(explicit)
    multiplier = int(np.prod(scaling))
    return {
        "mode": "manual",
        "scaling": scaling,
        "multiplier": multiplier,
        "denominators": {},
        "exact": None,
        "note": f"用户指定矩阵 {'x'.join(str(v) for v in scaling)}",
    }


def formula_dict(formula):
    """
    将化学式转换为元素计数字典。
    """
    comp = Composition(formula)
    return {
        str(el): float(amount)
        for el, amount in comp.get_el_amt_dict().items()
    }


def get_formula_units(structure, base_formula):
    """
    计算 structure 中包含多少个 base_formula 化学式单元。

    使用 composition 而不是逐 site.specie，以兼容存在部分占据
    （disorder）的结构。
    """
    base_counts = formula_dict(base_formula)
    struct_counts = {
        str(el): float(amount)
        for el, amount in structure.composition.get_el_amt_dict().items()
    }

    ratios = []
    for el, n_base in base_counts.items():
        if n_base <= 0:
            continue
        if el not in struct_counts:
            raise ValueError(f"结构中缺少元素 {el}")
        ratios.append(struct_counts[el] / n_base)

    if not ratios:
        raise ValueError(f"基础化学式 {base_formula} 为空")

    if max(ratios) - min(ratios) > COMPOSITION_MATCH_TOLERANCE:
        raise ValueError(
            f"结构组成与 {base_formula} 不匹配: "
            f"{dict(struct_counts)}, ratios={ratios}"
        )

    extra = set(struct_counts) - set(base_counts)
    if extra:
        raise ValueError(
            f"结构中出现基础化学式之外的元素: {sorted(extra)}"
        )

    return round_count(ratios[0])


def resolve_device(device="auto"):
    """
    将 'auto' 解析为实际可用设备（cuda / mps / cpu）。
    """
    if device and device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


# ------------------------------------------------------------
# 基体结构来源：优先本地 basestru/，否则回退 Materials Project
# ------------------------------------------------------------

# 脚本同级目录下的 basestru/：把基体结构放进去即可优先使用
DEFAULT_BASESTRU_DIR = Path(__file__).resolve().parent / "basestru"

# 不指定 --output 时，所有结果统一放在脚本同级目录的 output/ 下，
# 再按目标化学式分子目录，例如 main/output/Li5.4PS4.4Cl1.6。
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def formula_dirname(formula):
    """
    把化学式转成可作目录名的字符串（去掉空白等）。
    """
    return re.sub(r"\s+", "", str(formula))


def normalize_formula_key(text):
    """
    归一化化学式字符串，用于宽松的文件名匹配：
    忽略大小写、空格、下划线和连字符。
    """
    return re.sub(r"[\s_\-]+", "", str(text)).lower()


# 组成匹配容差：同一材料写成整数倍胞/小数下标时，元素比例应当完全一致，
# 这里只用来吸收浮点误差（实测为 0，见 basestru 下的文件），因此取 1e-5。
# 它同时用于两处，保证「匹配」不会比「校验」更宽松：
# - composition_matches_scaled 比较原子分数（差 <= tolerance）；
# - get_formula_units 比较各元素的比例 struct/base（极差 <= tolerance）。
# 由于原子分数差 ≈ 分数 × 比例相对差，且分数<=1、比例>=1，
# 校验通过的文件必然也能通过匹配，不会出现「选中后又被拒」。
COMPOSITION_MATCH_TOLERANCE = 1e-5


def candidate_geometry_key(structure, ndigits=4):
    """
    候选结构的「与位点顺序无关」指纹，用于候选判重。

    同一组取代/插入位点，若只是选择顺序不同，得到的是同一个结构；位点顺序
    会写在 cif 里，所以直接把 sites 排序后再比较（元素 + 分数坐标）。
    """
    entries = [
        (str(site.specie),)
        + tuple(np.round(site.frac_coords, ndigits).tolist())
        for site in structure.sites
    ]
    entries.sort()
    return tuple(entries)


def atomic_fraction_map(composition):
    """把组成换成 {元素: 原子分数}，空组成返回空字典。"""
    amounts = {
        str(element): float(amount)
        for element, amount in composition.get_el_amt_dict().items()
    }
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {element: amount / total for element, amount in amounts.items()}


def composition_matches_scaled(
    composition, reference, tolerance=COMPOSITION_MATCH_TOLERANCE
):
    """
    判断两个组成是否只相差一个正比例因子（即元素比例一致）。

    掺杂本质上是「小数个原子」，目标组成只能用扩胞表达，因此同一材料会
    以不同倍数写出，例如 Li5.4PS4.4Cl1Br0.6 的 5 倍胞是 Li27P5S22Br3Cl5。
    用约简组成的精确相等匹配会漏掉这种整数倍/小数下标写法，这里改为比较
    原子分数（比例不变量），扩胞前后都能命中。
    """
    reference_fractions = atomic_fraction_map(reference)
    composition_fractions = atomic_fraction_map(composition)
    if not reference_fractions or not composition_fractions:
        return False
    elements = set(reference_fractions) | set(composition_fractions)
    return all(
        abs(
            composition_fractions.get(element, 0.0)
            - reference_fractions.get(element, 0.0)
        )
        <= tolerance
        for element in elements
    )


def structure_hash(structure, ndigits=4):
    """
    基体结构的短哈希（组成 + 晶格 + 分数坐标）。

    用于候选指纹：MP 端换了基体结构、或本地 basestru 文件被替换时，
    都必须让旧的候选结构失效。
    """
    payload = {
        "formula": structure.composition.formula,
        "lattice": np.round(structure.lattice.matrix, ndigits).tolist(),
        "frac_coords": np.round(structure.frac_coords, ndigits).tolist(),
        "species": [str(specie) for specie in structure.species],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()[:12]


def find_base_structure(basestru_dir, base_formula):
    """
    在 basestru 目录里查找与 base_formula 匹配的基体结构文件。

    匹配顺序：
    1. 文件名（忽略大小写/空格/下划线/连字符）等于 base_formula 的
       各种写法（原式、约简式、Hill 式）；
    2. 否则逐个解析文件，按「元素比例一致」匹配。

    组成比较用 composition_matches_scaled，允许文件写成扩胞/小数下标，
    因此掺杂这种「小数个原子」的场景也能命中同一材料的基体结构。

    返回 Path 或 None。文件内容仍会校验组成，避免误用命名不一致的文件。
    """
    directory = Path(basestru_dir).expanduser()
    if not directory.is_dir():
        return None

    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        return None

    composition = Composition(base_formula)
    name_keys = {
        normalize_formula_key(value)
        for value in (
            base_formula,
            composition.reduced_formula,
            composition.hill_formula,
            composition.formula,
        )
    }

    by_name = [
        path for path in files
        if normalize_formula_key(path.stem) in name_keys
        or normalize_formula_key(path.name) in name_keys
    ]

    def matches(path):
        try:
            structure = Structure.from_file(str(path))
        except Exception as exc:
            print(f"[BaseStru] 跳过无法解析的 {path.name}: {exc}")
            return None
        if not composition_matches_scaled(structure.composition, composition):
            return None
        return structure

    def matched_structures(paths):
        found = []
        for path in paths:
            structure = matches(path)
            if structure is not None:
                found.append((len(structure), path))
        return found

    if by_name:
        matched = matched_structures(by_name)
        if matched:
            matched.sort(key=lambda item: (item[0], item[1].name))
            return matched[0][1]
        print(
            f"[BaseStru] 文件名匹配到 {[p.name for p in by_name]}，"
            f"但组成与 {base_formula} 不一致，改为按组成扫描整个目录"
        )

    matched = matched_structures(files)
    if not matched:
        return None

    # 同组成有多个文件时取原子数最少的（通常是惯用胞/原胞）
    matched.sort(key=lambda item: (item[0], item[1].name))
    if len(matched) > 1:
        print(
            f"[BaseStru] {directory} 中有 {len(matched)} 个组成匹配的结构，"
            f"使用原子数最少的 {matched[0][1].name}"
        )
    return matched[0][1]


def resolve_base_structure(args):
    """
    按优先级确定起始基体结构：

    1. --base-structure 显式指定的文件；
    2. basestru 目录中与 --base-formula 匹配的文件（--use-basestru）；
    3. 回退到 Materials Project（此时才需要 MP_API_KEY）。

    返回 (structure, source, base_file, mp_doc)。
    """
    if args.base_structure:
        path = Path(args.base_structure).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"--base-structure 指向的文件不存在: {args.base_structure}"
            )
        structure = Structure.from_file(str(path))
        print(f"[BaseStru] 使用 --base-structure 指定的结构: {path}")
        return structure, "file", str(path), None

    if args.use_basestru:
        local = find_base_structure(args.basestru_dir, args.base_formula)
        if local is not None:
            structure = Structure.from_file(str(local))
            print(f"[BaseStru] 在 basestru 中找到基体结构: {local}")
            return structure, "basestru", str(local), None
        print(
            f"[BaseStru] {args.basestru_dir} 中没有与 "
            f"{args.base_formula} 匹配的结构，回退 Materials Project"
        )
        print(
            "[BaseStru] 提示：把基体结构文件（如 "
            f"{args.base_formula}.cif）放到 {args.basestru_dir} 下，"
            "之后即可跳过 Materials Project 直接用本地结构"
        )

    mp_key = os.environ.get("MP_API_KEY")
    if not mp_key:
        raise RuntimeError(
            f"basestru 中没有 {args.base_formula}，且未设置 MP_API_KEY。"
            "请把基体结构放到 basestru 目录，或用 --base-structure 指定文件，"
            "或设置 MP_API_KEY（https://materialsproject.org/api）"
        )

    mp_doc, structure = search_material(mp_key, args.base_formula)
    return structure, "mp", None, mp_doc


# ------------------------------------------------------------
# Materials Project 查询
# ------------------------------------------------------------

def search_material(mp_key, formula):
    """
    从 Materials Project 搜索基础材料，返回 (doc, structure)。
    """
    # 延迟导入：几何搜索流程不依赖 MP API，便于离线/无网环境测试。
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError(
            "需要 mp-api 才能查询 Materials Project，"
            "请先安装 mp-api（pip install mp-api）"
        ) from exc

    with MPRester(mp_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id",
                "formula_pretty",
                "energy_above_hull",
                "structure",
                "symmetry",
            ],
        )

    if len(docs) == 0:
        raise RuntimeError(f"Materials Project 中没有找到 {formula}")

    # 优先保留元素比例匹配的条目（允许 MP 写成不同倍数的胞）
    target_comp = Composition(formula)
    matched = [
        d for d in docs
        if composition_matches_scaled(d.structure.composition, target_comp)
    ]
    if matched:
        docs = matched

    # 优先选择 energy above hull 较低者（热力学更稳定）
    docs = sorted(
        docs,
        key=lambda x: (
            x.energy_above_hull
            if x.energy_above_hull is not None
            else 999.0
        ),
    )

    print(f"[MP] 命中 {len(docs)} 个条目:")
    for d in docs[:5]:
        print(
            f"     {d.material_id}  {d.formula_pretty}  "
            f"E_hull={d.energy_above_hull}"
        )

    doc = docs[0]
    structure = doc.structure

    print(f"[MP] material_id = {doc.material_id}")
    print(f"[MP] formula     = {doc.formula_pretty}")
    print(f"[MP] nsites      = {len(structure)}")
    print(f"[MP] E_hull      = {doc.energy_above_hull}")

    return doc, structure


# ------------------------------------------------------------
# 对称性 / Wyckoff 分析
# ------------------------------------------------------------

def site_wyckoff_labels(structure, symprec=1e-3):
    """
    返回与 structure.sites 顺序一一对应的 Wyckoff 标签列表。

    注意：SymmetrizedStructure.wyckoff_symbols 的长度等于「轨道数」，
    并不是原子数；必须通过 equivalent_indices 映射回每个原子。
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symm = sga.get_symmetrized_structure()

    labels = ["?"] * len(structure)
    for group, symbol in zip(
        symm.equivalent_indices,
        symm.wyckoff_symbols,
    ):
        for index in group:
            labels[index] = symbol

    return sga, labels


def print_symmetry_info(structure, label="structure"):
    """
    自动判断空间群、Wyckoff 标签和现有原子的位点。
    """
    print(f"\n[Symmetry] {label}")

    for symprec in [1e-2, 1e-3, 1e-4]:
        try:
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            print(
                f"  symprec={symprec:g}: "
                f"{sga.get_space_group_symbol()} "
                f"({sga.get_space_group_number()})"
            )
        except Exception as exc:
            print(f"  symmetry analysis failed: {exc}")

    try:
        _, labels = site_wyckoff_labels(structure, symprec=1e-3)
        for i, site in enumerate(structure.sites):
            print(
                f"  atom {i:4d}: "
                f"{str(site.specie):>3s} "
                f"Wyckoff={labels[i]:>4s} "
                f"frac={np.round(site.frac_coords, 6).tolist()}"
            )
    except Exception as exc:
        print(f"  Wyckoff 分析失败: {exc}")


def get_wyckoff_report(structure, symprec=1e-3):
    """
    返回 JSON 可序列化的 Wyckoff 信息。
    """
    report = []

    try:
        sga, labels = site_wyckoff_labels(structure, symprec=symprec)

        for i, site in enumerate(structure.sites):
            report.append(
                {
                    "index": i,
                    "element": str(site.specie),
                    "wyckoff": labels[i],
                    "frac_coords": np.round(
                        site.frac_coords, 6
                    ).tolist(),
                }
            )

        return {
            "spacegroup": sga.get_space_group_symbol(),
            "spacegroup_number": sga.get_space_group_number(),
            "sites": report,
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "sites": report,
        }


# ------------------------------------------------------------
# 化学计量和电荷补偿
# ------------------------------------------------------------

def calculate_doping_numbers(
    base_structure,
    base_formula,
    target_formula,
    host,
    dopant,
    compensating_element,
    allow_approximation=False,
    stoichiometry_tolerance=0.03,
    multiplier=None,
):
    """
    根据基础化学式和目标化学式计算：

    - host 被替换的原子数；
    - dopant 数；
    - 额外补偿元素数；
    - 实际可达组成与目标组成的偏差。

    要求 target_formula 与扩胞后的 base_structure 具有一致的
    非掺杂元素比例。

    allow_approximation=False 时，任何非整数计数都会直接报错；
    为 True 时按最接近的整数取值，并由 deviation 字段显式给出偏差
    （偏差超过 stoichiometry_tolerance 时仍然报错）。
    """

    nfu = get_formula_units(base_structure, base_formula)

    base = formula_dict(base_formula)
    target = formula_dict(target_formula)

    base_host = base.get(host, 0.0)
    if base_host <= 0:
        raise ValueError(f"基础化学式中没有 host 元素 {host}")

    # host 与 dopant 共享同一套位点：替换前后 host+dopant 的总量应守恒。
    # dopant 也可能本来就存在于基体中（例如在 Li6PS5Cl 里用 Cl 取代 S，
    # 而 Cl 本就是基体元素），此时只比较 base[host] 会把基体自带的
    # dopant 漏掉，故比较 host+dopant 的总量。
    base_pool = base.get(host, 0.0) + base.get(dopant, 0.0)
    target_pool = target.get(host, 0.0) + target.get(dopant, 0.0)
    if abs(target_pool - base_pool) > 1e-6:
        raise ValueError(
            f"目标化学式中 {host}+{dopant} != 基础结构中的 "
            f"{host}+{dopant}: {target_pool} vs {base_pool}"
        )

    # 其余元素（既不是 host/dopant 也不是补偿元素）必须与基础化学式一致，
    # 否则会生成与目标化学式不符的结构。
    untouched_elements = (
        set(base) | set(target)
    ) - {host, dopant, compensating_element}
    for element in sorted(untouched_elements):
        base_amount = base.get(element, 0.0)
        target_amount = target.get(element, 0.0)
        if abs(target_amount - base_amount) > 1e-6:
            raise ValueError(
                f"目标化学式与基础化学式的非掺杂元素 {element} 不一致: "
                f"{target_amount} vs {base_amount}"
            )

    dopant_per_fu = target.get(dopant, 0.0)
    # dopant 是“新增”的量：基体自带的 dopant 不算替换进去的，
    # 否则 dopant 与 host 同为基体元素时 dopant_count 会被高估。
    dopant_added_per_fu = dopant_per_fu - base.get(dopant, 0.0)
    host_target_per_fu = target.get(host, 0.0)

    dopant_count = round_count(
        dopant_added_per_fu * nfu, strict=not allow_approximation
    )
    host_remove_count = round_count(
        (base_host - host_target_per_fu) * nfu,
        strict=not allow_approximation,
    )

    if dopant_count != host_remove_count:
        raise ValueError(
            "host 移除数和 dopant 添加数不一致: "
            f"{host_remove_count} vs {dopant_count}"
        )

    # 计算补偿元素在目标和基础结构中的差值。
    # 注意这里乘的是 nfu（= 每个基胞的 formula unit 数 × 体积倍数），
    # 也就是「每个 f.u. 的调整量 × Z × 体积」，漏掉 Z 会得到错误的数量。
    base_comp = base.get(compensating_element, 0.0)
    target_comp = target.get(compensating_element, 0.0)

    extra_comp_count = round_count(
        (target_comp - base_comp) * nfu,
        strict=not allow_approximation,
    )

    # 实际可达组成：按脚本真正会构建的原子数回算，显式暴露近似带来的偏差。
    # 各项贡献是相加的（基体 + 替换 + 插入），而不是互斥分支，
    # 否则 dopant 与补偿元素相同时会漏算插入的那部分。
    achieved_counts = {}
    for element in sorted(set(base) | set(target)):
        achieved = base.get(element, 0.0) * nfu
        if element == host:
            achieved -= host_remove_count
        if element == dopant:
            achieved += dopant_count
        if element == compensating_element:
            achieved += extra_comp_count
        if achieved > 0:
            achieved_counts[element] = achieved

    deviation = composition_deviation(
        base_formula,
        target_formula,
        achieved_counts,
        nfu,
    )

    # 取代比例偏差：整数化对「有几个位点可替换」的依赖性更强，
    # 因此和原子分数偏差一起作为近似是否可接受的判据。
    # 取代比例用「新增的 dopant / host 位点数」，与 dopant_count 一致
    requested_ratio = dopant_added_per_fu / base_host
    actual_ratio = dopant_count / (base_host * nfu)
    doping_ratio = {
        "requested": requested_ratio,
        "actual": actual_ratio,
        "deviation": actual_ratio - requested_ratio,
        "deviation_percent": (actual_ratio - requested_ratio) * 100.0,
    }
    max_deviation = max(
        deviation["max_deviation"], abs(doping_ratio["deviation"])
    )

    if max_deviation > stoichiometry_tolerance:
        hint = (
            "可用更大的 --scaling（分母 LCM）换取精度，"
            "或放宽 --stoichiometry-tolerance"
        )
        raise ValueError(
            f"整数化偏差过大：取代比例 {requested_ratio:.4f} -> "
            f"{actual_ratio:.4f}（{doping_ratio['deviation_percent']:+.2f}%），"
            f"组成最大偏差 {deviation['max_deviation_percent']:.3f}%，"
            f"超过容差 {stoichiometry_tolerance * 100:.2f}%；{hint}"
        )

    return {
        "formula_units": nfu,
        "dopant_count": dopant_count,
        "host_remove_count": host_remove_count,
        "extra_compensating_atoms": extra_comp_count,
        "base_counts": base,
        "target_counts": target,
        "achieved_counts": achieved_counts,
        "deviation": deviation,
        "doping_ratio": doping_ratio,
        "max_deviation": max_deviation,
        "max_deviation_percent": max_deviation * 100.0,
        "approx_volume_multiplier": multiplier,
    }


# ------------------------------------------------------------
# 几何间隙位点搜索
# ------------------------------------------------------------

def generate_random_interstitial_candidates(
    structure,
    n_samples=30000,
    min_distance=1.8,
    seed=1234,
    max_batch_elements=2_000_000,
):
    """
    随机分数坐标搜索几何间隙位点。

    这里的 min_distance 是候选点到已有原子的最小距离。
    对 Li 插入而言，1.8 Å 只是初筛阈值，不是最终物理判据。

    距离计算按批次向量化，避免在 Python 循环里逐个调用
    get_all_distances（大超胞下会非常慢）。

    返回候选（按 clearance 降序）：
        {
            "frac_coords": [...],
            "clearance": ...
        }
    """
    if n_samples <= 0:
        return []

    rng = np.random.default_rng(seed)
    frac_all = rng.random((n_samples, 3))

    reference = structure.frac_coords
    n_ref = max(len(reference), 1)
    batch_size = max(1, int(max_batch_elements // n_ref))

    accepted_frac = []
    accepted_clearance = []

    for start in range(0, n_samples, batch_size):
        batch = frac_all[start:start + batch_size]
        distance_matrix = structure.lattice.get_all_distances(
            batch,
            reference,
        )
        clearances = distance_matrix.min(axis=1)
        mask = clearances >= min_distance

        if np.any(mask):
            accepted_frac.append(batch[mask])
            accepted_clearance.append(clearances[mask])

    if accepted_frac:
        fracs = np.concatenate(accepted_frac, axis=0)
        clearances = np.concatenate(accepted_clearance, axis=0)
    else:
        fracs = np.empty((0, 3))
        clearances = np.empty((0,))

    order = np.argsort(-clearances)

    candidates = [
        {
            "frac_coords": fracs[i].tolist(),
            "clearance": float(clearances[i]),
        }
        for i in order
    ]

    print(
        f"[Interstitial] 从 {n_samples} 个采样点中找到 "
        f"{len(candidates)} 个候选，阈值={min_distance:.2f} Å"
    )

    if len(candidates) > 0:
        print(
            f"[Interstitial] 最大 clearance="
            f"{candidates[0]['clearance']:.3f} Å"
        )

    return candidates


def build_pool_distance_matrix(structure, pool):
    """
    预计算候选池内部的最小镜像距离矩阵。

    候选池在所有候选构型之间是固定的，因此该矩阵只需算一次，
    不必在每个候选的贪心选择里重复调用
    lattice.get_distance_and_image。
    """
    if len(pool) == 0:
        return np.zeros((0, 0))

    fracs = np.array([item["frac_coords"] for item in pool])
    return structure.lattice.get_all_distances(fracs, fracs)


def select_interstitial_sites(
    structure,
    candidates,
    n_sites,
    min_site_site_distance=2.0,
    pool_size=200,
    seed=1234,
    distance_matrix=None,
):
    """
    从候选位点中选择多个互相不太接近的插入点。
    使用简单贪心策略：先取 clearance 最大的前 pool_size 个候选，
    再随机打乱顺序后贪心挑选。

    distance_matrix 可选：由 build_pool_distance_matrix 预先算好的
    「候选池 × 候选池」最小镜像距离矩阵，多个候选之间可复用。

    后续可替换为：
    - Ewald 优化；
    - 整数规划；
    - cluster expansion；
    - agent 调用外部搜索器。
    """

    if n_sites <= 0:
        return []

    pool = list(candidates[:pool_size] if pool_size else candidates)
    if len(pool) < n_sites:
        raise RuntimeError(
            f"候选池只有 {len(pool)} 个位点，少于需要的 {n_sites} 个"
        )

    if (
        distance_matrix is None
        or distance_matrix.shape[0] != len(pool)
        or distance_matrix.shape[1] != len(pool)
    ):
        distance_matrix = build_pool_distance_matrix(structure, pool)

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)

    selected_positions = []

    for position in order:
        if all(
            distance_matrix[position, other] >= min_site_site_distance
            for other in selected_positions
        ):
            selected_positions.append(position)

        if len(selected_positions) >= n_sites:
            break

    if len(selected_positions) < n_sites:
        raise RuntimeError(
            f"只找到 {len(selected_positions)} 个互相分离的插入位点，"
            f"需要 {n_sites} 个。可尝试减小 "
            f"--min-site-site-distance 或增大 --candidate-pool-size"
        )

    return [pool[position] for position in selected_positions]


# ------------------------------------------------------------
# 替换位点选择
# ------------------------------------------------------------

def choose_substitution_sites(
    structure,
    host,
    n_substitutions,
    seed=1234,
    allowed_indices=None,
    sampler="greedy",
):
    """
    选择 host 原子作为 substitution 位点。

    sampler：
    - greedy（默认）：随机打乱后，贪心让 dopant 彼此距离较远。适合极稀掺杂
      （k=1 或 k 很小），避免掺杂原子聚在一起。
    - random：均匀随机取 k 个位点，不追求分散。适合取代比例不低、需要遍历
      不同构型的情况——贪心「最大间距」在对称晶格上解空间极小（例如 8 个
      候选位取 6 个，贪心几乎只会给出 1-2 种组合）。

    allowed_indices 可选：只在给定的结构索引里选（例如用 --host-wyckoff
    或 --host-site-filter 限定可取代位点）。

    后续可替换为：
    - Ewald 最小化；
    - SQS；
    - cluster expansion；
    - DFT 预筛选。
    """

    if n_substitutions <= 0:
        return []

    if allowed_indices is None:
        host_indices = [
            i for i, site in enumerate(structure.sites)
            if str(site.specie) == host
        ]
    else:
        host_indices = [
            i for i in allowed_indices
            if str(structure[i].specie) == host
        ]

    if len(host_indices) < n_substitutions:
        raise ValueError(
            f"host={host} 原子数不足："
            f"{len(host_indices)} < {n_substitutions}"
        )

    rng = random.Random(seed)
    if sampler == "random":
        return sorted(rng.sample(host_indices, n_substitutions))

    rng.shuffle(host_indices)

    position_of = {
        index: position for position, index in enumerate(host_indices)
    }
    host_fracs = np.array(
        [structure[index].frac_coords for index in host_indices]
    )
    lattice = structure.lattice

    # 第一个随机选择
    selected = [host_indices[0]]

    # min_distance[j] = 第 j 个 host 原子到「已选集合」的最小距离，
    # 每选出一个新位点后用一次向量化距离计算增量更新，
    # 避免每个候选都做 O(n_host × n_sub²) 次逐点距离调用。
    min_distance = lattice.get_all_distances(
        host_fracs,
        host_fracs[[position_of[selected[0]]]],
    ).ravel()

    remaining = host_indices[1:]

    while len(selected) < n_substitutions:
        best_index = max(
            remaining,
            key=lambda index: min_distance[position_of[index]],
        )
        selected.append(best_index)
        remaining.remove(best_index)

        new_distances = lattice.get_all_distances(
            host_fracs,
            host_fracs[[position_of[best_index]]],
        ).ravel()
        min_distance = np.minimum(min_distance, new_distances)

    return selected


def apply_substitution(structure, indices, new_element):
    """
    对指定原子执行元素替换。
    """
    structure = structure.copy()

    for index in indices:
        structure[index] = new_element

    return structure


def remove_sites_by_index(structure, indices):
    """
    按结构索引移除原子（用于「补偿元素需要减少」的场景，例如
    Cl 取代 S、同时减少 Li 以平衡电荷）。
    """
    structure = structure.copy()
    structure.remove_sites(sorted({int(i) for i in indices}))
    return structure


def choose_removal_sites(structure, element, n_removals, seed=1234):
    """
    选择要移除的 element 原子（贪心让被移除的位点尽量分散）。

    与 choose_substitution_sites 的策略一致，只是方向相反。
    """
    if n_removals <= 0:
        return []

    indices = [
        i for i, site in enumerate(structure.sites)
        if str(site.specie) == element
    ]
    if len(indices) < n_removals:
        raise ValueError(
            f"{element} 原子数不足，无法移除 {n_removals} 个："
            f"只有 {len(indices)} 个"
        )

    rng = random.Random(seed)
    pool = list(indices)
    rng.shuffle(pool)

    fracs = np.array([structure[i].frac_coords for i in pool])
    lattice = structure.lattice
    position_of = {index: pos for pos, index in enumerate(pool)}

    selected = [pool[0]]
    min_distance = lattice.get_all_distances(fracs, fracs[[0]]).ravel()
    remaining = pool[1:]

    while len(selected) < n_removals:
        best_index = max(
            remaining, key=lambda index: min_distance[position_of[index]]
        )
        selected.append(best_index)
        remaining.remove(best_index)
        new_distances = lattice.get_all_distances(
            fracs, fracs[[position_of[best_index]]]
        ).ravel()
        min_distance = np.minimum(min_distance, new_distances)

    return selected


def host_site_indices_for_substitution(
    primitive,
    supercell,
    host,
    wyckoff_labels=None,
):
    """
    返回 supercell 中可被取代的 host 原子索引。

    wyckoff_labels 为空时返回全部 host 原子；否则只保留「在基体原胞里
    属于指定 Wyckoff 轨道」的 host 原子。Wyckoff 标签在基体原胞
    (primitive) 上计算，再按分数坐标映射回超胞，因此与超胞自身
    降低后的对称性无关（超胞的 Wyckoff 字母会变，不能用）。

    这样可以把取代限制到化学上正确的位点，例如银锗矿 Li6PS5Cl 里
    Cl 应取代自由 S²⁻（原胞 1c），而不是 PS4 的结构硫（4e）。
    """
    host_indices = [
        i for i, site in enumerate(supercell.sites)
        if str(site.specie) == host
    ]
    if not wyckoff_labels:
        return host_indices

    _, prim_labels = site_wyckoff_labels(primitive, symprec=1e-3)
    wanted = set(wyckoff_labels)
    prim_frac = np.array([site.frac_coords for site in primitive.sites])

    result = []
    for i in host_indices:
        cart = supercell.lattice.get_cartesian_coords(
            supercell[i].frac_coords
        )
        frac = primitive.lattice.get_fractional_coords(cart)
        delta = prim_frac - frac
        delta -= np.round(delta)  # 最小镜像
        nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
        if prim_labels[nearest] in wanted:
            result.append(i)

    return result


def free_anion_site_indices(
    structure,
    host,
    framework_element="P",
    cutoff=3.0,
):
    """
    返回 structure 中「自由阴离子」host 原子的索引。

    判据是几何的：自由阴离子（银锗矿的 4a/4c 位）不参与四面体，
    到骨架元素（默认 P）的最近距离约 4.2-4.8 Å；而 PS4 结构硫
    的 P-S 键约 2.0-2.1 Å。到骨架元素最近距离 > cutoff 的 host
    原子即为自由位点。

    用最近距离而不是 Wyckoff 标签，是因为低对称（P1）的有序胞里
    所有原子都是 1a，Wyckoff 无法区分；距离判据对这种胞同样有效。

    framework_element 可以是逗号分隔的多个元素（如 "P,Si"）。这在
    以已掺杂结构为基体时是必须的：Si 占据 P 位后，SiS4 的硫没有 P
    近邻，只用 P 判断会被误认成自由硫。
    """
    if isinstance(framework_element, str):
        framework_elements = [
            token.strip()
            for token in framework_element.split(",")
            if token.strip()
        ]
    else:
        framework_elements = list(framework_element)

    framework_indices = [
        i for i, site in enumerate(structure.sites)
        if str(site.specie) in framework_elements
    ]
    if not framework_indices:
        raise ValueError(
            "结构中没有任何骨架元素 "
            f"{framework_elements}，无法判定自由阴离子位点"
        )

    frac = np.asarray(structure.frac_coords)
    frame_frac = frac[framework_indices]
    nearest = structure.lattice.get_all_distances(frac, frame_frac).min(axis=1)

    return [
        i for i, site in enumerate(structure.sites)
        if str(site.specie) == host and nearest[i] > cutoff
    ]


def add_interstitial_atoms(structure, element, selected_sites):
    """
    添加间隙原子。
    """
    structure = structure.copy()

    for item in selected_sites:
        structure.append(
            species=element,
            coords=item["frac_coords"],
            coords_are_cartesian=False,
        )

    return structure


# ------------------------------------------------------------
# MatterSim 单点筛选与批量弛豫
# ------------------------------------------------------------

class MattersimUnavailable(RuntimeError):
    """
    mattersim 不可用（未安装 / 权重无法加载）。

    这是环境级错误而不是单个材料的数据错误：批处理遇到它应该直接中止，
    而不是把它记成「某个材料失败」继续跑完。
    """


# mattersim 的预置模型名（这些会由 mattersim 自己下载并校验路径）
PRESET_MODELS = {
    "mattersim-v1.0.0-1m",
    "mattersim-v1.0.0-1m.pth",
    "mattersim-v1.0.0-5m",
    "mattersim-v1.0.0-5m.pth",
}


def resolve_model_path(load_path):
    """
    校验 --model。

    mattersim 通过 `torch.load`（pickle）读取 checkpoint，加载不可信文件
    等价于执行其中的任意代码，因此这里只接受两种输入：
    - mattersim 预置模型名（由 mattersim 自己下载）；
    - 本地已存在的 `.pth` 文件，但会明确提示需要可信来源。
    """
    if not load_path:
        return None

    if Path(load_path).name.lower() in PRESET_MODELS:
        return load_path

    path = Path(load_path).expanduser()
    if not path.is_file():
        raise RuntimeError(
            f"--model 指向的文件不存在或不是普通文件: {load_path}"
        )

    print(
        "[MatterSim] 警告: --model 指向本地权重文件 "
        f"{path}；mattersim 使用 torch.load（pickle）加载 checkpoint，"
        "请确认该文件来自可信来源。"
    )
    return str(path)


def build_mattersim_potential(device="auto", load_path=None):
    """
    加载 MatterSim Potential。

    模型只加载一次，筛选（单点能）和弛豫共用同一个 potential，
    与 mattergen 的 LSWO 流程一致。

    load_path 取值：
    - None / "mattersim-v1.0.0-1M"：默认 1M 预训练模型；
    - "mattersim-v1.0.0-5M"：5M 预训练模型；
    - 或者本地已存在的 `.pth` 路径（例如
      ~/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth）。
    """
    try:
        from mattersim.forcefield.potential import Potential
    except ImportError as exc:
        raise MattersimUnavailable(
            "需要 mattersim 才能进行筛选/弛豫。请安装 mattersim，"
            "或使用 --no-relax 只生成结构。"
        ) from exc

    load_path = resolve_model_path(load_path)
    resolved_device = resolve_device(device)
    print(
        f"[MatterSim] device={resolved_device}, "
        f"load_path={load_path or '默认预训练模型'}"
    )

    kwargs = {
        "device": resolved_device,
        "load_training_state": False,
    }
    if load_path:
        kwargs["load_path"] = load_path

    return Potential.from_checkpoint(**kwargs)


def screen_structures(potential, structures, batch_size=8):
    """
    批量单点能筛选：返回与 structures 等长的 (energy, energy_per_atom)。

    按 batch_size 分块计算并及时释放中间对象，
    避免大 candidate 集合把内存吃满。
    """
    from mattersim.datasets.utils.build import build_dataloader
    from pymatgen.io.ase import AseAtomsAdaptor

    if batch_size <= 0:
        batch_size = len(structures) or 1

    results = []

    for start in range(0, len(structures), batch_size):
        chunk = structures[start:start + batch_size]
        atoms = [AseAtomsAdaptor.get_atoms(s) for s in chunk]

        loader = build_dataloader(
            atoms,
            batch_size=len(atoms),
            only_inference=True,
        )
        energies, _, _ = potential.predict_properties(loader)

        for structure, energy in zip(chunk, energies):
            energy = float(energy)
            results.append((energy, energy / len(structure)))

        del loader, atoms, energies
        gc.collect()

    return results


def make_step_tracked_relaxer(
    potential,
    fmax,
    steps,
    relax_cell,
    optimizer,
    max_natoms_per_batch,
    timeout=0.0,
    progress_every=25,
    label="",
):
    """
    构造带「步数上限 / 收敛 / 耗时」统计的 BatchRelaxer。

    BatchRelaxer 本身只负责把结构塞进批次里跑 FIRE/BFGS，
    不暴露每个结构的步数和是否被截断；这里通过包装
    insert/step_batch 补齐这些信息，用法与 mattergen
    LSWO 流程里的 MatterSimRanker 相同。

    另外补了两个保护：
    - 单个结构原子数超过 max_natoms_per_batch 时直接报错
      （否则 BatchRelaxer 会空转）；
    - 达到 steps 或 timeout 的结构会从优化器列表中移除，
      并在结果中标记为「未收敛」。
    """
    from mattersim.applications.batch_relax import BatchRelaxer

    class StepTrackedBatchRelaxer(BatchRelaxer):
        def __init__(self, *relaxer_args, **relaxer_kwargs):
            super().__init__(*relaxer_args, **relaxer_kwargs)
            self.max_steps = steps
            self.max_seconds = timeout
            self.progress_every = progress_every
            self.batch_steps = 0
            self.step_counts = {}
            self.start_times = {}
            self.elapsed_seconds = {}
            self.converged_indices = set()
            self.capped_indices = set()
            self.timed_out_indices = set()

        def insert(self, atoms):
            super().insert(atoms)
            index = int(atoms.info["structure_index"])
            self.step_counts.setdefault(index, 0)
            self.start_times.setdefault(index, time.monotonic())

        def step_batch(self):
            active_before = {
                int(opt.atoms.info["structure_index"])
                for opt in self.optimizer_instances
            }

            super().step_batch()
            self.batch_steps += 1

            now = time.monotonic()
            for index in active_before:
                self.step_counts[index] += 1

            remaining = []
            for optimizer_instance in self.optimizer_instances:
                index = int(optimizer_instance.atoms.info["structure_index"])

                timed_out = (
                    self.max_seconds > 0
                    and now - self.start_times[index] >= self.max_seconds
                )
                if timed_out:
                    self.timed_out_indices.add(index)

                if self.step_counts[index] >= self.max_steps or timed_out:
                    self.capped_indices.add(index)
                    self.elapsed_seconds[index] = now - self.start_times[index]
                else:
                    remaining.append(optimizer_instance)

            remaining_indices = {
                int(opt.atoms.info["structure_index"]) for opt in remaining
            }
            for index in active_before - remaining_indices:
                if index in self.capped_indices or index in self.timed_out_indices:
                    continue
                self.converged_indices.add(index)
                self.elapsed_seconds[index] = now - self.start_times[index]

            self.optimizer_instances = remaining
            self.is_active_instance = [True] * len(remaining)
            self.finished = not remaining

            if self.progress_every and self.step_counts:
                done = len(self.converged_indices) + len(self.capped_indices)
                if self.batch_steps % self.progress_every == 0:
                    print(
                        f"[MatterSim{' ' + label if label else ''}] "
                        f"batch={self.batch_steps} "
                        f"active={len(remaining)} done={done} "
                        f"max_steps={max(self.step_counts.values())}",
                        flush=True,
                    )

        def relax_atoms(self, atoms_list):
            self.step_counts = {}
            self.start_times = {}
            self.elapsed_seconds = {}
            self.converged_indices = set()
            self.capped_indices = set()
            self.timed_out_indices = set()
            self.total_converged = 0
            self.finished = False
            self.batch_steps = 0
            return super().relax(atoms_list)

    relaxer = StepTrackedBatchRelaxer(
        potential=potential,
        optimizer=optimizer,
        # 必须传字符串：mattersim 1.1.2 用 ase.constraints.Filter 做
        # issubclass 校验，直接传 ase.filters.ExpCellFilter 会被拒绝
        # （ValueError: Unsupported filter）。
        filter="EXPCELLFILTER" if relax_cell else None,
        fmax=fmax,
        max_natoms_per_batch=max_natoms_per_batch,
    )

    return relaxer


def relax_structures(
    potential,
    structures,
    fmax=0.1,
    steps=500,
    relax_cell=False,
    optimizer="FIRE",
    max_natoms_per_batch=128,
    timeout=0.0,
    label="",
):
    """
    批量弛豫结构，返回与 structures 等长的结果字典列表：

        {
            "structure": Structure | None,
            "relaxed": bool,
            "converged": bool,
            "energy": float | None,
            "energy_per_atom": float | None,
            "steps": int | None,
            "seconds": float | None,
            "error": str,
        }
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    def failure(message):
        return {
            "structure": None,
            "relaxed": False,
            "converged": False,
            "energy": None,
            "energy_per_atom": None,
            "steps": None,
            "seconds": None,
            "error": message,
        }

    oversized = [
        i for i, structure in enumerate(structures)
        if len(structure) > max_natoms_per_batch
    ]
    if oversized:
        raise RuntimeError(
            f"以下结构的原子数超过 --relax-natoms-per-batch="
            f"{max_natoms_per_batch}: {oversized}，请调大该参数"
        )

    atoms_list = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    try:
        # 构造 relaxer 也可能抛错（例如 mattersim 版本不接受某个 filter），
        # 放在 try 内可以让该分块降级为「失败」而不是中止整个运行。
        relaxer = make_step_tracked_relaxer(
            potential=potential,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            optimizer=optimizer,
            max_natoms_per_batch=max_natoms_per_batch,
            timeout=timeout,
            label=label,
        )
        trajectories = relaxer.relax_atoms(atoms_list)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[MatterSim] 批量弛豫失败: {message}")
        return [failure(message) for _ in structures]

    results = []
    for index, structure in enumerate(structures):
        try:
            final_atoms = trajectories[index][-1]
            relaxed_structure = AseAtomsAdaptor.get_structure(final_atoms)
            energy = float(final_atoms.info["total_energy"])

            if index in relaxer.timed_out_indices:
                error = f"未在 {timeout:g} 秒内收敛"
            elif index in relaxer.capped_indices:
                error = f"未在 {steps} 步内收敛"
            else:
                error = ""

            results.append(
                {
                    "structure": relaxed_structure,
                    "relaxed": True,
                    "converged": index in relaxer.converged_indices,
                    "energy": energy,
                    "energy_per_atom": energy / len(relaxed_structure),
                    "steps": relaxer.step_counts.get(index),
                    "seconds": relaxer.elapsed_seconds.get(index),
                    "error": error,
                }
            )
        except Exception as exc:
            results.append(
                failure(f"{type(exc).__name__}: {exc}")
            )

    return results


# ------------------------------------------------------------
# 落盘、断点续跑
# ------------------------------------------------------------

def write_cif_atomic(structure, path):
    """
    原子写 CIF：先写临时文件再替换，避免中断留下半个文件。

    所有最终产物（候选、最优结构、rank 列表）都走这里。
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    structure.to(filename=str(temporary))
    temporary.replace(path)
    return path


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 候选生成算法的版本号。生成逻辑本身变化时（即使参数没变）也必须
# 让旧候选失效，因此把它并入候选指纹。
CANDIDATE_FORMAT_VERSION = 3

# 筛选/弛豫记录的版本号，作用同上。
ML_FORMAT_VERSION = 2


def run_fingerprint(args, scaling, numbers, base_identity=""):
    """
    候选指纹：只要它不变，就复用上次生成的候选结构 CIF。

    必须覆盖所有会改变候选几何的输入，包括从 Materials Project
    实际拿到的基体结构（base_identity），否则 MP 端换了基体结构时
    会静默复用旧结构的候选。
    """
    payload = {
        "base_formula": args.base_formula,
        "target_formula": args.target_formula,
        "host": args.host,
        "dopant": args.dopant,
        "compensating_element": args.compensating_element,
        "host_wyckoff": args.host_wyckoff,
        "scaling": list(scaling),
        "seed": args.seed,
        "n_candidates": args.n_candidates,
        "n_samples": args.n_samples,
        "min_distance": args.min_distance,
        "min_site_site_distance": args.min_site_site_distance,
        "candidate_pool_size": args.candidate_pool_size,
        "n_dopant": numbers["dopant_count"],
        "n_extra": numbers["extra_compensating_atoms"],
        "base_identity": str(base_identity),
        "candidate_format_version": CANDIDATE_FORMAT_VERSION,
    }
    # 位点筛选只在启用时进入指纹：这样默认（all）运行的指纹保持不变，
    # 已有的候选不会被无谓地作废（0.01 级金属目标的 1240 原子胞重生成很贵）。
    if args.host_site_filter != "all":
        payload["host_site_filter"] = args.host_site_filter
        payload["host_framework_element"] = args.host_framework_element
        payload["host_framework_cutoff"] = args.host_framework_cutoff
    if args.substitution_sampler != "greedy":
        payload["substitution_sampler"] = args.substitution_sampler
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def model_identity(load_path):
    """
    模型指纹用的身份串。

    - 预置模型名：直接用名字（版本写在名字里，天然稳定）；
    - 本地 .pth：用「路径 + 大小 + mtime」，这样同一个路径下的权重
      被替换（重新下载/换权重）后 ML 指纹会自动失效，不会复用旧能量。
    """
    if not load_path:
        return "default"
    if Path(load_path).name.lower() in PRESET_MODELS:
        return str(load_path)
    path = Path(load_path).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return str(path)
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def ml_fingerprint(args, candidate_fingerprint):
    """
    筛选/弛豫结果的指纹。

    候选几何没变但 ML 设置变了（例如 --model 1M → 5M、收紧 --fmax），
    旧能量就不再有效；该指纹会写进 screening.csv 和每个候选的
    relaxed json，加载时逐条校验，因此不需要靠删除文件来失效。
    """
    payload = {
        "candidate_fingerprint": candidate_fingerprint,
        "model": model_identity(args.model),
        "fmax": args.fmax,
        "relax_steps": args.relax_steps,
        "relax_cell": bool(args.relax_cell),
        "relax_optimizer": args.relax_optimizer,
        "relax_natoms_per_batch": args.relax_natoms_per_batch,
        "screen_batch_size": args.screen_batch_size,
        "relax_timeout": args.relax_timeout,
        "ml_format_version": ML_FORMAT_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def generate_candidates(
    supercell,
    args,
    n_dopant,
    n_extra,
    interstitial_pool,
    candidate_dir,
    fingerprint,
    pool_distance_matrix=None,
    host_substitution_indices=None,
):
    """
    生成 n_candidates 个掺杂候选结构并落盘。

    每个候选：
    - 结构写入 candidate_dir/candidate_%04d.cif；
    - 替换位点 / 插入位点 / 移除位点记录在 candidates_meta.json。

    n_extra > 0：插入补偿元素（原有行为）；
    n_extra < 0：从结构里移除 |n_extra| 个补偿元素（例如掺 Cl 取代 S
                 时减少 Li 以平衡电荷）。

    指纹一致时只复用已有 cif（缺少的单个文件按同一确定性算法补生成），
    不会因为个别文件缺失就丢弃全部候选；指纹变化时才整批重新生成。
    旧候选文件不会被删除，而是被新指纹的结果覆盖。

    候选按「位点顺序无关」的结构指纹判重，同一组位点只保留一个；位点组合
    枚举完后（连续多次抽到重复）自动提前结束，因此实际候选数可能少于
    --n-candidates（会打印提示）。
    """
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    meta_file = candidate_dir / "candidates_meta.json"

    existing = {}
    fingerprint_matches = False

    if meta_file.exists():
        meta = read_json(meta_file)
        fingerprint_matches = meta.get("fingerprint") == fingerprint
        if fingerprint_matches:
            existing = {
                int(item["candidate_id"]): item
                for item in meta.get("candidates", [])
            }

    if not fingerprint_matches:
        if meta_file.exists():
            print(
                "[Candidates] 候选指纹已变化，重新生成候选结构"
                f"（新 fingerprint={fingerprint}）"
            )
        existing = {}
    elif len(existing) == args.n_candidates:
        print(
            f"[Candidates] 复用已有 {len(existing)} 个候选结构"
            f"（断点续跑，fingerprint={fingerprint}）"
        )
    else:
        print(
            f"[Candidates] 候选指纹一致，复用 {len(existing)} 个已有结构，"
            "并补生成缺失的候选"
        )

    records = []
    generated = 0
    reused = 0
    seen_keys = set()

    # 第一步：指纹一致时按原样复用已有候选（断点续跑）。
    pending_slots = []
    for candidate_id in range(args.n_candidates):
        file_name = f"candidate_{candidate_id:04d}.cif"
        path = candidate_dir / file_name
        item = existing.get(candidate_id)

        if item is None or not path.exists():
            pending_slots.append(candidate_id)
            continue

        try:
            structure = Structure.from_file(str(path))
        except Exception as exc:
            print(f"[Candidates] 读取 {file_name} 失败，重新生成: {exc}")
            pending_slots.append(candidate_id)
            continue

        seen_keys.add(candidate_geometry_key(structure))
        records.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "structure": structure,
                "substitution_indices": item["substitution_indices"],
                "interstitial_sites": item["interstitial_sites"],
                "removed_indices": item.get("removed_indices", []),
            }
        )
        reused += 1

    # 第二步：为剩余槽位生成新候选。
    # 判重按「位点顺序无关」的结构指纹；同一组位点只会保留一次。位点组合
    # 枚举完后（连续多次抽到重复）自动结束，不再输出重复候选。
    max_duplicate_streak = max(50, 2 * args.n_candidates)

    for candidate_id in pending_slots:
        file_name = f"candidate_{candidate_id:04d}.cif"
        path = candidate_dir / file_name
        accepted = False
        duplicate_streak = 0

        for retry in range(max_duplicate_streak + 1):
            # retry=0 时与旧版一致：seed = args.seed + candidate_id
            seed = args.seed + candidate_id + retry * args.n_candidates
            try:
                substitution_indices = choose_substitution_sites(
                    structure=supercell,
                    host=args.host,
                    n_substitutions=n_dopant,
                    seed=seed,
                    allowed_indices=host_substitution_indices,
                    sampler=args.substitution_sampler,
                )
                if n_extra > 0:
                    selected_interstitials = select_interstitial_sites(
                        structure=supercell,
                        candidates=interstitial_pool,
                        n_sites=n_extra,
                        min_site_site_distance=args.min_site_site_distance,
                        pool_size=args.candidate_pool_size,
                        seed=seed,
                        distance_matrix=pool_distance_matrix,
                    )
                    removed_indices = []
                else:
                    selected_interstitials = []
                    removed_indices = choose_removal_sites(
                        structure=supercell,
                        element=args.compensating_element,
                        n_removals=-n_extra,
                        seed=seed,
                    )
            except Exception as exc:
                # 单个构型失败不应中断整个搜索
                print(
                    f"[Candidates] candidate {candidate_id} 生成失败，"
                    f"跳过: {exc}"
                )
                continue

            doped = apply_substitution(
                structure=supercell,
                indices=substitution_indices,
                new_element=args.dopant,
            )
            if selected_interstitials:
                doped = add_interstitial_atoms(
                    structure=doped,
                    element=args.compensating_element,
                    selected_sites=selected_interstitials,
                )
            if removed_indices:
                doped = remove_sites_by_index(doped, removed_indices)

            key = candidate_geometry_key(doped)
            if key in seen_keys:
                duplicate_streak += 1
                if duplicate_streak >= max_duplicate_streak:
                    break
                continue

            seen_keys.add(key)
            write_cif_atomic(doped, path)
            generated += 1
            records.append(
                {
                    "candidate_id": candidate_id,
                    "file": file_name,
                    "structure": doped,
                    "substitution_indices": substitution_indices,
                    "interstitial_sites": selected_interstitials,
                    "removed_indices": removed_indices,
                }
            )
            accepted = True
            break

        if not accepted:
            print(
                "[Candidates] 无法再生成新的不同候选（位点组合已枚举完，或该槽位"
                "反复生成失败），提前结束："
                f"去重后共 {len(records)} 个不同候选，少于 --n-candidates "
                f"{args.n_candidates}。candidates/ 里可能残留旧指纹的 "
                "candidate_*.cif，以下游使用的 candidates_meta.json 为准"
            )
            break

    if not records:
        raise RuntimeError("没有成功生成任何候选结构，请检查输入参数")

    if generated:
        print(
            f"[Candidates] 新生成 {generated} 个候选结构"
            f"（复用 {reused} 个，去重后共 {len(records)} 个）"
        )

    write_json_atomic(
        meta_file,
        {
            "fingerprint": fingerprint,
            "candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "file": item["file"],
                    "substitution_indices": item["substitution_indices"],
                    "interstitial_sites": item["interstitial_sites"],
                    "removed_indices": item.get("removed_indices", []),
                }
                for item in records
            ],
        },
    )

    return records


def load_screening(path, fingerprint):
    """
    读取单点能筛选结果（断点续跑）。

    只接受 fingerprint 与当前 ML 设置一致的行，因此改过 --model /
    --fmax 等参数后旧能量会被自动忽略并重算，无需删除文件。
    """
    path = Path(path)
    if not path.exists():
        return {}

    records = {}
    stale = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("fingerprint", "") != fingerprint:
                stale += 1
                continue
            index = int(row["candidate_id"])
            records[index] = {
                "candidate_id": index,
                "energy": float(row["energy_ev"]),
                "energy_per_atom": float(row["energy_per_atom_ev"]),
            }

    if stale:
        print(
            f"[Screen] 忽略 {stale} 条与当前 ML 设置不匹配的旧筛选结果"
        )
    return records


def write_screening(path, records, fingerprint):
    """
    写入单点能结果（原子写）。

    与旧版本不同，这里会**保留其它 fingerprint 的旧行**：改过
    --model / --fmax 等设置再改回来时，之前的单点能仍然可用，
    不必整批重算。合并键是 (fingerprint, candidate_id)。
    """
    path = Path(path)
    fields = ["candidate_id", "energy_ev", "energy_per_atom_ev", "fingerprint"]

    merged = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                row_fingerprint = row.get("fingerprint", "")
                if row_fingerprint == fingerprint:
                    # 当前设置的结果下面会用内存里的最新值覆盖
                    continue
                if not row.get("candidate_id"):
                    continue
                merged[(row_fingerprint, int(row["candidate_id"]))] = {
                    "candidate_id": int(row["candidate_id"]),
                    "energy_ev": row.get("energy_ev", ""),
                    "energy_per_atom_ev": row.get(
                        "energy_per_atom_ev", ""
                    ),
                    "fingerprint": row_fingerprint,
                }

    for index in sorted(records):
        merged[(fingerprint, index)] = {
            "candidate_id": index,
            "energy_ev": records[index]["energy"],
            "energy_per_atom_ev": records[index]["energy_per_atom"],
            "fingerprint": fingerprint,
        }

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(merged, key=lambda item: (item[1], item[0])):
            writer.writerow(merged[key])
    temporary.replace(path)


def load_relaxations(indices, relaxed_dir, fingerprint):
    """
    读取已完成的弛豫结果（断点续跑），并把结构读回内存。

    与 load_screening 一样按 fingerprint 过滤，避免复用旧设置算出的
    弛豫结构/能量。
    """
    records = {}
    stale = 0
    for index in indices:
        meta_file = Path(relaxed_dir) / f"candidate_{index:04d}.json"
        cif_file = Path(relaxed_dir) / f"candidate_{index:04d}.cif"
        if not (meta_file.exists() and cif_file.exists()):
            continue

        record = read_json(meta_file)
        if record.get("fingerprint", "") != fingerprint:
            stale += 1
            continue

        record["structure"] = Structure.from_file(str(cif_file))
        records[index] = record

    if stale:
        print(
            f"[Relax] 忽略 {stale} 个与当前 ML 设置不匹配的旧弛豫结果"
        )
    return records


def write_relaxation(record, index, relaxed_dir, fingerprint):
    """
    写入单个候选的弛豫结果（cif + json），用于断点续跑。

    返回写出的 cif 路径（失败时为 None）。
    """
    relaxed_dir = Path(relaxed_dir)
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    structure = record.get("structure")
    if structure is None:
        # 失败的结构不写 checkpoint，下次运行会重试
        return None

    cif_file = relaxed_dir / f"candidate_{index:04d}.cif"
    write_cif_atomic(structure, cif_file)

    payload = {
        "candidate_id": index,
        "converged": bool(record.get("converged", False)),
        "energy": record.get("energy"),
        "energy_per_atom": record.get("energy_per_atom"),
        "steps": record.get("steps"),
        "seconds": record.get("seconds"),
        "error": record.get("error", ""),
        "relaxed_file": str(cif_file),
        "fingerprint": fingerprint,
    }
    write_json_atomic(relaxed_dir / f"candidate_{index:04d}.json", payload)
    return cif_file


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="材料掺杂 + 间隙位点搜索 + MatterSim 弛豫",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-formula",
        required=True,
        help="基础材料化学式，例如 Li6AsS5I",
    )

    parser.add_argument(
        "--target-formula",
        required=True,
        help="目标化学式，例如 Li6.7As0.3Si0.7S5I",
    )

    parser.add_argument(
        "--host",
        required=True,
        help="被替换元素，例如 As",
    )

    parser.add_argument(
        "--dopant",
        required=True,
        help="掺杂元素，例如 Si",
    )

    parser.add_argument(
        "--compensating-element",
        default="Li",
        help=(
            "电荷补偿元素：目标化学式要求增加时插入，要求减少时移除"
            "（例如掺 Cl 取代 S 时用 Li 减少来平衡电荷）"
        ),
    )

    parser.add_argument(
        "--host-wyckoff",
        default=None,
        help=(
            "可选：把取代限制到基体原胞中这些 Wyckoff 轨道上的 host 位点，"
            "逗号分隔，例如 '1c'。不指定则所有 host 原子都可被取代。"
            "用于避免取代到化学上不该动的结构位点"
            "（如 Li6PS5Cl 里 Cl 应取代自由 S²⁻ 而非 PS4 的结构硫）"
        ),
    )

    parser.add_argument(
        "--host-site-filter",
        choices=["all", "free-anion"],
        default="all",
        help=(
            "取代位点筛选：all = 所有 host 原子都可取代；"
            "free-anion = 只允许「自由阴离子」host 位点"
            "（到 --host-framework-element 最近距离 > --host-framework-cutoff），"
            "用于把 O/Se 等取代限制到银锗矿的自由 S²⁻，避免破坏 PS4 四面体。"
            "低对称（P1）有序胞里 Wyckoff 标签不可用，此时只能用该判据"
        ),
    )

    parser.add_argument(
        "--host-framework-element",
        default="P",
        help=(
            "--host-site-filter=free-anion 时的骨架元素（判断 host 是否与之成键）。"
            "可逗号分隔多个，例如 'P,Si'；以已掺杂结构为基体时必须把所有"
            "四面体中心元素都列上"
        ),
    )

    parser.add_argument(
        "--host-framework-cutoff",
        type=float,
        default=3.0,
        help=(
            "--host-site-filter=free-anion 的距离阈值 Å："
            "host 到骨架元素最小距离大于该值才算自由阴离子位点"
        ),
    )

    parser.add_argument(
        "--substitution-sampler",
        choices=["greedy", "random"],
        default="greedy",
        help=(
            "取代位点的抽样方式：greedy = 贪心让掺杂原子尽量分散（适合极稀掺杂，"
            "如 k=1）；random = 均匀随机取 k 个（适合需要遍历不同构型的情况，"
            "贪心在对称晶格上往往只能给出极少数组合）"
        ),
    )

    parser.add_argument(
        "--min-formula-units",
        type=int,
        default=0,
        help=(
            "auto 扩胞时的最小 f.u. 数（0 = 不限制）：把 LCM 体积倍数再乘一个"
            "整数因子把胞放大到至少这么多 f.u.，用于可取代位点太少、"
            "小胞产生不了构型多样性的情况。手动 --scaling 时忽略"
        ),
    )

    parser.add_argument(
        "--base-structure",
        default=None,
        help=(
            "直接指定基体结构文件（CIF/POSCAR 等），"
            "优先级高于 basestru 目录与 Materials Project"
        ),
    )

    parser.add_argument(
        "--basestru-dir",
        default=str(DEFAULT_BASESTRU_DIR),
        help="本地基体结构目录：里面有与 base-formula 匹配的文件时优先使用",
    )

    parser.add_argument(
        "--use-basestru",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否优先使用 basestru 目录中的基体结构（--no-use-basestru 强制用 MP）",
    )

    parser.add_argument(
        "--scaling",
        default="auto",
        help=(
            "超胞矩阵对角元素，例如 1,2,5（体积倍数 = 三分量之积），"
            "或 auto：按目标化学式下标分母的最小公倍数自动定体积倍数"
        ),
    )

    parser.add_argument(
        "--max-supercell-atoms",
        type=int,
        default=500,
        help=(
            "auto 模式下的原子数预算（硬上限）：精确整数化超过该值时"
            "自动改用有理逼近；手工 --scaling 超过该值时仅警告"
        ),
    )

    parser.add_argument(
        "--allow-stoichiometry-approximation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "超出原子数预算时是否自动改用有理逼近"
            "（默认允许；--no-allow-stoichiometry-approximation 则直接报错）"
        ),
    )

    parser.add_argument(
        "--stoichiometry-tolerance",
        type=float,
        default=0.03,
        help="允许的最大原子分数偏差（0.03 = 3%%）",
    )

    parser.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help=(
            "生成并做单点能筛选的候选结构数上限。候选会按结构判重，"
            "位点组合枚举完后自动提前结束，因此实际数量可能更少"
        ),
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=30000,
        help="随机几何间隙采样点数",
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.8,
        help="间隙点到现有原子的初筛最小距离 Å",
    )

    parser.add_argument(
        "--min-site-site-distance",
        type=float,
        default=2.0,
        help="额外插入原子之间的最小距离 Å",
    )

    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=200,
        help="从 clearance 最大的前 N 个候选位点中贪心挑选",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "输出目录。不指定时放在脚本同级目录的 output/ 下，"
            "并按目标化学式分子目录，例如 main/output/Li5.4PS4.4Cl1.6"
        ),
    )

    parser.add_argument(
        "--relax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否用 MatterSim 做筛选与弛豫（--no-relax 只生成结构）",
    )

    parser.add_argument(
        "--relax-cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="弛豫时是否同时优化晶胞",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="弛豫收敛力阈值 eV/Å",
    )

    parser.add_argument(
        "--relax-steps",
        type=int,
        default=500,
        help="弛豫最大步数",
    )

    parser.add_argument(
        "--relax-optimizer",
        default="FIRE",
        choices=["FIRE", "BFGS"],
        help="弛豫优化器",
    )

    parser.add_argument(
        "--n-relax",
        type=int,
        default=None,
        help=(
            "筛选后实际做弛豫的候选数，"
            "默认 min(n-candidates, 20)"
        ),
    )

    parser.add_argument(
        "--screen-batch-size",
        type=int,
        default=8,
        help="单点能筛选的批大小",
    )

    parser.add_argument(
        "--relax-natoms-per-batch",
        type=int,
        default=128,
        help="批量弛豫时每个 batch 的最大原子数",
    )

    parser.add_argument(
        "--relax-chunk-size",
        type=int,
        default=4,
        help="每次批量弛豫的结构数（也是断点续跑的粒度）",
    )

    parser.add_argument(
        "--relax-timeout",
        type=float,
        default=0.0,
        help="单个结构弛豫的墙钟时间上限（秒），0 表示不限制",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "MatterSim 模型：mattersim-v1.0.0-1M / "
            "mattersim-v1.0.0-5M 或本地 .pth 文件路径，"
            "默认使用 1M 预训练模型。注意 .pth 通过 pickle 加载，"
            "必须来自可信来源"
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="筛选/弛豫设备：auto / cpu / cuda / mps",
    )

    parser.add_argument(
        "MP_API",
        default="0G2LB7SliM9T6uvmmSX5iyAh1BB4LqxJ",
        help="MP数据库API，六个月有效"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.n_candidates <= 0:
        raise ValueError("--n-candidates 必须为正整数")

    # host / dopant / compensating-element 必须是三个不同元素：
    # 若 dopant 与补偿元素相同，「替换 + 插入」会让最终元素量翻倍，
    # 而按目标式推导的数量与偏差报告都无法表达这种重叠语义。
    if len(
        {args.host, args.dopant, args.compensating_element}
    ) != 3:
        raise ValueError(
            "host / dopant / compensating-element 必须互不相同，收到 "
            f"{args.host} / {args.dopant} / {args.compensating_element}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        # 默认放在脚本同级目录的 output/ 下，按目标化学式分子目录
        # （例如 output/Li5.4PS4.4Cl1.6）。
        output_dir = DEFAULT_OUTPUT_DIR / formula_dirname(args.target_formula)
    args.output = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 记录本次运行的参数，便于复现。
    # scaling 在 Step 3 才最终确定（auto 模式需要先拿到基胞），届时回填。
    run_config = {k: v for k, v in vars(args).items()}

    # --------------------------------------------------------
    # Step 1: 确定基体结构（优先本地 basestru/，否则回退 MP）
    # --------------------------------------------------------

    (
        base_structure,
        base_source,
        base_file,
        mp_doc,
    ) = resolve_base_structure(args)

    # 本地读取的结构同样要校验组成与 base_formula 一致
    if base_source != "mp":
        formula_units = get_formula_units(base_structure, args.base_formula)
        print(
            f"[BaseStru] source={base_source} "
            f"formula={base_structure.composition.reduced_formula} "
            f"nsites={len(base_structure)} "
            f"formula_units={formula_units}"
        )

    base_material_id = (
        None if mp_doc is None else str(mp_doc.material_id)
    )

    print_symmetry_info(
        base_structure,
        label=f"{base_source} base structure",
    )

    # 保存实际使用的起始结构
    write_cif_atomic(
        base_structure, output_dir / "base_input.cif"
    )

    # --------------------------------------------------------
    # Step 2: 转换到 primitive cell
    # --------------------------------------------------------

    primitive = base_structure.get_primitive_structure()

    print(
        f"\n[Primitive] nsites={len(primitive)}"
    )
    print(
        f"[Primitive] composition={primitive.composition.reduced_formula}"
    )

    print_symmetry_info(
        primitive,
        label="primitive structure",
    )

    write_cif_atomic(
        primitive, output_dir / "base_primitive.cif"
    )

    # --------------------------------------------------------
    # Step 3: 超胞（分数化学计量 -> 整数原子数）
    # --------------------------------------------------------

    explicit_scaling = None
    if str(args.scaling).strip().lower() != "auto":
        explicit_scaling = parse_scaling(args.scaling)

    # 原子数预算是 auto 路径的唯一硬约束：体积倍数与原子数一一对应
    # （make_supercell(np.diag(scaling)) 的行列式即体积倍数），
    # 因此 max_multiplier = 预算 // 基胞原子数。该值同时充当
    # limit_denominator 的分母上限与 LCM 硬上限。
    max_multiplier = max(
        1, args.max_supercell_atoms // max(len(primitive), 1)
    )
    scaling_info = plan_scaling_for_target(
        structure=primitive,
        base_formula=args.base_formula,
        target_formula=args.target_formula,
        max_multiplier=max_multiplier,
        explicit=explicit_scaling,
        min_formula_units=args.min_formula_units,
    )

    inexact = (
        scaling_info["scaling"] is None
        or scaling_info["exact"] is False
    )
    if inexact:
        if not args.allow_stoichiometry_approximation:
            raise ValueError(
                f"{scaling_info['note']}；目标下标的分数分母为 "
                f"{scaling_info['denominators']}。"
                "提高 --max-supercell-atoms（扩大原子数预算）可精确表示，"
                "或按分母 LCM 显式指定 --scaling；"
                "--no-allow-stoichiometry-approximation 下不会自动近似，"
                "去掉该开关即自动改用有理逼近（届时会显式给出"
                "组成与取代比例的偏差）。"
            )
        if scaling_info["scaling"] is None:
            # 精确整数化需要过大的超胞：退到预算允许的最大倍数做有理逼近
            exact_multiplier = scaling_info["multiplier"]
            multiplier = max_multiplier
            scaling_info = {
                "mode": "approximate",
                "scaling": best_diagonal_scaling(primitive, multiplier),
                "multiplier": multiplier,
                "denominators": scaling_info["denominators"],
                "exact": False,
                "note": (
                    f"精确整数化需要体积倍数 {exact_multiplier}"
                    f"（≈{len(primitive) * exact_multiplier} 原子）"
                    f"> --max-supercell-atoms {args.max_supercell_atoms}，"
                    f"自动改用 {multiplier} 倍体积"
                    f"（≈{len(primitive) * multiplier} 原子）做有理逼近；"
                    "具体偏差会在下面显式报告"
                ),
            }
        else:
            scaling_info["mode"] = "approximate"
            scaling_info["note"] += "（有理逼近）"
        print(f"[Stoichiometry] ⚠ {scaling_info['note']}")

    scaling = scaling_info["scaling"]
    run_config["scaling"] = scaling

    print(
        f"\n[Supercell] mode={scaling_info['mode']} "
        f"matrix={'x'.join(str(v) for v in scaling)} "
        f"体积倍数={scaling_info['multiplier']} "
        f"（{scaling_info['note']}）"
    )

    supercell_matrix = np.diag(scaling)
    supercell = primitive.copy()
    supercell.make_supercell(supercell_matrix)

    print(
        f"[Supercell] 基胞 f.u.="
        f"{get_formula_units(primitive, args.base_formula)} "
        f"-> 超胞 f.u.={get_formula_units(supercell, args.base_formula)} "
        f"nsites={len(supercell)}"
    )
    print(
        f"[Supercell] composition="
        f"{supercell.composition.reduced_formula}"
    )

    if len(supercell) > args.max_supercell_atoms:
        print(
            f"[Supercell] ⚠ 警告: 超胞 {len(supercell)} 原子 > "
            f"--max-supercell-atoms {args.max_supercell_atoms}，"
            "已超过原子预算，算力成本会显著上升；"
            "如需自动避免超大胞，请不要指定 --scaling（用 auto）"
        )

    write_cif_atomic(
        supercell, output_dir / "base_supercell.cif"
    )

    # --------------------------------------------------------
    # Step 4: 计算掺杂和补偿数量
    # --------------------------------------------------------

    numbers = calculate_doping_numbers(
        base_structure=supercell,
        base_formula=args.base_formula,
        target_formula=args.target_formula,
        host=args.host,
        dopant=args.dopant,
        compensating_element=args.compensating_element,
        allow_approximation=args.allow_stoichiometry_approximation,
        stoichiometry_tolerance=args.stoichiometry_tolerance,
        multiplier=scaling_info["multiplier"],
    )

    print(
        f"\n[Stoichiometry] 超胞 f.u.={numbers['formula_units']} "
        f"host({args.host}) 移除 {numbers['host_remove_count']} 个，"
        f"{args.dopant} 替换 {numbers['dopant_count']} 个，"
        f"补偿元素 {args.compensating_element} "
        f"{'插入' if numbers['extra_compensating_atoms'] >= 0 else '移除'} "
        f"{abs(numbers['extra_compensating_atoms'])} 个"
    )
    print("[Stoichiometry] 目标 vs 实际可达组成（原子分数）")
    print(
        f"    {'元素':<4}{'目标/f.u.':>12}{'实际/f.u.':>12}"
        f"{'目标分数':>12}{'实际分数':>12}{'偏差':>10}"
    )
    for row in numbers["deviation"]["rows"]:
        print(
            f"    {row['element']:<4}"
            f"{row['target_per_fu']:>12.4f}"
            f"{row['actual_per_fu']:>12.4f}"
            f"{row['target_fraction']:>12.4f}"
            f"{row['actual_fraction']:>12.4f}"
            f"{row['deviation'] * 100:>9.2f}%"
        )
    ratio = numbers["doping_ratio"]
    print(
        f"[Stoichiometry] 取代比例 {args.host}->{args.dopant}: "
        f"目标 {ratio['requested'] * 100:.2f}% -> "
        f"实际 {ratio['actual'] * 100:.2f}%"
        f"（{ratio['deviation_percent']:+.2f}%）"
    )
    print(
        f"[Stoichiometry] 最大偏差 "
        f"{numbers['max_deviation_percent']:.3f}% "
        f"（组成 {numbers['deviation']['max_deviation_percent']:.3f}% + "
        f"取代比例 {abs(ratio['deviation_percent']):.3f}%，"
        f"容差 {args.stoichiometry_tolerance * 100:.2f}%）"
    )

    numbers_summary = dict(numbers)
    numbers_summary.pop("deviation", None)
    print("\n[Composition]")
    print(json.dumps(numbers_summary, indent=2, ensure_ascii=False))

    n_dopant = numbers["dopant_count"]
    n_extra = numbers["extra_compensating_atoms"]

    if n_dopant <= 0:
        raise ValueError(
            "没有计算出正的掺杂数量"
        )

    # n_extra < 0 表示需要「移除」补偿元素（如掺 Cl 取代 S 时减少 Li），
    # 在 generate_candidates 里按 |n_extra| 个位点移除。

    # --------------------------------------------------------
    # Step 5: Wyckoff 信息
    # --------------------------------------------------------

    wyckoff_report = get_wyckoff_report(primitive)

    with open(
        output_dir / "wyckoff_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            wyckoff_report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if "error" in wyckoff_report:
        print(
            f"[Wyckoff] 分析失败（已跳过）: "
            f"{wyckoff_report['error']}"
        )

    # --------------------------------------------------------
    # Step 6: 几何间隙搜索
    # --------------------------------------------------------

    candidates = []

    if n_extra > 0:
        candidates = generate_random_interstitial_candidates(
            structure=supercell,
            n_samples=args.n_samples,
            min_distance=args.min_distance,
            seed=args.seed,
        )

        if len(candidates) < n_extra:
            raise RuntimeError(
                "候选间隙位点数量不足，请降低 --min-distance "
                "或增加 --n-samples"
            )
    else:
        print(
            "[Interstitial] 目标化学式不需要额外插入原子，跳过搜索"
        )

    # --------------------------------------------------------
    # Step 7: 生成候选构型（落盘，支持断点续跑）
    # --------------------------------------------------------

    candidate_fingerprint = run_fingerprint(
        args,
        scaling,
        numbers,
        base_identity=structure_hash(base_structure),
    )
    ml_fp = ml_fingerprint(args, candidate_fingerprint)
    candidate_dir = output_dir / "candidates"
    screening_file = output_dir / "screening.csv"
    relaxed_dir = output_dir / "relaxed"

    # 候选池距离矩阵在所有候选之间复用，只算一次
    pool_distance_matrix = None
    if candidates:
        pool = (
            candidates[:args.candidate_pool_size]
            if args.candidate_pool_size
            else list(candidates)
        )
        pool_distance_matrix = build_pool_distance_matrix(supercell, pool)

    if n_extra > args.candidate_pool_size:
        print(
            f"[Interstitial] ⚠ 需要插入 {n_extra} 个补偿原子，"
            f"但候选池只有 {args.candidate_pool_size} 个位点："
            "请提高 --candidate-pool-size，或减小 --scaling"
            "（超胞越大，需要插入的原子越多）"
        )

    # 可选的取代位点限制：
    # - --host-wyckoff：只保留基体原胞里属于指定 Wyckoff 轨道的 host 位点；
    # - --host-site-filter=free-anion：只保留「自由阴离子」host 位点
    #   （到骨架元素最近距离大于阈值；低对称 P1 胞里 Wyckoff 不可用时的判据）。
    total_host = sum(
        1 for site in supercell.sites if str(site.specie) == args.host
    )
    host_substitution_indices = None

    if args.host_wyckoff:
        wanted = [
            token.strip()
            for token in args.host_wyckoff.split(",")
            if token.strip()
        ]
        host_substitution_indices = host_site_indices_for_substitution(
            primitive=primitive,
            supercell=supercell,
            host=args.host,
            wyckoff_labels=wanted,
        )
        print(
            f"[Substitution] --host-wyckoff={wanted}："
            f"可选 {len(host_substitution_indices)}/{total_host} 个 "
            f"{args.host} 位点"
        )
        if not host_substitution_indices:
            raise ValueError(
                f"--host-wyckoff={args.host_wyckoff} 在超胞里没有匹配的 "
                f"{args.host} 位点；请检查基体原胞的 Wyckoff 标签"
                "（见输出 base_primitive.cif 的对称性分析）"
            )

    if args.host_site_filter == "free-anion":
        free_indices = set(
            free_anion_site_indices(
                structure=supercell,
                host=args.host,
                framework_element=args.host_framework_element,
                cutoff=args.host_framework_cutoff,
            )
        )
        if host_substitution_indices is None:
            host_substitution_indices = [
                i for i, site in enumerate(supercell.sites)
                if str(site.specie) == args.host and i in free_indices
            ]
        else:
            host_substitution_indices = [
                i for i in host_substitution_indices if i in free_indices
            ]
        print(
            f"[Substitution] --host-site-filter=free-anion："
            f"可选 {len(host_substitution_indices)}/{total_host} 个 "
            f"{args.host} 位点（最近 {args.host_framework_element} > "
            f"{args.host_framework_cutoff} Å）"
        )
        if not host_substitution_indices:
            raise ValueError(
                f"--host-site-filter=free-anion 在超胞里没有匹配的 "
                f"{args.host} 位点；请检查 --host-framework-element 与 "
                "--host-framework-cutoff"
            )

    if host_substitution_indices is not None:
        if len(host_substitution_indices) < abs(n_dopant):
            raise ValueError(
                f"位点限定后只剩 {len(host_substitution_indices)} 个可取代位点，"
                f"少于需要的 {abs(n_dopant)} 个"
            )
        # 组合数就是构型多样性的上限；候选数超过它时必然大量重复。
        combinations = math.comb(
            len(host_substitution_indices), abs(n_dopant)
        )
        if combinations < args.n_candidates:
            print(
                f"[Substitution] ⚠ 限定位点后最多只有 C("
                f"{len(host_substitution_indices)},{abs(n_dopant)})="
                f"{combinations} 种取代组合，少于 --n-candidates "
                f"{args.n_candidates}，候选会大量重复。"
                "可用 --min-formula-units 或更大的 --scaling 增加可取代位点"
            )

    records = generate_candidates(
        supercell=supercell,
        args=args,
        n_dopant=n_dopant,
        n_extra=n_extra,
        interstitial_pool=candidates,
        candidate_dir=candidate_dir,
        fingerprint=candidate_fingerprint,
        pool_distance_matrix=pool_distance_matrix,
        host_substitution_indices=host_substitution_indices,
    )

    # 说明：这里不再删除旧的 screening.csv / relaxed/。
    # 旧结果是否有效由 fingerprint 逐条校验（load_screening /
    # load_relaxations），指纹一致的候选与结果会被原样复用，
    # 因此不会因为个别文件缺失而丢掉昂贵的弛豫结果。

    records_by_id = {item["candidate_id"]: item for item in records}
    all_ids = sorted(records_by_id)
    selected = []
    screening = {}
    relaxations = {}

    # --------------------------------------------------------
    # Step 8: MatterSim 单点能筛选（第一阶段漏斗）
    # --------------------------------------------------------

    # MatterSim 模型按需加载：如果全部结果都能从 checkpoint 复用，
    # 就完全不需要加载权重。
    potential = None

    if args.relax:
        screening = load_screening(screening_file, ml_fp)
        pending = [index for index in all_ids if index not in screening]

        if not pending:
            print(
                f"[Screen] 已有全部 {len(screening)} 个单点能结果"
                f"（断点续跑）"
            )
        else:
            potential = build_mattersim_potential(
                device=args.device,
                load_path=args.model,
            )

            batch_size = max(1, args.screen_batch_size)
            initial_done = len(screening)
            started = time.monotonic()

            for offset in range(0, len(pending), batch_size):
                chunk = pending[offset:offset + batch_size]
                structures = [
                    records_by_id[index]["structure"] for index in chunk
                ]
                values = screen_structures(
                    potential,
                    structures,
                    batch_size=len(structures),
                )

                for index, (energy, energy_per_atom) in zip(chunk, values):
                    screening[index] = {
                        "candidate_id": index,
                        "energy": energy,
                        "energy_per_atom": energy_per_atom,
                    }

                write_screening(screening_file, screening, ml_fp)

                done = len(screening) - initial_done
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = done / elapsed
                remaining = (
                    (len(all_ids) - len(screening)) / rate
                    if rate
                    else float("inf")
                )
                print(
                    f"[Screen] {len(screening)}/{len(all_ids)} "
                    f"elapsed={elapsed / 60:.1f} min "
                    f"ETA={remaining / 60:.1f} min",
                    flush=True,
                )
    else:
        print("[Screen] --no-relax：跳过 MatterSim 筛选与弛豫")

    # --------------------------------------------------------
    # Step 9: 取筛选能量最低的前 N 个候选做批量弛豫（第二阶段漏斗）
    # --------------------------------------------------------

    initial_order = sorted(
        all_ids,
        key=lambda index: screening.get(index, {}).get(
            "energy_per_atom", float("inf")
        ),
    )
    initial_ranks = {
        index: rank for rank, index in enumerate(initial_order, start=1)
    }

    if args.relax:
        n_relax = (
            args.n_relax
            if args.n_relax is not None
            else min(len(all_ids), 20)
        )
        n_relax = max(1, min(n_relax, len(all_ids)))
        selected = initial_order[:n_relax]

        print(
            f"[Relax] 筛选前 {n_relax} 名候选 = "
            f"{[f'{i}' for i in selected]}"
        )

        relaxations = load_relaxations(selected, relaxed_dir, ml_fp)
        pending = [index for index in selected if index not in relaxations]
        chunk_size = max(1, args.relax_chunk_size)

        if pending and potential is None:
            potential = build_mattersim_potential(
                device=args.device,
                load_path=args.model,
            )

        if not pending:
            print(
                f"[Relax] 已有全部 {len(relaxations)} 个弛豫结果"
                f"（断点续跑）"
            )

        for offset in range(0, len(pending), chunk_size):
            chunk = pending[offset:offset + chunk_size]
            structures = [
                records_by_id[index]["structure"] for index in chunk
            ]
            # 至少保证单个结构能放进一个 batch
            max_natoms = max(
                args.relax_natoms_per_batch,
                max(len(structure) for structure in structures),
            )

            started = time.monotonic()
            results = relax_structures(
                potential=potential,
                structures=structures,
                fmax=args.fmax,
                steps=args.relax_steps,
                relax_cell=args.relax_cell,
                optimizer=args.relax_optimizer,
                max_natoms_per_batch=max_natoms,
                timeout=args.relax_timeout,
                label=f"chunk-{offset}",
            )
            elapsed = time.monotonic() - started

            for index, result in zip(chunk, results):
                # 把写出的 cif 路径写回内存记录，保证首次运行与续跑
                # 生成的 ranking.csv / results.json 字段一致
                relaxed_file = write_relaxation(
                    result, index, relaxed_dir, ml_fp
                )
                result["relaxed_file"] = (
                    "" if relaxed_file is None else str(relaxed_file)
                )
                relaxations[index] = result
                print(
                    f"[Relax] candidate={index:04d} "
                    f"converged={result['converged']} "
                    f"steps={result['steps']} "
                    f"energy_per_atom={result['energy_per_atom']} "
                    f"error={result['error'] or '-'}",
                    flush=True,
                )

            print(
                f"[Relax] {len(relaxations)}/{len(selected)} "
                f"batch_seconds={elapsed:.1f}",
                flush=True,
            )

    # --------------------------------------------------------
    # Step 10: 按弛豫能量排序并输出
    # --------------------------------------------------------

    usable = [
        index for index in selected
        if relaxations.get(index, {}).get("energy_per_atom") is not None
    ]
    converged = [
        index for index in usable
        if relaxations[index].get("converged")
    ]
    final_order = sorted(
        converged or usable,
        key=lambda index: relaxations[index]["energy_per_atom"],
    )
    if not final_order and args.relax:
        print(
            "[Rank] 没有可用的弛豫能量（可能全部未收敛或失败），"
            "按单点筛选顺序输出"
        )
        final_order = [i for i in initial_order if i in selected]
    if not final_order:
        final_order = all_ids

    relaxed_ranks = {
        index: rank for rank, index in enumerate(final_order, start=1)
    }
    best_index = final_order[0]
    best_structure = (
        relaxations.get(best_index, {}).get("structure")
        or records_by_id[best_index]["structure"]
    )
    best_record = relaxations.get(best_index, {})

    # 分等级输出弛豫结构
    for rank, index in enumerate(final_order, start=1):
        structure = relaxations.get(index, {}).get("structure")
        if structure is None:
            continue
        structure.to(
            filename=str(
                relaxed_dir / f"rank_{rank:03d}_candidate_{index:04d}.cif"
            )
        )

    best_file = output_dir / "best_structure.cif"
    write_cif_atomic(best_structure, best_file)

    # 汇总表：筛选 + 弛豫（与 LSWO 的 mattersim_ranking_200.csv 对应）
    ranking_fields = [
        "candidate_id",
        "candidate_file",
        "initial_energy_ev",
        "initial_energy_per_atom_ev",
        "initial_rank",
        "selected_for_relaxation",
        "converged",
        "relaxation_steps",
        "relaxation_seconds",
        "relaxed_energy_ev",
        "relaxed_energy_per_atom_ev",
        "relaxed_rank",
        "relaxed_file",
        "error",
    ]
    ranking_file = output_dir / "ranking.csv"
    temporary = ranking_file.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranking_fields)
        writer.writeheader()
        for index in initial_order:
            screen = screening.get(index, {})
            relaxed = relaxations.get(index, {})
            writer.writerow(
                {
                    "candidate_id": index,
                    "candidate_file": str(
                        (candidate_dir / records_by_id[index]["file"])
                        .relative_to(output_dir)
                    ),
                    "initial_energy_ev": screen.get("energy", ""),
                    "initial_energy_per_atom_ev": screen.get(
                        "energy_per_atom", ""
                    ),
                    "initial_rank": initial_ranks[index],
                    "selected_for_relaxation": index in selected,
                    "converged": relaxed.get("converged", ""),
                    "relaxation_steps": relaxed.get("steps", ""),
                    "relaxation_seconds": relaxed.get("seconds", ""),
                    "relaxed_energy_ev": relaxed.get("energy", ""),
                    "relaxed_energy_per_atom_ev": relaxed.get(
                        "energy_per_atom", ""
                    ),
                    "relaxed_rank": relaxed_ranks.get(index, ""),
                    "relaxed_file": relaxed.get("relaxed_file", ""),
                    "error": relaxed.get("error", ""),
                }
            )
    temporary.replace(ranking_file)

    summary = {
        "base_source": base_source,
        "base_material_id": base_material_id,
        "base_structure_file": base_file,
        "base_formula": args.base_formula,
        "target_formula": args.target_formula,
        "scaling": scaling,
        "candidate_fingerprint": candidate_fingerprint,
        "ml_fingerprint": ml_fp,
        "candidate_count": len(all_ids),
        "screened_count": len(screening),
        "mattersim_enabled": bool(args.relax),
        "selected_for_relaxation": len(selected),
        "relaxed_count": len(relaxations),
        "converged_count": len(converged),
        "fmax_ev_per_angstrom": args.fmax,
        "max_steps": args.relax_steps,
        "relax_cell": bool(args.relax_cell),
        "relax_optimizer": args.relax_optimizer,
        "screen_batch_size": args.screen_batch_size,
        "relax_chunk_size": args.relax_chunk_size,
        "relax_natoms_per_batch": args.relax_natoms_per_batch,
        "best_candidate_id": best_index,
        "best_initial_rank": initial_ranks.get(best_index),
        "best_relaxed_rank": relaxed_ranks.get(best_index),
        "best_energy_ev": best_record.get("energy"),
        "best_energy_per_atom_ev": best_record.get("energy_per_atom"),
        "best_converged": best_record.get("converged"),
        "formula": best_structure.composition.formula,
        "num_atoms": len(best_structure),
        "best_structure_file": str(best_file),
    }

    write_json_atomic(output_dir / "summary.json", summary)

    write_json_atomic(
        output_dir / "results.json",
        {
            "base_source": base_source,
            "base_material_id": base_material_id,
            "base_structure_file": base_file,
            "base_formula": args.base_formula,
            "target_formula": args.target_formula,
            "host": args.host,
            "dopant": args.dopant,
            "compensating_element": args.compensating_element,
            "scaling": scaling,
            "candidate_fingerprint": candidate_fingerprint,
            "ml_fingerprint": ml_fp,
            "composition_calculation": numbers,
            "wyckoff_report": wyckoff_report,
            "run_config": run_config,
            "candidates": [
                {
                    "candidate_id": index,
                    "file": records_by_id[index]["file"],
                    "substitution_indices": records_by_id[index][
                        "substitution_indices"
                    ],
                    "interstitial_sites": records_by_id[index][
                        "interstitial_sites"
                    ],
                    "removed_indices": records_by_id[index].get(
                        "removed_indices", []
                    ),
                    "formula": records_by_id[index][
                        "structure"
                    ].composition.formula,
                }
                for index in all_ids
            ],
            "screening": [screening[index] for index in sorted(screening)],
            "relaxation": [
                {
                    key: value
                    for key, value in relaxations[index].items()
                    if key != "structure"
                }
                for index in sorted(relaxations)
            ],
            "ranking": final_order,
            "best_candidate": summary,
        },
    )

    print("\n==============================")
    print("Summary")
    print("==============================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
