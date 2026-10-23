#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无序 → 有序 结构搜索 + MatterSim 筛选/弛豫。

这是 mattergen 里 LSWO 流程在本仓库的复现，包含两个阶段：

阶段 A（无序 → 有序，移植自 mattergen/OBELIX/order_disordered_cifs.py）：
    1. 解析 CIF，丢弃占据 < --min-occupancy 的物种（受参考化学式保护的元素除外）；
    2. 检测 split site：同元素近邻（距离 ≤ --split-site-max-distance）
       且占据和 ≈ 1 的两个位点；以及 split cluster：一个中心位点 + 多个
       对称卫星位点构成的一原子簇（中心-卫星距离 ≤ --split-cluster-max-distance）；
    3. 按空间群等价轨道分组（失败时回退到物种签名），split site / cluster
       单独成组，并作为「互斥」约束处理；
    4. 用最大余数法把占据概率整数化，同时保护痕量元素；枚举对角超胞
       （行列式 ≤ --max-supercell-multiplier、原子数 ≤ --max-atoms），
       以占据近似误差最小者为方案；
    5. 生成候选排列：random（随机赋值 + StructureMatcher 去重）/
       ewald（pymatgen Ewald 排序）/ enumerate（enumlib 精确枚举）/ auto；
    6. 校验占据误差与组成漂移（可提供参考化学式），写出候选 CIF 与报告。

阶段 B（ML 筛选与排序，复用 main/generate_stru.py 的实现）：
    7. 对全部候选做 MatterSim 单点能筛选；
    8. 取能量最低的 --n-relax 个做批量结构弛豫；
    9. 排序（收敛优先 → 每原子能量），输出 rank_*.cif 与最优有序结构。

示例（复现 outputs/LSWO-ordered）：

python main/order_stru.py \
    --input /tmp/LSWO-input/LSWO.cif \
    --num-candidates 200 \
    --n-relax 20 \
    --output outputs/LSWO-ordered

或者批量处理一个目录：

python main/order_stru.py \
    --input OBELIX/cifs \
    --num-candidates 10 \
    --keep-top 1 \
    --output outputs/obelix_ordered

依赖：

- pymatgen（必需）；
- mattersim（阶段 B 需要，用 --no-rank 可只做无序→有序）。
"""

import csv
import sys
import json
import math
import time
import random
import hashlib
import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Collection, Mapping, Sequence

import numpy as np

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 复用 generate_stru.py 的 MatterSim 漏斗与落盘工具，避免重复实现。
# 优先使用相对导入：作为 package（main.order_stru）导入时可以和
# main.generate_stru 共享同一个模块实例；否则两个模块名会各自加载一份，
# MatterSimUnavailable 之类的异常类型会不相等，环境级失败的判定就会失效。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from . import generate_stru as _shared
except ImportError:  # 直接以脚本方式运行：python main/order_stru.py
    import generate_stru as _shared

MattersimUnavailable = _shared.MattersimUnavailable
best_diagonal_scaling = _shared.best_diagonal_scaling
build_mattersim_potential = _shared.build_mattersim_potential
load_relaxations = _shared.load_relaxations
load_screening = _shared.load_screening
ml_fingerprint = _shared.ml_fingerprint
read_json = _shared.read_json
relax_structures = _shared.relax_structures
screen_structures = _shared.screen_structures
write_cif_atomic = _shared.write_cif_atomic
write_json_atomic = _shared.write_json_atomic
write_relaxation = _shared.write_relaxation
write_screening = _shared.write_screening


# ------------------------------------------------------------
# 常量与数据结构（与 OBELIX 保持一致）
# ------------------------------------------------------------

VACANCY = "__vacancy__"
GROUP_PROPERTY = "_ordering_group"
CLUSTER_INSTANCE_PROPERTY = "_split_cluster_instance"
CLUSTER_ROLE_PROPERTY = "_split_cluster_role"

# 位点占据之和是否「等于 1」的数值容差（浮点保护，不是物理阈值）
OCCUPANCY_SUM_TOLERANCE = 1e-5

# 算法版本号：有序化逻辑变化时用它让旧候选失效
ORDERING_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SplitSitePair:
    species: str
    site_indices: tuple[int, int]
    occupancy_sum: float


@dataclass(frozen=True)
class SplitSiteCluster:
    family_id: int
    species: str
    site_indices: tuple[int, ...]
    max_center_distance: float
    occupancy_sum: float
    per_site_occupancies: tuple[float, ...]

    @property
    def cluster_size(self) -> int:
        return len(self.site_indices)


@dataclass(frozen=True)
class OccupancyGroup:
    group_id: int
    site_indices: tuple[int, ...]
    probabilities: dict[str, float]
    split_site_species: str | None = None
    cluster_size: int | None = None
    cluster_max_distance: float | None = None
    cluster_center_occupancy: float | None = None
    cluster_satellite_occupancy: float | None = None


@dataclass(frozen=True)
class AllocationPlan:
    multiplier: int
    scaling_matrix: tuple[int, int, int]
    allocations: dict[int, dict[str, int]]
    split_site_group_ids: tuple[int, ...]
    cluster_group_ids: tuple[int, ...]
    cluster_center_counts: dict[int, int]
    max_occupancy_error: float
    mean_occupancy_error: float
    composition_max_error: float | None
    composition_max_drift: float | None
    estimated_num_atoms: int
    within_error_tolerance: bool
    within_composition_tolerance: bool | None


@dataclass
class CandidateRecord:
    """单个候选的逐候选报告行（阶段 A 部分对齐 OBELIX 的 CandidateResult）。"""

    parent_id: str
    source_file: str
    status: str
    message: str = ""
    reference_formula: str = ""
    source_formula: str = ""
    cleaned_formula: str = ""
    source_num_sites: int | None = None
    dropped_species_count: int = 0
    dropped_occupancy_sum: float = 0.0
    grouping_method: str = ""
    candidate_method: str = ""
    split_site_pair_count: int = 0
    split_site_cluster_count: int = 0
    split_site_species: str = ""
    supercell_multiplier: int | None = None
    supercell_matrix: str = ""
    occupancy_max_error: float | None = None
    occupancy_mean_error: float | None = None
    within_error_tolerance: bool | None = None
    source_composition_max_error: float | None = None
    composition_max_error: float | None = None
    composition_max_drift: float | None = None
    within_composition_tolerance: bool | None = None
    missing_reference_elements: str = ""
    candidate_index: int | None = None
    num_atoms: int | None = None
    ordered_formula: str = ""
    output_file: str = ""


# ------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------

def stable_seed(base_seed: int, material_id: str) -> int:
    """由固定种子和材料 ID 推出稳定随机种子（与 OBELIX 一致）。"""
    digest = hashlib.sha256(material_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**32)


def atomic_fractions(composition) -> dict[str, float]:
    """把组成换成原子分数（用于组成误差比较）。"""
    amounts: dict[str, float] = {}
    for element, amount in composition.items():
        if float(amount) <= 0:
            continue
        symbol = element.symbol if hasattr(element, "symbol") else str(element)
        amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("无法比较空组成")
    return {element: amount / total for element, amount in amounts.items()}


def composition_errors(reference, actual) -> dict[str, float]:
    reference_fractions = atomic_fractions(reference)
    actual_fractions = atomic_fractions(actual)
    elements = set(reference_fractions) | set(actual_fractions)
    return {
        element: abs(
            actual_fractions.get(element, 0.0)
            - reference_fractions.get(element, 0.0)
        )
        for element in elements
    }


def composition_error_metrics(
    reference, actual, baseline
) -> tuple[float, float]:
    """返回 (与参考组成的最大误差, 相对原始 CIF 的最大漂移)。"""
    actual_errors = composition_errors(reference, actual)
    baseline_errors = composition_errors(reference, baseline)
    elements = set(actual_errors) | set(baseline_errors)
    max_error = max(actual_errors.values(), default=0.0)
    max_drift = max(
        (
            max(
                0.0,
                actual_errors.get(element, 0.0)
                - baseline_errors.get(element, 0.0),
            )
            for element in elements
        ),
        default=0.0,
    )
    return max_error, max_drift


def parse_structure(path: Path) -> Structure:
    structures = CifParser(path).parse_structures(
        primitive=True, on_error="ignore"
    )
    if not structures:
        raise ValueError("无法从 CIF 解析出结构")
    return structures[0]


def load_reference_compositions(path: Path) -> dict[str, Composition]:
    """
    读取参考化学式表（列名与 OBELIX/all.csv 一致：ID + True Composition）。
    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ID", "True Composition"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} 缺少列: {', '.join(sorted(missing))}"
            )
        references: dict[str, Composition] = {}
        for line_number, row in enumerate(reader, start=2):
            material_id = (row["ID"] or "").strip()
            formula = (row["True Composition"] or "").strip()
            if not material_id or not formula:
                continue
            if material_id in references:
                raise ValueError(
                    f"{path} 第 {line_number} 行 material ID 重复: "
                    f"{material_id!r}"
                )
            try:
                references[material_id] = Composition(formula)
            except Exception as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行无法解析 {material_id!r} 的 "
                    f"True Composition: {formula!r}"
                ) from exc
    return references


# ------------------------------------------------------------
# 阶段 A-1：占据清洗与 split site 检测
# ------------------------------------------------------------

def normalize_probabilities(
    probabilities: dict[str, float], tolerance: float
) -> dict[str, float]:
    total = sum(probabilities.values())
    if total > 1.0 + tolerance:
        raise ValueError(f"位点占据之和 {total:.6f} 超过 1")
    if math.isclose(total, 1.0, abs_tol=tolerance):
        return {key: value / total for key, value in probabilities.items()}
    normalized = dict(probabilities)
    normalized[VACANCY] = max(0.0, 1.0 - total)
    return normalized


def species_signature(site) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (species.symbol, round(float(occupancy), 8))
            for species, occupancy in site.species.items()
        )
    )


def clean_small_occupancies(
    structure: Structure,
    min_occupancy: float,
    occupancy_sum_tolerance: float = OCCUPANCY_SUM_TOLERANCE,
    protected_elements: Collection[str] = (),
) -> tuple[Structure, int, float]:
    """丢弃低于阈值的占据；被参考化学式保护的痕量元素例外。"""
    cleaned = structure.copy()
    remove_indices: list[int] = []
    dropped_species_count = 0
    dropped_occupancy_sum = 0.0
    protected = set(protected_elements)
    source_elements = {
        species.symbol
        for site in structure
        for species, occupancy in site.species.items()
        if occupancy > 0
    }
    elements_above_threshold = {
        species.symbol
        for site in structure
        for species, occupancy in site.species.items()
        if occupancy >= min_occupancy
    }
    keep_below_threshold = (protected & source_elements) - elements_above_threshold

    for index, site in enumerate(structure):
        original = {
            species: float(occupancy)
            for species, occupancy in site.species.items()
        }
        kept: dict[object, float] = {}
        for species, occupancy in original.items():
            if (
                occupancy < min_occupancy
                and species.symbol not in keep_below_threshold
            ):
                dropped_species_count += 1
                dropped_occupancy_sum += occupancy
            else:
                kept[species] = occupancy

        if not kept:
            remove_indices.append(index)
            continue

        original_total = sum(original.values())
        if math.isclose(
            original_total, 1.0, abs_tol=occupancy_sum_tolerance
        ):
            kept_total = sum(kept.values())
            kept = {
                species: value / kept_total
                for species, value in kept.items()
            }
        cleaned.replace(index, Composition(kept))

    if remove_indices:
        cleaned.remove_sites(remove_indices)
    return cleaned, dropped_species_count, dropped_occupancy_sum


def find_split_site_clusters(
    structure: Structure,
    max_distance: float,
    occupancy_tolerance: float,
    symprec: float = 0.1,
) -> list[SplitSiteCluster]:
    """找出「一个中心轨道 + 多个卫星轨道」构成的一原子簇。"""
    try:
        equivalent_indices = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetrized_structure().equivalent_indices
    except Exception:
        return []

    orbit_data: dict[str, list[tuple[tuple[int, ...], float]]] = {}
    for indices in equivalent_indices:
        orbit = tuple(int(index) for index in indices)
        reference = structure[orbit[0]]
        if len(reference.species) != 1:
            continue
        species, occupancy = next(iter(reference.species.items()))
        value = float(occupancy)
        if 0.0 < value < 1.0:
            orbit_data.setdefault(species.symbol, []).append((orbit, value))

    clusters: list[SplitSiteCluster] = []
    family_id = 0
    for species, element_orbits in orbit_data.items():
        sorted_orbits = sorted(element_orbits, key=lambda item: len(item[0]))
        used_orbits: set[int] = set()

        for center_position, (
            center_indices,
            center_occupancy,
        ) in enumerate(sorted_orbits):
            if center_position in used_orbits:
                continue
            for satellite_position, (
                satellite_indices,
                satellite_occupancy,
            ) in enumerate(sorted_orbits):
                if satellite_position == center_position:
                    continue
                if satellite_position in used_orbits:
                    continue
                if len(satellite_indices) % len(center_indices):
                    continue
                satellites_per_center = len(satellite_indices) // len(
                    center_indices
                )
                if satellites_per_center < 2:
                    continue

                occupancy_sum = (
                    center_occupancy
                    + satellites_per_center * satellite_occupancy
                )
                if abs(occupancy_sum - 1.0) > occupancy_tolerance + 1e-12:
                    continue

                instances: list[tuple[int, tuple[int, ...], float]] = []
                satellite_usage: dict[int, int] = {
                    index: 0 for index in satellite_indices
                }
                valid = True
                for center_index in center_indices:
                    neighbors = sorted(
                        (
                            float(
                                structure.get_distance(
                                    center_index, satellite_index
                                )
                            ),
                            satellite_index,
                        )
                        for satellite_index in satellite_indices
                        if structure.get_distance(
                            center_index, satellite_index
                        )
                        <= max_distance + 1e-8
                    )
                    if len(neighbors) != satellites_per_center:
                        valid = False
                        break
                    for _, satellite_index in neighbors:
                        satellite_usage[satellite_index] += 1
                    instances.append(
                        (
                            center_index,
                            tuple(index for _, index in neighbors),
                            max(distance for distance, _ in neighbors),
                        )
                    )

                if not valid or any(
                    count != 1 for count in satellite_usage.values()
                ):
                    continue

                for center_index, nearby_satellites, farthest in instances:
                    clusters.append(
                        SplitSiteCluster(
                            family_id=family_id,
                            species=species,
                            site_indices=(center_index, *nearby_satellites),
                            max_center_distance=farthest,
                            occupancy_sum=occupancy_sum,
                            per_site_occupancies=(
                                center_occupancy,
                                *(
                                    satellite_occupancy
                                    for _ in nearby_satellites
                                ),
                            ),
                        )
                    )
                used_orbits.update((center_position, satellite_position))
                family_id += 1
                break

    return clusters


def find_split_site_pairs(
    structure: Structure,
    max_distance: float,
    occupancy_tolerance: float,
    excluded_indices: Collection[int] = (),
) -> list[SplitSitePair]:
    """找出同元素近邻、占据和 ≈ 1 的互斥位点对。"""
    excluded = set(excluded_indices)
    partial_sites: dict[str, list[tuple[int, float]]] = {}
    for index, site in enumerate(structure):
        if index in excluded:
            continue
        if len(site.species) != 1:
            continue
        species, occupancy = next(iter(site.species.items()))
        value = float(occupancy)
        if 0.0 < value < 1.0:
            partial_sites.setdefault(species.symbol, []).append((index, value))

    if not partial_sites:
        return []

    distances = np.asarray(structure.distance_matrix, dtype=float)
    candidates: list[tuple[float, str, int, int, float]] = []
    for species, sites in partial_sites.items():
        if len(sites) < 2:
            continue
        nearest = {
            index: min(
                distances[index, other_index]
                for other_index, _ in sites
                if other_index != index
            )
            for index, _ in sites
        }
        for (first_index, first_occupancy), (
            second_index,
            second_occupancy,
        ) in itertools.combinations(sites, 2):
            distance = float(distances[first_index, second_index])
            occupancy_sum = first_occupancy + second_occupancy
            if distance > max_distance:
                continue
            if abs(occupancy_sum - 1.0) > occupancy_tolerance + 1e-12:
                continue
            if distance > nearest[first_index] + 1e-6:
                continue
            if distance > nearest[second_index] + 1e-6:
                continue
            candidates.append(
                (distance, species, first_index, second_index, occupancy_sum)
            )

    pairs: list[SplitSitePair] = []
    assigned_indices: set[int] = set()
    for (
        distance,
        species,
        first_index,
        second_index,
        occupancy_sum,
    ) in sorted(candidates):
        if first_index in assigned_indices or second_index in assigned_indices:
            continue
        pairs.append(
            SplitSitePair(
                species=species,
                site_indices=(first_index, second_index),
                occupancy_sum=occupancy_sum,
            )
        )
        assigned_indices.update((first_index, second_index))
    return pairs


def find_occupancy_groups(
    structure: Structure,
    symprec: float,
    occupancy_sum_tolerance: float = OCCUPANCY_SUM_TOLERANCE,
    split_site_pairs: Sequence[SplitSitePair] = (),
    split_site_clusters: Sequence[SplitSiteCluster] = (),
) -> tuple[list[OccupancyGroup], str]:
    """
    把位点分成占据组：默认按空间群等价轨道，split site/cluster 单独成组。

    分组结果会写回 `structure` 的 site property（GROUP_PROPERTY 等），
    后续候选生成直接依赖这些标签。
    """
    try:
        equivalent_indices = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetrized_structure().equivalent_indices
        grouping_method = "spacegroup"
    except Exception:
        by_signature: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for index, site in enumerate(structure):
            by_signature.setdefault(species_signature(site), []).append(index)
        equivalent_indices = list(by_signature.values())
        grouping_method = "species_signature_fallback"

    assigned_split_indices: set[int] = set()
    clusters_by_family: dict[int, list[SplitSiteCluster]] = {}
    for cluster in split_site_clusters:
        overlap = assigned_split_indices & set(cluster.site_indices)
        if overlap:
            raise ValueError(
                f"split-site cluster 在 {sorted(overlap)} 处重叠"
            )
        assigned_split_indices.update(cluster.site_indices)
        clusters_by_family.setdefault(cluster.family_id, []).append(cluster)

    for pair in split_site_pairs:
        overlap = assigned_split_indices & set(pair.site_indices)
        if overlap:
            raise ValueError(f"split-site pair 在 {sorted(overlap)} 处重叠")
        assigned_split_indices.update(pair.site_indices)

    grouped_indices: list[
        tuple[
            tuple[int, ...],
            SplitSitePair | None,
            list[SplitSiteCluster] | None,
        ]
    ] = []
    for family_clusters in clusters_by_family.values():
        family_indices = tuple(
            index
            for cluster in family_clusters
            for index in cluster.site_indices
        )
        grouped_indices.append((family_indices, None, family_clusters))
    for pair in split_site_pairs:
        grouped_indices.append((pair.site_indices, pair, None))
    for indices in equivalent_indices:
        remaining = tuple(
            int(index)
            for index in indices
            if index not in assigned_split_indices
        )
        if remaining:
            grouped_indices.append((remaining, None, None))

    if split_site_clusters:
        grouping_method += "+split_site_clusters"
    if split_site_pairs:
        grouping_method += "+split_site_pairs"

    groups: list[OccupancyGroup] = []
    group_ids = [-1] * len(structure)
    for group_id, (indices, split_pair, family_clusters) in enumerate(
        grouped_indices
    ):
        if family_clusters is not None:
            reference_cluster = family_clusters[0]
            cluster_size = reference_cluster.cluster_size
            center_occupancy = reference_cluster.per_site_occupancies[0]
            satellite_occupancy = reference_cluster.per_site_occupancies[1]
            if any(
                cluster.cluster_size != cluster_size
                for cluster in family_clusters
            ):
                raise ValueError(f"split-site cluster family {group_id} 不一致")
            average_occupancy = reference_cluster.occupancy_sum / cluster_size
            probabilities = {
                reference_cluster.species: average_occupancy,
                VACANCY: 1.0 - average_occupancy,
            }
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
                split_site_species=reference_cluster.species,
                cluster_size=cluster_size,
                cluster_max_distance=max(
                    cluster.max_center_distance
                    for cluster in family_clusters
                ),
                cluster_center_occupancy=center_occupancy,
                cluster_satellite_occupancy=satellite_occupancy,
            )
        elif split_pair is None:
            reference = structure[indices[0]]
            probabilities = {
                species.symbol: float(occupancy)
                for species, occupancy in reference.species.items()
            }
            probabilities = normalize_probabilities(
                probabilities, occupancy_sum_tolerance
            )
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
            )
        else:
            probabilities = {
                split_pair.species: split_pair.occupancy_sum / len(indices)
            }
            probabilities = normalize_probabilities(
                probabilities, occupancy_sum_tolerance
            )
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
                split_site_species=split_pair.species,
            )
        groups.append(group)
        for index in indices:
            group_ids[index] = group_id

    if any(group_id < 0 for group_id in group_ids):
        raise RuntimeError("存在未分配占据组的位点")
    structure.add_site_property(GROUP_PROPERTY, group_ids)

    cluster_instance_ids = [-1] * len(structure)
    cluster_roles = [-1] * len(structure)
    for instance_id, cluster in enumerate(split_site_clusters):
        for role, index in enumerate(cluster.site_indices):
            cluster_instance_ids[index] = instance_id
            cluster_roles[index] = 0 if role == 0 else 1
    structure.add_site_property(CLUSTER_INSTANCE_PROPERTY, cluster_instance_ids)
    structure.add_site_property(CLUSTER_ROLE_PROPERTY, cluster_roles)

    return groups, grouping_method


# ------------------------------------------------------------
# 阶段 A-2：占据整数化与超胞方案
# ------------------------------------------------------------

def largest_remainder_allocation(
    num_positions: int,
    probabilities: dict[str, float],
) -> tuple[dict[str, int], list[float]]:
    raw = {
        key: num_positions * value for key, value in probabilities.items()
    }
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = num_positions - sum(counts.values())
    ranked = sorted(
        probabilities,
        key=lambda key: (raw[key] - counts[key], probabilities[key], key),
        reverse=True,
    )
    for key in ranked[:remainder]:
        counts[key] += 1

    return counts


def preserve_species_globally(
    groups: Sequence[OccupancyGroup],
    allocations: dict[int, dict[str, int]],
    multiplier: int,
    required_species: Collection[str],
) -> None:
    """保证每种必须保留的物种至少有 1 个原子（代价最小的组内转移）。"""
    required = set(required_species) - {VACANCY}
    global_counts: dict[str, int] = {}
    for counts in allocations.values():
        for species, count in counts.items():
            global_counts[species] = global_counts.get(species, 0) + count

    missing = {
        species for species in required if global_counts.get(species, 0) == 0
    }
    while missing:
        recipient = min(
            missing,
            key=lambda species: (
                sum(species in group.probabilities for group in groups),
                species,
            ),
        )
        transfers: list[tuple[float, float, int, str]] = []
        for group in groups:
            if group.probabilities.get(recipient, 0.0) <= 0:
                continue
            counts = allocations[group.group_id]
            num_positions = len(group.site_indices) * multiplier
            for donor, donor_count in counts.items():
                if donor == recipient or donor_count <= 0:
                    continue
                if donor in required and global_counts.get(donor, 0) <= 1:
                    continue
                before = abs(
                    counts[donor] / num_positions
                    - group.probabilities[donor]
                )
                before += abs(
                    counts[recipient] / num_positions
                    - group.probabilities[recipient]
                )
                after = abs(
                    (counts[donor] - 1) / num_positions
                    - group.probabilities[donor]
                )
                after += abs(
                    (counts[recipient] + 1) / num_positions
                    - group.probabilities[recipient]
                )
                transfers.append(
                    (
                        after - before,
                        -group.probabilities[recipient],
                        group.group_id,
                        donor,
                    )
                )

        if not transfers:
            raise ValueError(
                f"在 multiplier={multiplier} 下无法保留必需物种 {recipient!r}"
            )
        _, _, group_id, donor = min(transfers)
        allocations[group_id][donor] -= 1
        allocations[group_id][recipient] += 1
        global_counts[donor] -= 1
        global_counts[recipient] = 1
        missing.remove(recipient)


def allocation_counts(
    allocations: Mapping[int, Mapping[str, int]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group_counts in allocations.values():
        for species, count in group_counts.items():
            if species != VACANCY:
                counts[species] = counts.get(species, 0) + count
    return counts


def diagonal_scaling_matrix(
    structure: Structure, multiplier: int
) -> tuple[int, int, int]:
    """
    （保留 OBELIX 的命名）返回行列式等于 multiplier 的最佳对角超胞。

    实现与 main/generate_stru.py 的 best_diagonal_scaling 共用一份，
    避免两处规则各自演化。
    """
    return best_diagonal_scaling(structure, multiplier)


def build_allocation_plan(
    structure: Structure,
    groups: Sequence[OccupancyGroup],
    max_multiplier: int,
    max_atoms: int,
    max_occupancy_error: float,
    preserve_retained_species: bool,
    reference_composition: Composition | None = None,
    baseline_composition: Composition | None = None,
    composition_tolerance: float = 0.05,
    required_species: Collection[str] = (),
) -> AllocationPlan:
    """
    枚举对角超胞，找占据整数化误差可接受的最小方案。

    优先返回「占据误差与组成漂移都在容差内」的方案；否则返回评分最好的方案，
    并在报告里标为 approximation_warning。
    """
    feasible: list[AllocationPlan] = []
    retained_species = {
        species
        for group in groups
        for species, probability in group.probabilities.items()
        if species != VACANCY and probability > 0
    }
    species_to_preserve = set(required_species)
    if preserve_retained_species:
        species_to_preserve.update(retained_species)

    for multiplier in range(1, max_multiplier + 1):
        allocations: dict[int, dict[str, int]] = {}
        cluster_center_counts: dict[int, int] = {}
        for group in groups:
            num_positions = len(group.site_indices) * multiplier
            if group.cluster_size is not None:
                num_instances = len(group.site_indices) // group.cluster_size
                total_clusters = num_instances * multiplier
                species = group.split_site_species
                if species is None:
                    raise ValueError(
                        f"split-site cluster 组 {group.group_id} 缺少物种"
                    )
                counts = {
                    species: total_clusters,
                    VACANCY: total_clusters * (group.cluster_size - 1),
                }
                center_occupancy = group.cluster_center_occupancy or 0.0
                satellite_occupancy = (
                    group.cluster_satellite_occupancy or 0.0
                )
                normalized_center = center_occupancy / (
                    center_occupancy
                    + (group.cluster_size - 1) * satellite_occupancy
                )
                role_counts = largest_remainder_allocation(
                    total_clusters,
                    {
                        "center": normalized_center,
                        "satellite": 1.0 - normalized_center,
                    },
                )
                cluster_center_counts[group.group_id] = role_counts["center"]
            elif group.split_site_species is not None:
                if len(group.site_indices) != 2:
                    raise ValueError(
                        f"split-site 组 {group.group_id} 必须包含两个位点"
                    )
                counts = {
                    group.split_site_species: multiplier,
                    VACANCY: multiplier,
                }
            else:
                counts = largest_remainder_allocation(
                    num_positions, group.probabilities
                )
            allocations[group.group_id] = counts

        try:
            preserve_species_globally(
                groups,
                allocations,
                multiplier,
                required_species=species_to_preserve,
            )
        except ValueError:
            continue

        errors: list[float] = []
        for group in groups:
            if group.cluster_size is not None:
                total_clusters = (
                    len(group.site_indices) // group.cluster_size * multiplier
                )
                center_count = cluster_center_counts[group.group_id]
                satellite_count = total_clusters - center_count
                errors.extend(
                    (
                        abs(
                            center_count / total_clusters
                            - (group.cluster_center_occupancy or 0.0)
                        ),
                        abs(
                            satellite_count
                            / ((group.cluster_size - 1) * total_clusters)
                            - (group.cluster_satellite_occupancy or 0.0)
                        ),
                    )
                )
                continue
            num_positions = len(group.site_indices) * multiplier
            errors.extend(
                abs(
                    allocations[group.group_id].get(species, 0)
                    / num_positions
                    - probability
                )
                for species, probability in group.probabilities.items()
            )

        ordered_counts = allocation_counts(allocations)
        estimated_num_atoms = sum(ordered_counts.values())
        if estimated_num_atoms > max_atoms:
            continue

        max_error = max(errors, default=0.0)
        composition_max_error: float | None = None
        composition_max_drift: float | None = None
        within_composition_tolerance: bool | None = None
        if reference_composition is not None:
            baseline = baseline_composition or structure.composition
            (
                composition_max_error,
                composition_max_drift,
            ) = composition_error_metrics(
                reference_composition, ordered_counts, baseline
            )
            within_composition_tolerance = (
                composition_max_drift <= composition_tolerance + 1e-12
            )

        plan = AllocationPlan(
            multiplier=multiplier,
            scaling_matrix=diagonal_scaling_matrix(structure, multiplier),
            allocations=allocations,
            split_site_group_ids=tuple(
                group.group_id
                for group in groups
                if group.split_site_species is not None
                and group.cluster_size is None
            ),
            cluster_group_ids=tuple(
                group.group_id
                for group in groups
                if group.cluster_size is not None
            ),
            cluster_center_counts=cluster_center_counts,
            max_occupancy_error=max_error,
            mean_occupancy_error=float(np.mean(errors)) if errors else 0.0,
            composition_max_error=composition_max_error,
            composition_max_drift=composition_max_drift,
            estimated_num_atoms=estimated_num_atoms,
            within_error_tolerance=max_error <= max_occupancy_error + 1e-12,
            within_composition_tolerance=within_composition_tolerance,
        )
        feasible.append(plan)
        if (
            plan.within_error_tolerance
            and plan.within_composition_tolerance is not False
        ):
            return plan

    if not feasible:
        raise ValueError(
            f"multiplier ≤ {max_multiplier} 内没有满足 max_atoms={max_atoms} 的方案"
        )
    if reference_composition is None:
        return min(
            feasible,
            key=lambda plan: (
                plan.max_occupancy_error,
                plan.mean_occupancy_error,
                plan.multiplier,
            ),
        )

    def plan_score(plan: AllocationPlan) -> tuple[float, float, float, float, int]:
        occupancy_excess = max(
            0.0, plan.max_occupancy_error - max_occupancy_error
        )
        composition_drift = plan.composition_max_drift or 0.0
        composition_excess = max(
            0.0, composition_drift - composition_tolerance
        )
        return (
            max(occupancy_excess, composition_excess),
            occupancy_excess + composition_excess,
            composition_drift,
            plan.max_occupancy_error,
            plan.multiplier,
        )

    return min(feasible, key=plan_score)


# ------------------------------------------------------------
# 阶段 A-3：候选生成
# ------------------------------------------------------------

def finalize_candidate_cell(structure: Structure) -> Structure:
    """把候选约化到原胞 + Niggli 约化。"""
    primitive = structure.get_primitive_structure(tolerance=0.1)
    return primitive.get_reduced_structure(reduction_algo="niggli")


def nearest_site_pairs(
    structure: Structure,
    indices: Sequence[int],
) -> list[tuple[int, int]]:
    if len(indices) % 2:
        raise ValueError(f"split site 数量为奇数，无法配对: {len(indices)}")
    edges = sorted(
        (structure.get_distance(first, second), first, second)
        for first, second in itertools.combinations(indices, 2)
    )
    assigned: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, first, second in edges:
        if first in assigned or second in assigned:
            continue
        pairs.append((first, second))
        assigned.update((first, second))
    if len(assigned) != len(indices):
        raise RuntimeError("超胞中有 split site 无法完成配对")
    return pairs


def split_cluster_copies(
    structure: Structure,
    indices: Sequence[int],
    cluster_size: int,
    max_center_distance: float,
) -> list[tuple[int, ...]]:
    """借助持久化的 site label 恢复超胞里的每个一原子簇实例。"""
    by_instance: dict[int, list[int]] = {}
    for index in indices:
        instance_id = int(structure[index].properties[CLUSTER_INSTANCE_PROPERTY])
        by_instance.setdefault(instance_id, []).append(index)

    clusters: list[tuple[int, ...]] = []
    satellites_per_center = cluster_size - 1
    for instance_id, instance_indices in sorted(by_instance.items()):
        if instance_id < 0:
            raise RuntimeError("split-site cluster 缺少 instance 标签")
        centers = sorted(
            index
            for index in instance_indices
            if int(structure[index].properties[CLUSTER_ROLE_PROPERTY]) == 0
        )
        satellites = sorted(
            index
            for index in instance_indices
            if int(structure[index].properties[CLUSTER_ROLE_PROPERTY]) == 1
        )
        if len(satellites) != satellites_per_center * len(centers):
            raise RuntimeError(
                f"cluster instance {instance_id} 的 role 数量不一致"
            )

        satellite_usage = {index: 0 for index in satellites}
        instance_clusters: list[tuple[int, ...]] = []
        for center in centers:
            nearby = sorted(
                (
                    float(structure.get_distance(center, satellite)),
                    satellite,
                )
                for satellite in satellites
                if structure.get_distance(center, satellite)
                <= max_center_distance + 1e-5
            )
            if len(nearby) != satellites_per_center:
                raise RuntimeError(
                    f"cluster instance {instance_id} 在超胞展开后无法唯一重建"
                )
            for _, satellite in nearby:
                satellite_usage[satellite] += 1
            instance_clusters.append(
                (center, *(index for _, index in nearby))
            )

        if any(count != 1 for count in satellite_usage.values()):
            raise RuntimeError(
                f"cluster instance {instance_id} 在超胞展开后出现重叠"
            )
        clusters.extend(instance_clusters)
    return clusters


def make_ordered_candidate(
    decorated_structure: Structure,
    plan: AllocationPlan,
    groups: Sequence[OccupancyGroup],
    rng: random.Random,
) -> Structure:
    """按方案做一次随机整数占据赋值，得到一个有序候选。"""
    candidate = decorated_structure.copy()
    candidate.make_supercell(plan.scaling_matrix)

    indices_by_group: dict[int, list[int]] = {}
    for index, site in enumerate(candidate):
        group_id = int(site.properties[GROUP_PROPERTY])
        indices_by_group.setdefault(group_id, []).append(index)

    group_by_id = {group.group_id: group for group in groups}
    split_site_group_ids = set(plan.split_site_group_ids)
    cluster_group_ids = set(plan.cluster_group_ids)
    remove_indices: list[int] = []

    for group_id, indices in indices_by_group.items():
        if group_id in cluster_group_ids:
            group = group_by_id[group_id]
            cluster_size = group.cluster_size
            max_distance = group.cluster_max_distance
            species = group.split_site_species
            if cluster_size is None or max_distance is None or species is None:
                raise RuntimeError(f"split-site cluster 组 {group_id} 信息不全")
            clusters = split_cluster_copies(
                candidate,
                indices,
                cluster_size=cluster_size,
                max_center_distance=max_distance,
            )
            center_count = plan.cluster_center_counts[group_id]
            if not 0 <= center_count <= len(clusters):
                raise RuntimeError(
                    f"cluster 组 {group_id} 的 center 数量非法"
                )
            choose_center = [True] * center_count + [False] * (
                len(clusters) - center_count
            )
            rng.shuffle(choose_center)
            for cluster, use_center in zip(clusters, choose_center):
                chosen = cluster[0] if use_center else rng.choice(cluster[1:])
                candidate.replace(chosen, Element(species))
                remove_indices.extend(
                    index for index in cluster if index != chosen
                )
            continue

        if group_id in split_site_group_ids:
            counts = plan.allocations[group_id]
            occupied_species = [
                species
                for species, count in counts.items()
                if species != VACANCY and count > 0
            ]
            pairs = nearest_site_pairs(candidate, indices)
            if len(occupied_species) != 1:
                raise RuntimeError(
                    f"split-site 组 {group_id} 只能包含一种物种"
                )
            species = occupied_species[0]
            if (
                counts[species] != len(pairs)
                or counts.get(VACANCY, 0) != len(pairs)
            ):
                raise RuntimeError(
                    f"split-site 组 {group_id} 不是每对「一个原子 + 一个空位」"
                )
            for first, second in pairs:
                occupied, vacant = (
                    (first, second)
                    if rng.random() < 0.5
                    else (second, first)
                )
                candidate.replace(occupied, Element(species))
                remove_indices.append(vacant)
            continue

        assignments: list[str] = []
        for species, count in plan.allocations[group_id].items():
            assignments.extend([species] * count)
        if len(assignments) != len(indices):
            raise RuntimeError(
                f"组 {group_id} 有 {len(indices)} 个位点但 "
                f"{len(assignments)} 个赋值"
            )
        rng.shuffle(assignments)
        for index, assignment in zip(sorted(indices), assignments):
            if assignment == VACANCY:
                remove_indices.append(index)
            else:
                candidate.replace(index, Element(assignment))

    if remove_indices:
        candidate.remove_sites(sorted(set(remove_indices), reverse=True))
    for property_name in (
        GROUP_PROPERTY,
        CLUSTER_INSTANCE_PROPERTY,
        CLUSTER_ROLE_PROPERTY,
    ):
        if property_name in candidate.site_properties:
            candidate.remove_site_property(property_name)
    if not candidate.is_ordered:
        raise RuntimeError("整数赋值后候选仍然无序")

    # 故意不在这里做原胞/Niggli 约化：随机尝试会被去重丢掉，
    # 对丢弃的候选取约化纯属浪费。调用方在去重通过后再约化。
    return candidate


def candidate_invariant(structure: Structure, cutoff: float = 3.5) -> tuple:
    """
    便宜的「先排除明显不同」用的不变量：
    每个位点的 (物种, 该位点 cutoff 内各种物种的近邻数)。

    平移/旋转/位点顺序都不影响它，因此同一晶体的不同写法会得到相同的
    不变量；但这只是加速用的近似判据，真正的重复判定仍然交给
    StructureMatcher，所以不变量即使不够严格也不会误删不同的结构。
    """
    distances = structure.lattice.get_all_distances(
        structure.frac_coords, structure.frac_coords
    )
    symbols = [str(site.specie) for site in structure.sites]
    signature = []
    for index, symbol in enumerate(symbols):
        neighbors: dict[str, int] = {}
        for other_index, other_symbol in enumerate(symbols):
            if other_index == index:
                continue
            if distances[index, other_index] > cutoff:
                continue
            neighbors[other_symbol] = neighbors.get(other_symbol, 0) + 1
        signature.append((symbol, tuple(sorted(neighbors.items()))))
    return tuple(sorted(signature))


def generate_unique_candidates(
    decorated_structure: Structure,
    plan: AllocationPlan,
    groups: Sequence[OccupancyGroup],
    num_candidates: int,
    max_attempts_per_candidate: int,
    seed: int,
) -> list[Structure]:
    """random 方法：随机赋值 + 去重（便宜不变量预筛 + StructureMatcher）。"""
    rng = random.Random(seed)
    matcher = StructureMatcher(
        primitive_cell=True, scale=False, attempt_supercell=False
    )
    candidates: list[Structure] = []
    invariants: list[tuple] = []
    max_attempts = max(1, num_candidates * max_attempts_per_candidate)
    for _ in range(max_attempts):
        candidate = make_ordered_candidate(
            decorated_structure, plan, groups, rng
        )
        # 先比便宜不变量：不同则不可能是重复，跳过昂贵的 StructureMatcher。
        # 不变量相同仍由 StructureMatcher 决定，因此这里不够严格也不会误删。
        invariant = candidate_invariant(candidate)
        is_duplicate = any(
            invariant == existing_invariant
            and matcher.fit(candidate, existing)
            for existing_invariant, existing in zip(invariants, candidates)
        )
        if is_duplicate:
            continue
        candidates.append(candidate)
        invariants.append(invariant)
        if len(candidates) >= num_candidates:
            break

    # 只对最终保留下来的候选取原胞 + Niggli
    return [finalize_candidate_cell(candidate) for candidate in candidates]


def snap_element_amounts_to_integers(
    structure: Structure, tolerance: float = 1e-6
) -> Structure:
    """
    把「接近整数」的元素总量修正为精确整数。

    pymatgen 的氧化态猜测要求组成是整数，而等量分数占据在非二进制小数上
    （例如 20/28）会累积浮点误差，得到 O=72.00000000000004，
    使得 ewald/enumerate 后端被静默放弃。这里把残差加到该元素所在的一个
    位点上，修正量约 1e-14，对物理结果没有影响。
    """
    amounts = structure.composition.get_el_amt_dict()
    for element_symbol, amount in amounts.items():
        target = round(float(amount))
        residual = target - float(amount)
        if residual == 0.0 or abs(residual) > tolerance:
            continue
        for index in range(len(structure) - 1, -1, -1):
            species = {
                str(specie): float(occupancy)
                for specie, occupancy in structure[index].species.items()
            }
            if len(species) != 1 or element_symbol not in species:
                continue
            updated = species[element_symbol] + residual
            if not 0.0 < updated <= 1.0 + 1e-12:
                continue
            structure.replace(
                index, Composition({element_symbol: updated})
            )
            break
    return structure


def plan_occupancy_supercell(
    decorated_structure: Structure,
    plan: AllocationPlan,
) -> Structure:
    """把组的整数计数摊成等量分数占据，供精确枚举器使用。"""
    supercell = decorated_structure.copy()
    supercell.make_supercell(plan.scaling_matrix)
    indices_by_group: dict[int, list[int]] = {}
    for index, site in enumerate(supercell):
        group_id = int(site.properties[GROUP_PROPERTY])
        indices_by_group.setdefault(group_id, []).append(index)

    for group_id, indices in indices_by_group.items():
        counts = plan.allocations[group_id]
        num_positions = len(indices)
        fractional = {
            species: count / num_positions
            for species, count in counts.items()
            if species != VACANCY and count > 0
        }
        if not fractional:
            raise ValueError(f"占据组 {group_id} 没有可枚举的物种")
        composition = Composition(fractional)
        for index in indices:
            supercell.replace(index, composition)

    for property_name in (
        GROUP_PROPERTY,
        CLUSTER_INSTANCE_PROPERTY,
        CLUSTER_ROLE_PROPERTY,
    ):
        if property_name in supercell.site_properties:
            supercell.remove_site_property(property_name)

    # 修正浮点残差，否则 pymatgen 的氧化态猜测会拒绝非整数组成
    return snap_element_amounts_to_integers(supercell)


def enumerate_ordered_candidates(
    decorated_structure: Structure,
    plan: AllocationPlan,
    num_candidates: int,
    method: str,
    symprec: float,
    ewald_algo: int = 0,
) -> list[Structure] | None:
    """
    enumerate/ewald 方法：调用 pymatgen 的精确枚举或 Ewald 排序。

    含 split site / cluster 时返回 None，由调用方回退到 random。
    """
    if plan.split_site_group_ids or plan.cluster_group_ids:
        return None

    supercell = plan_occupancy_supercell(decorated_structure, plan)
    if supercell.is_ordered:
        return [finalize_candidate_cell(supercell)]

    try:
        working = supercell.copy()
        try:
            working.add_oxidation_state_by_guess()
        except Exception:
            working = supercell

        if method == "enumerate":
            from pymatgen.transformations.advanced_transformations import (
                EnumerateStructureTransformation,
            )

            transformation = EnumerateStructureTransformation(
                min_cell_size=1,
                max_cell_size=1,
                symm_prec=symprec,
                sort_criteria="ewald",
            )
        elif method == "ewald":
            from pymatgen.transformations.standard_transformations import (
                OrderDisorderedStructureTransformation,
            )

            transformation = OrderDisorderedStructureTransformation(
                algo=ewald_algo
            )
        else:
            return None

        results = transformation.apply_transformation(
            working, return_ranked_list=max(1, num_candidates)
        )
    except Exception:
        return None

    if not isinstance(results, list):
        results = [{"structure": results}]

    matcher = StructureMatcher(
        primitive_cell=True, scale=False, attempt_supercell=False
    )
    candidates: list[Structure] = []
    for entry in results:
        structure = entry["structure"] if isinstance(entry, dict) else entry
        structure = structure.copy()
        structure.remove_oxidation_states()
        candidate = finalize_candidate_cell(structure)
        if not candidate.is_ordered:
            continue
        if any(matcher.fit(candidate, existing) for existing in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= num_candidates:
            break
    return candidates or None


# ------------------------------------------------------------
# 阶段 A 主入口：单个无序 CIF -> 有序候选
# ------------------------------------------------------------

def order_disordered_structure(
    path: Path,
    args,
    reference_composition: Composition | None,
    candidate_dir: Path,
) -> tuple[list[Structure], list[CandidateRecord], AllocationPlan, dict]:
    """
    对单个无序 CIF 做占位整数化 + 有序化枚举。

    返回 (候选结构列表, 报告行, 方案, 元信息)。
    """
    material_id = safe_material_id(path)
    source = parse_structure(path)
    source_formula = source.composition.formula

    reference_elements: set[str] = set()
    missing_reference_elements: list[str] = []
    source_composition_max_error: float | None = None
    if reference_composition is not None:
        reference_elements = {
            element.symbol for element in reference_composition.elements
        }
        source_elements = {
            element.symbol for element in source.composition.elements
        }
        missing_reference_elements = sorted(
            reference_elements - source_elements
        )
        source_composition_max_error = max(
            composition_errors(
                reference_composition, source.composition
            ).values(),
            default=0.0,
        )

    reference_formula = (
        reference_composition.formula
        if reference_composition is not None
        else ""
    )

    base_info = {
        "parent_id": material_id,
        "source_file": str(path),
        "reference_formula": reference_formula,
        "source_formula": source_formula,
        "source_num_sites": len(source),
        "source_composition_max_error": source_composition_max_error,
        "missing_reference_elements": ",".join(missing_reference_elements),
    }

    if source.is_ordered:
        record = CandidateRecord(
            status="skipped_ordered",
            cleaned_formula=source_formula,
            message="输入 CIF 已经是有序结构",
            **base_info,
        )
        print(f"[{material_id}] 输入已经有序，跳过有序化")
        return [], [record], None, base_info

    cleaned, dropped_count, dropped_sum = clean_small_occupancies(
        source,
        min_occupancy=args.min_occupancy,
        protected_elements=reference_elements,
    )

    split_site_clusters = (
        find_split_site_clusters(
            cleaned,
            max_distance=args.split_cluster_max_distance,
            occupancy_tolerance=args.split_cluster_occupancy_tolerance,
            symprec=args.symprec,
        )
        if args.detect_split_clusters
        else []
    )
    cluster_indices = {
        index
        for cluster in split_site_clusters
        for index in cluster.site_indices
    }
    split_site_pairs = (
        find_split_site_pairs(
            cleaned,
            max_distance=args.split_site_max_distance,
            occupancy_tolerance=args.split_site_occupancy_tolerance,
            excluded_indices=cluster_indices,
        )
        if args.detect_split_sites
        else []
    )
    groups, grouping_method = find_occupancy_groups(
        cleaned,
        symprec=args.symprec,
        split_site_pairs=split_site_pairs,
        split_site_clusters=split_site_clusters,
    )
    plan = build_allocation_plan(
        cleaned,
        groups,
        max_multiplier=args.max_supercell_multiplier,
        max_atoms=args.max_atoms,
        max_occupancy_error=args.max_occupancy_error,
        preserve_retained_species=args.preserve_retained_species,
        reference_composition=reference_composition,
        baseline_composition=source.composition,
        composition_tolerance=args.composition_tolerance,
        required_species=reference_elements
        & {
            element.symbol
            for element in source.composition.elements
        },
    )

    print(
        f"[{material_id}] 分组={grouping_method} "
        f"split_pairs={len(split_site_pairs)} "
        f"split_clusters={len(split_site_clusters)} "
        f"supercell={plan.multiplier} "
        f"(matrix={'x'.join(str(v) for v in plan.scaling_matrix)}) "
        f"occupancy_err(max/mean)="
        f"{plan.max_occupancy_error:.3e}/{plan.mean_occupancy_error:.3e} "
        f"atoms={plan.estimated_num_atoms}"
    )

    candidate_method = "random"
    candidates: list[Structure] | None = None
    if args.candidate_method != "random":
        ewald_algo = {"fast": 0, "complete": 1, "best_first": 2}[
            args.ewald_algo
        ]
        backends = (
            ["enumerate", "ewald"]
            if args.candidate_method == "auto"
            else [args.candidate_method]
        )
        for backend in backends:
            candidates = enumerate_ordered_candidates(
                cleaned,
                plan,
                num_candidates=args.num_candidates,
                method=backend,
                symprec=args.symprec,
                ewald_algo=ewald_algo,
            )
            if candidates:
                candidate_method = backend
                break
            print(
                f"[{material_id}] {backend} 后端不可用"
                "（enumlib 缺失、占据非整数或含 split site），"
                "回退 random"
            )

    if not candidates:
        candidates = generate_unique_candidates(
            cleaned,
            plan,
            groups,
            num_candidates=args.num_candidates,
            max_attempts_per_candidate=args.max_attempts_per_candidate,
            seed=stable_seed(args.seed, material_id),
        )
        candidate_method = "random"

    if not candidates:
        raise RuntimeError(f"[{material_id}] 没有生成出任何有序候选")

    candidate_dir.mkdir(parents=True, exist_ok=True)
    records: list[CandidateRecord] = []
    for candidate_index, structure in enumerate(candidates):
        output_path = candidate_dir / f"candidate_{candidate_index:04d}.cif"
        write_cif_atomic(structure, output_path)

        messages: list[str] = []
        if missing_reference_elements:
            messages.append(
                "参考化学式中的元素在输入中缺失: "
                + ", ".join(missing_reference_elements)
            )
        if not plan.within_error_tolerance:
            messages.append(
                f"占据误差 {plan.max_occupancy_error:.6f} 超过 "
                f"--max-occupancy-error {args.max_occupancy_error:.6f}"
            )
        if plan.within_composition_tolerance is False:
            messages.append(
                f"组成漂移 {plan.composition_max_drift:.6f} 超过 "
                f"--composition-tolerance {args.composition_tolerance:.6f}"
            )

        if (
            plan.within_error_tolerance
            and plan.within_composition_tolerance is not False
            and not missing_reference_elements
        ):
            status = "ok"
        else:
            status = "approximation_warning"

        records.append(
            CandidateRecord(
                status=status,
                message="; ".join(messages),
                cleaned_formula=cleaned.composition.formula,
                dropped_species_count=dropped_count,
                dropped_occupancy_sum=dropped_sum,
                grouping_method=grouping_method,
                candidate_method=candidate_method,
                split_site_pair_count=len(split_site_pairs),
                split_site_cluster_count=len(split_site_clusters),
                split_site_species=",".join(
                    sorted(
                        {pair.species for pair in split_site_pairs}
                        | {cluster.species for cluster in split_site_clusters}
                    )
                ),
                supercell_multiplier=plan.multiplier,
                supercell_matrix="x".join(
                    str(value) for value in plan.scaling_matrix
                ),
                occupancy_max_error=plan.max_occupancy_error,
                occupancy_mean_error=plan.mean_occupancy_error,
                within_error_tolerance=plan.within_error_tolerance,
                composition_max_error=plan.composition_max_error,
                composition_max_drift=plan.composition_max_drift,
                within_composition_tolerance=(
                    plan.within_composition_tolerance
                ),
                candidate_index=candidate_index,
                num_atoms=len(structure),
                ordered_formula=structure.composition.formula,
                output_file=str(output_path),
                **base_info,
            )
        )

    print(
        f"[{material_id}] 生成 {len(candidates)} 个候选 "
        f"（method={candidate_method}, atoms={len(candidates[0])}）"
    )
    return candidates, records, plan, base_info


# ------------------------------------------------------------
# 阶段 B：MatterSim 单点能筛选 + top-N 弛豫 + 排序
# ------------------------------------------------------------

def order_fingerprint(
    args,
    material_id: str,
    source_hash: str,
    reference_formula: str = "",
) -> str:
    """
    有序化结果的指纹：决定候选 CIF 能否复用。

    必须覆盖所有会改变候选结构的输入，包括：
    - --ewald-algo（会影响 ewald/auto 后端的排列结果）；
    - 实际生效的参考化学式（来自 --reference-formula 或 --reference-csv），
      它会影响分配方案与必需物种。
    """
    payload = {
        "material_id": material_id,
        "source_hash": source_hash,
        "reference_formula": reference_formula,
        "ewald_algo": args.ewald_algo,
        "min_occupancy": args.min_occupancy,
        "max_occupancy_error": args.max_occupancy_error,
        "composition_tolerance": args.composition_tolerance,
        "preserve_retained_species": bool(args.preserve_retained_species),
        "detect_split_sites": bool(args.detect_split_sites),
        "detect_split_clusters": bool(args.detect_split_clusters),
        "split_site_max_distance": args.split_site_max_distance,
        "split_site_occupancy_tolerance": (
            args.split_site_occupancy_tolerance
        ),
        "split_cluster_max_distance": args.split_cluster_max_distance,
        "split_cluster_occupancy_tolerance": (
            args.split_cluster_occupancy_tolerance
        ),
        "max_supercell_multiplier": args.max_supercell_multiplier,
        "max_atoms": args.max_atoms,
        "symprec": args.symprec,
        "seed": args.seed,
        "num_candidates": args.num_candidates,
        "candidate_method": args.candidate_method,
        "max_attempts_per_candidate": args.max_attempts_per_candidate,
        "ordering_format_version": ORDERING_FORMAT_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_ordered_candidates(candidate_dir: Path, count: int):
    """
    读回已有的有序候选（断点续跑）。

    任何一个文件缺失或无法解析都返回 None（视为需要重新生成），
    这样被中断写坏的 CIF 不会让该材料永久无法续跑。
    """
    structures = []
    for index in range(count):
        path = candidate_dir / f"candidate_{index:04d}.cif"
        if not path.exists():
            return None
        try:
            structures.append(Structure.from_file(str(path)))
        except Exception as exc:
            print(
                f"[Candidates] {path.name} 无法读取（{exc}），重新生成候选"
            )
            return None
    return structures


def run_mattersim_funnel(
    material_id: str,
    candidates: list[Structure],
    candidate_fingerprint: str,
    candidate_dir: Path,
    output_dir: Path,
    args,
    potential=None,
) -> tuple[dict, dict, list[int], list[int], object]:
    """
    阶段 B：单点能筛选 -> top-N 弛豫 -> 排序。

    potential 可由调用方传入以便在多个材料之间复用（避免每个材料都
    重新读取 MatterSim 权重）；返回时把最终使用的 potential 一并带回。

    返回 (screening, relaxations, initial_order, final_order, potential)。
    """
    screening_file = output_dir / "screening.csv"
    relaxed_dir = output_dir / "relaxed"
    all_ids = list(range(len(candidates)))

    ml_fp = ml_fingerprint(args, candidate_fingerprint)
    # 注意：potential 由调用方传入并复用，这里不能重置为 None
    screening: dict[int, dict] = {}
    relaxations: dict[int, dict] = {}
    selected: list[int] = []

    if not args.rank:
        print(f"[{material_id}] --no-rank：只做无序→有序，跳过筛选与弛豫")
        return screening, relaxations, all_ids, all_ids, potential

    # ---- 单点能筛选 ----
    screening = load_screening(screening_file, ml_fp)
    pending = [index for index in all_ids if index not in screening]

    if not pending:
        print(
            f"[{material_id}] 已有全部 {len(screening)} 个单点能结果"
            "（断点续跑）"
        )
    else:
        if potential is None:
            potential = build_mattersim_potential(
                device=args.device,
                load_path=args.model,
            )
        batch_size = max(1, args.screen_batch_size)
        started = time.monotonic()
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset + batch_size]
            structures = [candidates[index] for index in chunk]
            values = screen_structures(
                potential, structures, batch_size=len(structures)
            )
            for index, (energy, energy_per_atom) in zip(chunk, values):
                screening[index] = {
                    "candidate_id": index,
                    "energy": energy,
                    "energy_per_atom": energy_per_atom,
                }
            write_screening(screening_file, screening, ml_fp)
            done = len(screening)
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = max(done, 1) / elapsed
            remaining = (len(all_ids) - done) / rate
            print(
                f"[{material_id}][Screen] {done}/{len(all_ids)} "
                f"elapsed={elapsed / 60:.1f} min "
                f"ETA={remaining / 60:.1f} min",
                flush=True,
            )

    initial_order = sorted(
        all_ids,
        key=lambda index: screening.get(index, {}).get(
            "energy_per_atom", float("inf")
        ),
    )

    # ---- 取筛选最低的 n_relax 个做批量弛豫 ----
    n_relax = (
        args.n_relax if args.n_relax is not None else min(len(all_ids), 20)
    )
    n_relax = max(1, min(n_relax, len(all_ids)))
    selected = initial_order[:n_relax]
    print(f"[{material_id}][Relax] 筛选前 {n_relax} 名进入弛豫")

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
            f"[{material_id}][Relax] 已有全部 {len(relaxations)} 个弛豫结果"
            "（断点续跑）"
        )

    for offset in range(0, len(pending), chunk_size):
        chunk = pending[offset:offset + chunk_size]
        structures = [candidates[index] for index in chunk]
        max_natoms = max(
            args.relax_natoms_per_batch,
            max(len(structure) for structure in structures),
        )
        results = relax_structures(
            potential=potential,
            structures=structures,
            fmax=args.fmax,
            steps=args.relax_steps,
            relax_cell=args.relax_cell,
            optimizer=args.relax_optimizer,
            max_natoms_per_batch=max_natoms,
            timeout=args.relax_timeout,
            label=f"{material_id}-chunk-{offset}",
        )
        for index, result in zip(chunk, results):
            relaxed_file = write_relaxation(
                result, index, relaxed_dir, ml_fp
            )
            result["relaxed_file"] = (
                "" if relaxed_file is None else str(relaxed_file)
            )
            relaxations[index] = result
            print(
                f"[{material_id}][Relax] candidate={index:04d} "
                f"converged={result['converged']} "
                f"steps={result['steps']} "
                f"E/atom={result['energy_per_atom']} "
                f"error={result['error'] or '-'}",
                flush=True,
            )

    # ---- 排序：收敛优先，其次每原子能量 ----
    usable = [
        index
        for index in selected
        if relaxations.get(index, {}).get("energy_per_atom") is not None
    ]
    converged = [
        index for index in usable if relaxations[index].get("converged")
    ]
    final_order = sorted(
        converged or usable,
        key=lambda index: relaxations[index]["energy_per_atom"],
    )
    if not final_order:
        print(f"[{material_id}][Rank] 没有可用的弛豫能量，回退单点能顺序")
        final_order = [index for index in initial_order if index in selected]

    return screening, relaxations, initial_order, final_order, potential


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

def write_ordering_report(path: Path, records: Sequence[CandidateRecord]):
    """阶段 A 报告（对应 OBELIX 的 ordering_report.csv）。"""
    path = Path(path)
    field_names = [
        "parent_id",
        "source_file",
        "status",
        "message",
        "reference_formula",
        "source_formula",
        "cleaned_formula",
        "source_num_sites",
        "dropped_species_count",
        "dropped_occupancy_sum",
        "grouping_method",
        "candidate_method",
        "split_site_pair_count",
        "split_site_cluster_count",
        "split_site_species",
        "supercell_multiplier",
        "supercell_matrix",
        "occupancy_max_error",
        "occupancy_mean_error",
        "within_error_tolerance",
        "source_composition_max_error",
        "composition_max_error",
        "composition_max_drift",
        "within_composition_tolerance",
        "missing_reference_elements",
        "candidate_index",
        "num_atoms",
        "ordered_formula",
        "output_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            row = {name: getattr(record, name) for name in field_names}
            writer.writerow(row)


SUMMARY_ALL_FIELDS = [
    "material_id",
    "status",
    "num_candidates",
    "best_candidate_id",
    "best_energy_per_atom_ev",
    "best_file",
]


def error_summary_row(material_id: str, path: Path, exc: Exception) -> dict:
    """单个材料失败时写入汇总表的行（不中断整个批处理）。"""
    return {
        "material_id": material_id,
        "status": f"error: {type(exc).__name__}: {exc}"[:200],
        "num_candidates": 0,
        "best_candidate_id": "",
        "best_energy_per_atom_ev": "",
        "best_file": str(path),
    }


def write_summary_all(output_root: Path, rows: Sequence[dict]) -> Path:
    """
    把已完成材料的汇总行原子写入 summary_all.csv（可随时中断）。

    与旧版本不同，这里会**合并已有文件**（按 material_id）：多次运行、
    增量扩目录或中途重启都不会丢掉之前材料的记录。
    """
    path = Path(output_root) / "summary_all.csv"

    merged: dict[str, dict] = {}
    if path.exists():
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("material_id"):
                        merged[row["material_id"]] = row
        except Exception as exc:
            print(f"[Summary] 读取已有 {path.name} 失败（将覆盖）: {exc}")

    for row in rows:
        material_id = str(row.get("material_id", ""))
        merged[material_id] = {
            field: row.get(field, "") for field in SUMMARY_ALL_FIELDS
        }

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_ALL_FIELDS)
        writer.writeheader()
        for material_id in sorted(merged):
            writer.writerow(merged[material_id])
    temporary.replace(path)
    return path


def write_ranking(path: Path, rows: Sequence[dict], field_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


RANKING_FIELDS = [
    "candidate_id",
    "candidate_file",
    "ordered_formula",
    "num_atoms",
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


def ordered_output_path(output_dir: Path, material_id: str, index: int) -> Path:
    """与 OBELIX 一致的命名：<id>_ordered.cif / <id>_ordered_01.cif。"""
    suffix = "" if index == 0 else f"_{index:02d}"
    return output_dir / f"{material_id}_ordered{suffix}.cif"


# ------------------------------------------------------------
# 命令行
# ------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="无序 → 有序 结构搜索 + MatterSim 筛选/弛豫",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("输入输出")
    io_group.add_argument(
        "--input",
        required=True,
        help="无序 CIF 文件或目录（目录下所有 CIF 都会被处理）",
    )
    io_group.add_argument(
        "--output",
        default="ordered_results",
        help="结果根目录，每个材料写在自己的子目录下",
    )
    io_group.add_argument(
        "--reference-csv",
        default=None,
        help=(
            "可选参考化学式表（列名同 OBELIX/all.csv："
            "ID + True Composition），用于约束组成漂移"
        ),
    )
    io_group.add_argument(
        "--reference-formula",
        default=None,
        help="可选：直接给出参考化学式（单个材料时更方便）",
    )

    clean_group = parser.add_argument_group("阶段 A：占据清洗")
    clean_group.add_argument(
        "--min-occupancy",
        type=float,
        default=0.01,
        help="丢弃占据低于该阈值的物种",
    )
    clean_group.add_argument(
        "--max-occupancy-error",
        type=float,
        default=0.05,
        help="单类别允许的最大占据近似误差",
    )
    clean_group.add_argument(
        "--composition-tolerance",
        type=float,
        default=0.05,
        help="允许的组成漂移（原子分数）",
    )
    clean_group.add_argument(
        "--preserve-retained-species",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保证每个保留物种至少 1 个原子",
    )

    split_group = parser.add_argument_group("阶段 A：split site")
    split_group.add_argument(
        "--detect-split-sites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把同元素近邻的两个部分占据位点当作互斥位点对",
    )
    split_group.add_argument(
        "--split-site-max-distance",
        type=float,
        default=1.2,
        help="split-site 对的最大间距 Å",
    )
    split_group.add_argument(
        "--split-site-occupancy-tolerance",
        type=float,
        default=0.1,
        help="split-site 对占据和与 1 的允许偏差",
    )
    split_group.add_argument(
        "--detect-split-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把「一个中心 + 多个卫星」的一原子簇当作互斥位点集",
    )
    split_group.add_argument(
        "--split-cluster-max-distance",
        type=float,
        default=1.5,
        help="cluster 中心-卫星的最大距离 Å",
    )
    split_group.add_argument(
        "--split-cluster-occupancy-tolerance",
        type=float,
        default=0.1,
        help="cluster 占据和与 1 的允许偏差",
    )

    supercell_group = parser.add_argument_group("阶段 A：超胞与整数化")
    supercell_group.add_argument(
        "--max-supercell-multiplier",
        type=int,
        default=8,
        help="对角超胞行列式的上限",
    )
    supercell_group.add_argument(
        "--max-atoms",
        type=int,
        default=128,
        help="有序候选的最大原子数",
    )
    supercell_group.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="对称性分析的容差",
    )

    candidate_group = parser.add_argument_group("阶段 A：候选生成")
    candidate_group.add_argument(
        "--num-candidates",
        type=int,
        default=200,
        help="每个材料生成的有序候选数（LSWO 流程用 200）",
    )
    candidate_group.add_argument(
        "--candidate-method",
        choices=("random", "enumerate", "ewald", "auto"),
        default="random",
        help=(
            "random：随机赋值 + StructureMatcher 去重；"
            "enumerate：enumlib 精确枚举（需要 enum.x/makestr.x）；"
            "ewald：pymatgen Ewald 排序；auto：依次尝试"
        ),
    )
    candidate_group.add_argument(
        "--ewald-algo",
        choices=("fast", "complete", "best_first"),
        default="fast",
    )
    candidate_group.add_argument(
        "--max-attempts-per-candidate",
        type=int,
        default=20,
        help="random 方法下每个候选的最大尝试次数",
    )
    candidate_group.add_argument("--seed", type=int, default=17)

    rank_group = parser.add_argument_group("阶段 B：MatterSim 筛选/弛豫")
    rank_group.add_argument(
        "--rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否做 MatterSim 筛选与弛豫（--no-rank 只做有序化）",
    )
    rank_group.add_argument(
        "--n-relax",
        type=int,
        default=None,
        help="筛选后做弛豫的候选数，默认 min(num-candidates, 20)",
    )
    rank_group.add_argument(
        "--keep-top",
        type=int,
        default=0,
        help="额外把最低能的 N 个写成 <id>_ordered[_XX].cif，0 表示只写最优",
    )
    rank_group.add_argument("--fmax", type=float, default=0.1)
    rank_group.add_argument("--relax-steps", type=int, default=500)
    rank_group.add_argument(
        "--relax-cell",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="弛豫时是否同时优化晶胞（原本 LSWO 流程为固定晶胞）",
    )
    rank_group.add_argument(
        "--relax-optimizer", choices=("FIRE", "BFGS"), default="FIRE"
    )
    rank_group.add_argument("--screen-batch-size", type=int, default=8)
    rank_group.add_argument(
        "--relax-natoms-per-batch", type=int, default=128
    )
    rank_group.add_argument("--relax-chunk-size", type=int, default=4)
    rank_group.add_argument(
        "--relax-timeout",
        type=float,
        default=0.0,
        help="单个结构弛豫的墙钟时间上限（秒），0 表示不限制",
    )
    rank_group.add_argument(
        "--model",
        default=None,
        help=(
            "MatterSim 模型：mattersim-v1.0.0-1M / -5M 或本地 .pth 路径。"
            "注意 .pth 通过 pickle 加载，必须来自可信来源"
        ),
    )
    rank_group.add_argument(
        "--device",
        default="auto",
        help="筛选/弛豫设备：auto / cpu / cuda / mps",
    )
    return parser


def validate_args(args) -> None:
    if args.num_candidates < 1:
        raise ValueError("--num-candidates 必须 ≥ 1")
    if args.n_relax is not None and args.n_relax < 1:
        raise ValueError("--n-relax 必须 ≥ 1")
    if args.keep_top < 0:
        raise ValueError("--keep-top 不能为负")
    if args.min_occupancy < 0:
        raise ValueError("--min-occupancy 不能为负")
    if args.max_occupancy_error < 0:
        raise ValueError("--max-occupancy-error 不能为负")
    if args.composition_tolerance < 0:
        raise ValueError("--composition-tolerance 不能为负")
    if args.max_supercell_multiplier < 1:
        raise ValueError("--max-supercell-multiplier 必须 ≥ 1")
    if args.max_atoms < 1:
        raise ValueError("--max-atoms 必须 ≥ 1")
    if args.relax_steps < 1:
        raise ValueError("--relax-steps 必须 ≥ 1")
    if args.relax_timeout < 0:
        raise ValueError("--relax-timeout 不能为负")
    if args.screen_batch_size < 1:
        raise ValueError("--screen-batch-size 必须 ≥ 1")
    if args.relax_chunk_size < 1:
        raise ValueError("--relax-chunk-size 必须 ≥ 1")
    if args.relax_natoms_per_batch < 1:
        raise ValueError("--relax-natoms-per-batch 必须 ≥ 1")
    if args.max_attempts_per_candidate < 1:
        raise ValueError("--max-attempts-per-candidate 必须 ≥ 1")
    if args.reference_csv and args.reference_formula:
        raise ValueError(
            "--reference-csv 与 --reference-formula 只能给出一个"
        )


def safe_material_id(path: Path) -> str:
    """
    从输入文件名取 material_id，并拒绝会逃出 --output 的名字。

    文件名会被直接当作输出子目录名，因此 `...cif`（stem 为 `..`）这类
    名字必须拒绝，否则输出会写到 --output 的上一级目录并覆盖已有文件。
    """
    material_id = path.stem
    if (
        not material_id
        or material_id in (".", "..")
        or "/" in material_id
        or "\\" in material_id
        or Path(material_id).name != material_id
    ):
        raise ValueError(
            f"非法的输入文件名（不能作为输出目录名）: {path.name!r}"
        )
    return material_id


def select_input_files(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入不存在: {input_path}")
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in (".cif", ".mcif")
    )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_files = select_input_files(Path(args.input))
    if not input_files:
        raise FileNotFoundError(f"{args.input} 下没有 CIF 文件")

    references: dict[str, Composition] = {}
    if args.reference_csv:
        references = load_reference_compositions(Path(args.reference_csv))
        print(f"[Reference] 载入 {len(references)} 条参考化学式")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    # MatterSim 模型在 batch 内复用：只加载一次，跨材料传递
    potential = None

    for path in input_files:
        material_id = safe_material_id(path)
        output_dir = output_root / material_id
        if not output_dir.resolve().is_relative_to(output_root.resolve()):
            raise ValueError(
                f"material_id {material_id!r} 会写到 --output 之外，已拒绝"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir = output_dir / "candidates"

        print(f"\n================ {material_id} ================")

        if args.reference_formula:
            reference_composition = Composition(args.reference_formula)
        else:
            reference_composition = references.get(material_id)

        source_hash = file_hash(path)
        ordering_fp = order_fingerprint(
            args,
            material_id,
            source_hash,
            reference_formula=(
                reference_composition.formula
                if reference_composition is not None
                else ""
            ),
        )

        meta_file = candidate_dir / "candidates_meta.json"
        meta = read_json(meta_file) if meta_file.exists() else {}
        existing = None
        if meta.get("fingerprint") == ordering_fp:
            existing = load_ordered_candidates(
                candidate_dir, int(meta.get("count", 0))
            )
            if existing:
                print(
                    f"[{material_id}] 复用已有 {len(existing)} 个有序候选"
                    f"（断点续跑，fingerprint={ordering_fp}）"
                )

        if existing:
            candidates = existing
        else:
            try:
                (
                    candidates,
                    records,
                    plan,
                    base_info,
                ) = order_disordered_structure(
                    path,
                    args,
                    reference_composition,
                    candidate_dir,
                )
            except Exception as exc:
                # 单个材料的有序化失败不应该中断整批
                print(
                    f"[{material_id}] 有序化失败，跳过: "
                    f"{type(exc).__name__}: {exc}"
                )
                summary_rows.append(
                    error_summary_row(material_id, path, exc)
                )
                write_summary_all(output_root, summary_rows)
                continue
            if not candidates:
                # 输入已经是有序结构：只写报告与 summary
                write_ordering_report(
                    output_dir / "ordering_report.csv", records
                )
                write_json_atomic(
                    output_dir / "summary.json",
                    {
                        "material_id": material_id,
                        "source_file": str(path),
                        "reference_formula": (
                            reference_composition.formula
                            if reference_composition is not None
                            else None
                        ),
                        "status": records[0].status,
                        "message": records[0].message,
                        "ordered_candidate_count": 0,
                        "ordering_fingerprint": ordering_fp,
                        "mattersim_enabled": False,
                        "num_atoms": records[0].source_num_sites,
                        "formula": records[0].source_formula,
                    },
                )
                summary_rows.append(
                    {
                        "material_id": material_id,
                        "status": records[0].status,
                        "num_candidates": 0,
                        "best_candidate_id": "",
                        "best_energy_per_atom_ev": "",
                        "best_file": "",
                    }
                )
                write_summary_all(output_root, summary_rows)
                continue

            write_ordering_report(
                output_dir / "ordering_report.csv", records
            )
            write_json_atomic(
                candidate_dir / "candidates_meta.json",
                {
                    "fingerprint": ordering_fp,
                    "source_file": str(path),
                    "source_hash": source_hash,
                    "count": len(candidates),
                    "num_atoms": len(candidates[0]),
                    "supercell_multiplier": plan.multiplier,
                    "supercell_matrix": list(plan.scaling_matrix),
                    "grouping_method": records[0].grouping_method,
                    "candidate_method": records[0].candidate_method,
                    "occupancy_max_error": plan.max_occupancy_error,
                    "occupancy_mean_error": plan.mean_occupancy_error,
                },
            )

        # ---------------- 阶段 B ----------------
        try:
            (
                screening,
                relaxations,
                initial_order,
                final_order,
                potential,
            ) = run_mattersim_funnel(
                material_id,
                candidates,
                ordering_fp,
                candidate_dir,
                output_dir,
                args,
                potential=potential,
            )
        except MattersimUnavailable:
            # 环境没装 mattersim / 权重不可用：整批都不可能成功，直接中止
            raise
        except Exception as exc:
            print(
                f"[{material_id}] 筛选/弛豫失败，跳过: "
                f"{type(exc).__name__}: {exc}"
            )
            summary_rows.append(error_summary_row(material_id, path, exc))
            write_summary_all(output_root, summary_rows)
            continue

        relaxed_dir = output_dir / "relaxed"
        initial_ranks = {
            index: rank for rank, index in enumerate(initial_order, start=1)
        }
        relaxed_ranks = {
            index: rank for rank, index in enumerate(final_order, start=1)
        }

        # 分等级写出弛豫结构（对应 run_mattersim_resume.py 的 rank_*.cif）
        for rank, index in enumerate(final_order, start=1):
            structure = relaxations.get(index, {}).get("structure")
            if structure is None:
                continue
            write_cif_atomic(
                structure,
                relaxed_dir
                / f"rank_{rank:03d}_candidate_{index:04d}.cif",
            )

        # 最优结构（对应 OBELIX 的 <id>_ordered.cif 命名）
        best_index = final_order[0] if final_order else 0
        best_structure = (
            relaxations.get(best_index, {}).get("structure")
            or candidates[best_index]
        )
        best_file = ordered_output_path(output_dir, material_id, 0)
        write_cif_atomic(best_structure, best_file)

        keep_paths = [str(best_file)]
        if args.keep_top > 0 and args.rank:
            for rank, index in enumerate(final_order[1:args.keep_top], start=1):
                structure = relaxations.get(index, {}).get("structure")
                if structure is None:
                    continue
                path_out = ordered_output_path(
                    output_dir, material_id, rank
                )
                write_cif_atomic(structure, path_out)
                keep_paths.append(str(path_out))

        if args.rank:
            rows = []
            for index in initial_order:
                screen = screening.get(index, {})
                relaxed = relaxations.get(index, {})
                rows.append(
                    {
                        "candidate_id": index,
                        "candidate_file": str(
                            (
                                candidate_dir
                                / f"candidate_{index:04d}.cif"
                            ).relative_to(output_root)
                        ),
                        "ordered_formula": candidates[
                            index
                        ].composition.formula,
                        "num_atoms": len(candidates[index]),
                        "initial_energy_ev": screen.get("energy", ""),
                        "initial_energy_per_atom_ev": screen.get(
                            "energy_per_atom", ""
                        ),
                        "initial_rank": initial_ranks[index],
                        "selected_for_relaxation": index in initial_order[
                            : (args.n_relax or min(len(candidates), 20))
                        ],
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
            write_ranking(output_dir / "ranking.csv", rows, RANKING_FIELDS)

        best_record = relaxations.get(best_index, {})
        summary = {
            "material_id": material_id,
            "source_file": str(path),
            "reference_formula": (
                reference_composition.formula
                if reference_composition is not None
                else None
            ),
            "ordered_candidate_count": len(candidates),
            "ordering_fingerprint": ordering_fp,
            "mattersim_enabled": bool(args.rank),
            "screened_count": len(screening),
            "relaxed_count": len(relaxations),
            "converged_count": sum(
                1 for item in relaxations.values() if item.get("converged")
            ),
            "num_atoms": len(best_structure),
            "formula": best_structure.composition.formula,
            "best_candidate_id": best_index,
            "best_initial_rank": initial_ranks.get(best_index),
            "best_relaxed_rank": relaxed_ranks.get(best_index),
            "best_energy_ev": best_record.get("energy"),
            "best_energy_per_atom_ev": best_record.get("energy_per_atom"),
            "best_converged": best_record.get("converged"),
            "fmax_ev_per_angstrom": args.fmax,
            "relax_steps": args.relax_steps,
            "relax_cell": bool(args.relax_cell),
            "best_structure_file": str(best_file),
            "keep_top_files": keep_paths,
        }
        write_json_atomic(output_dir / "summary.json", summary)

        write_json_atomic(
            output_dir / "results.json",
            {
                "summary": summary,
                "run_config": {
                    key: value
                    for key, value in vars(args).items()
                },
                "screening": [
                    screening[index] for index in sorted(screening)
                ],
                "relaxation": [
                    {
                        key: value
                        for key, value in relaxations[index].items()
                        if key != "structure"
                    }
                    for index in sorted(relaxations)
                ],
                "ranking": final_order,
            },
        )

        summary_rows.append(
            {
                "material_id": material_id,
                "status": "ok",
                "num_candidates": len(candidates),
                "best_candidate_id": best_index,
                "best_energy_per_atom_ev": best_record.get(
                    "energy_per_atom", ""
                ),
                "best_file": str(best_file),
            }
        )
        write_summary_all(output_root, summary_rows)

        print(
            f"[{material_id}] 最优 candidate={best_index} "
            f"E/atom={best_record.get('energy_per_atom')} "
            f"-> {best_file}"
        )

    if summary_rows:
        print(f"\n汇总: {write_summary_all(output_root, summary_rows)}")
    print(f"输出目录: {output_root}")

    succeeded = [row for row in summary_rows if row.get("status") == "ok"]
    if summary_rows and not succeeded:
        # 所有材料都失败：绝不能返回 0，否则无人值守任务会「假成功」
        print(
            f"错误: {len(summary_rows)} 个材料全部失败，没有任何候选生成成功",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
