"""Single source of truth for structure-generation parameters.

The parameter fields are defined once here.  generate_stru.py builds its
argparse parser from this class, and batch jobs, Python callers or a future
frontend can construct the same object.  Execution is deterministic:
parameters are converted into generate_stru.py CLI arguments; no language model
is involved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_FORMULA_RE = re.compile(r"^[A-Za-z0-9().]+$")

# Field names that do not map to --<name-with-dashes> on the engine CLI.
_CLI_FLAGS = {
    "host_element": "--host",
    "dopant_element": "--dopant",
}
_CLI_DESTS = {
    "host_element": "host",
    "dopant_element": "dopant",
}
_REQUIRED_FIELDS = ("base_formula", "target_formula", "host_element", "dopant_element")


class GenerationParams(BaseModel):
    """All user-facing parameters for one structure-generation target."""

    base_formula: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="基础材料化学式，例如 Li6PS5Cl",
    )
    target_formula: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="目标材料化学式，例如 Li5.5PS4.5Cl1.5",
    )
    host_element: str = Field(
        ...,
        description="被掺杂/被替换元素，例如 S",
    )
    dopant_element: str = Field(
        ...,
        description="掺杂元素，例如 Cl",
    )
    compensating_element: str = Field(
        "Li",
        description="组分变化/电荷补偿元素，通常为 Li",
    )

    host_wyckoff: Optional[str] = Field(
        None,
        description="只允许取代指定 Wyckoff 轨道，例如 1c",
    )
    host_site_filter: Literal["all", "free-anion", "free-anion-prefer"] = Field(
        "all",
        description="取代位点筛选：all / free-anion / free-anion-prefer",
    )
    host_framework_element: str = Field(
        "P",
        description="自由阴离子判断用骨架元素，可逗号分隔",
    )
    host_framework_cutoff: float = Field(
        3.0,
        gt=0,
        le=20,
        description="自由阴离子距离阈值，Å",
    )
    substitution_sampler: Literal["greedy", "random"] = Field(
        "greedy",
        description="取代位点采样方式：greedy / random",
    )

    min_formula_units: int = Field(
        0,
        ge=0,
        le=10000,
        description="auto 扩胞时的最小 f.u. 数，0 表示不限制",
    )
    base_structure: Optional[str] = Field(
        None,
        description="基体结构文件；main/basestru 文件名或项目内 CIF 路径",
    )
    basestru_dir: Optional[str] = Field(
        None,
        description="本地基体结构目录；默认 main/basestru",
    )
    use_basestru: bool = Field(
        True,
        description="是否优先使用 basestru 目录中的基体",
    )
    keep_cell: bool = Field(
        False,
        description="保留给定胞，不约化为 primitive",
    )
    supercell_multiplier: Optional[int] = Field(
        None,
        ge=1,
        description="直接指定精确体积倍数；与 scaling 互斥",
    )
    scaling: str = Field(
        "auto",
        description="auto 或对角超胞 a,b,c，例如 1,2,5",
    )
    max_supercell_atoms: int = Field(
        500,
        ge=1,
        le=1_000_000,
        description="超胞原子数预算",
    )
    allow_stoichiometry_approximation: bool = Field(
        True,
        description="超出原子预算时是否允许有理逼近",
    )
    stoichiometry_tolerance: float = Field(
        0.03,
        gt=0,
        le=1,
        description="允许的最大原子分数偏差",
    )

    n_candidates: int = Field(
        200,
        ge=1,
        le=100000,
        description="生成并做单点能筛选的候选数上限",
    )
    n_samples: int = Field(
        30000,
        ge=1,
        le=10_000_000,
        description="随机几何间隙采样点数",
    )
    min_distance: float = Field(
        1.8,
        gt=0,
        le=20,
        description="间隙点到现有原子的初筛最小距离，Å",
    )
    min_site_site_distance: float = Field(
        2.0,
        gt=0,
        le=20,
        description="额外插入原子之间的最小距离，Å",
    )
    candidate_pool_size: int = Field(
        200,
        ge=1,
        le=1_000_000,
        description="从 clearance 最大的前 N 个位点中候选",
    )
    seed: int = Field(
        1234,
        ge=0,
        le=2**31 - 1,
        description="随机种子",
    )
    output: Optional[str] = Field(
        None,
        description="输出目录；调用方可以覆盖为任务目录",
    )

    relax: bool = Field(True, description="是否执行 MatterSim 筛选与弛豫")
    relax_cell: bool = Field(True, description="弛豫时是否优化晶胞")
    fmax: float = Field(
        0.05,
        gt=0,
        le=10,
        description="弛豫收敛力阈值，eV/Å",
    )
    relax_steps: int = Field(
        1000,
        ge=1,
        le=1_000_000,
        description="弛豫最大步数",
    )
    relax_optimizer: Literal["FIRE", "BFGS"] = Field(
        "FIRE",
        description="弛豫优化器",
    )
    n_relax: int = Field(
        10,
        ge=1,
        le=10000,
        description="筛选后实际做弛豫的候选数",
    )
    screen_batch_size: int = Field(
        8,
        ge=1,
        le=10000,
        description="单点能筛选批大小",
    )
    relax_natoms_per_batch: int = Field(
        128,
        ge=1,
        le=1_000_000,
        description="批量弛豫时每个 batch 的最大原子数",
    )
    relax_chunk_size: int = Field(
        4,
        ge=1,
        le=10000,
        description="每次批量弛豫的结构数",
    )
    relax_timeout: float = Field(
        0.0,
        ge=0,
        description="单个结构弛豫的墙钟时间上限，秒；0 表示不限",
    )
    model: Optional[str] = Field(
        None,
        description="势函数预置名或 models/ 下的 .pth 路径",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto",
        description="筛选/弛豫设备",
    )

    @field_validator("base_formula", "target_formula")
    @classmethod
    def _validate_formula(cls, value: str) -> str:
        value = value.strip()
        if not value or not _FORMULA_RE.fullmatch(value):
            raise ValueError("化学式只能包含元素、数字、小数点和括号")
        try:
            from pymatgen.core import Composition

            Composition(value)
        except Exception as exc:
            raise ValueError(f"无效化学式: {value}") from exc
        return value

    @field_validator("host_element", "dopant_element", "compensating_element")
    @classmethod
    def _validate_element(cls, value: str) -> str:
        value = value.strip()
        if not _ELEMENT_RE.fullmatch(value):
            raise ValueError(f"无效元素符号: {value}")
        try:
            from pymatgen.core import Element

            Element(value)
        except Exception as exc:
            raise ValueError(f"未知元素: {value}") from exc
        return value

    @field_validator("host_framework_element")
    @classmethod
    def _validate_framework_elements(cls, value: str) -> str:
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("骨架元素不能为空")
        for token in tokens:
            if not _ELEMENT_RE.fullmatch(token):
                raise ValueError(f"无效元素符号: {token}")
            try:
                from pymatgen.core import Element

                Element(token)
            except Exception as exc:
                raise ValueError(f"未知元素: {token}") from exc
        return ",".join(tokens)

    @field_validator("scaling")
    @classmethod
    def _validate_scaling(cls, value: str) -> str:
        value = value.strip().lower()
        if value == "auto":
            return value
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError("scaling 必须是 auto 或 a,b,c")
        try:
            numbers = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("scaling 分量必须是整数") from exc
        if any(number <= 0 for number in numbers):
            raise ValueError("scaling 分量必须为正整数")
        return ",".join(str(number) for number in numbers)

    @model_validator(mode="after")
    def _validate_combination(self) -> "GenerationParams":
        elements = {
            self.host_element,
            self.dopant_element,
            self.compensating_element,
        }
        if len(elements) != 3:
            raise ValueError(
                "host_element / dopant_element / compensating_element "
                "必须互不相同"
            )
        if self.n_relax > self.n_candidates:
            raise ValueError("n_relax 不能大于 n_candidates")
        if self.supercell_multiplier is not None and self.scaling != "auto":
            raise ValueError("supercell_multiplier 不能与显式 scaling 同时使用")
        return self

    def to_engine_args(
        self,
        output_dir: Path,
        engine_path: Path,
        base_structure: Optional[str] = None,
        model: Optional[str] = None,
        python_executable: Optional[str] = None,
    ) -> List[str]:
        """Convert this object into generate_stru.py CLI arguments."""
        command = [python_executable or sys.executable, str(engine_path)]

        for name in _REQUIRED_FIELDS:
            command += [self._cli_flag(name), str(getattr(self, name))]

        # Optional fields: booleans become --flag / --no-flag.
        for name, field in type(self).model_fields.items():
            if name in _REQUIRED_FIELDS or name in {
                "base_structure",
                "model",
                "output",
            }:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            flag = self._cli_flag(name)
            annotation = _unwrap_annotation(field.annotation)
            if annotation is bool:
                command.append(flag if value else _negative_flag(flag))
            else:
                command += [flag, str(value)]

        base_structure = base_structure or self.base_structure
        if base_structure:
            command += ["--base-structure", str(base_structure)]
        model = model or self.model
        if model:
            command += ["--model", str(model)]

        command += ["--output", str(output_dir)]
        return command

    @staticmethod
    def _cli_flag(name: str) -> str:
        return _CLI_FLAGS.get(name, "--" + name.replace("_", "-"))


def _unwrap_annotation(annotation: Any) -> Any:
    args = get_args(annotation)
    if args and type(None) in args:
        return next(arg for arg in args if arg is not type(None))
    return annotation


def _negative_flag(flag: str) -> str:
    return "--no-" + flag[2:]


def add_engine_arguments(
    parser: argparse.ArgumentParser,
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """Add GenerationParams fields to an argparse parser.

    ``defaults`` only overrides CLI defaults where the standalone engine keeps a
    different historical default (for example n_candidates=10 vs API 200).
    Field names, types, help, choices and required flags still come from the
    class.
    """
    defaults = defaults or {}
    for name, field in GenerationParams.model_fields.items():
        flag = GenerationParams._cli_flag(name)
        kwargs: Dict[str, Any] = {
            "dest": _CLI_DESTS.get(name, name),
            "help": field.description or name,
        }
        if field.is_required():
            kwargs["required"] = True
            # Required flags have no meaningful default in --help.
            kwargs["default"] = argparse.SUPPRESS
        else:
            kwargs["default"] = defaults.get(name, field.default)

        annotation = _unwrap_annotation(field.annotation)
        if annotation is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif get_origin(annotation) is Literal:
            kwargs["choices"] = list(get_args(annotation))
            kwargs["type"] = str
        elif annotation is int:
            kwargs["type"] = int
        elif annotation is float:
            kwargs["type"] = float
        else:
            kwargs["type"] = str

        parser.add_argument(flag, **kwargs)
