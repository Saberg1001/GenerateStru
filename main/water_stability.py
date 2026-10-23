def analyze_gas_stability(
    material_id: str | None = None,
    struct_id: str | None = None,
    symbols: list | None = None,
    positions: list | None = None,
    cell: list | None = None,
    total_energy: float | None = None,
    gases: list = None,
    temperature: float = 300.0,
    pressures: dict | None = None,
    gas_thermo_table: dict | None = None,
    make_plot: bool = False,
    save_format: str = "png",
) -> dict:
    """
    评估材料在 O2 / CO2 / H2O 气氛中的热力学稳定性 (气相腐蚀/分解判据)。

    三段机制 (能标自洽, DFT 固相 + NIST-JANAF 气相自由能):
      - O2  (开放系综): GrandPotentialPhaseDiagram, μ_O 含 T/p 修正,
        报告分解能 e_above_hull。
      - CO2 / H2O (闭系伪二元): InterfacialReactivity, 气体侧用真实 μ_gas(T,p)
        (含 ΔG_f° + 有限温气相熵 + 分压), 报告最放热反应能 minimum[1]。

    重要: 结果为 0K DFT(固相) + 有限温气相自由能的热力学判据, 不含反应动力学
    与钝化层保护, 实际稳定性可能更好。O2(开放) 与 CO2/H2O(闭系伪二元) 热力学
    含义不同, 不可直接比大小, 仅在各自意义下解读。

    Args:
        material_id: Materials Project ID (如 'mp-149')。
        struct_id: 缓存中的结构 ID。
        symbols/positions/cell: 手动输入结构时使用。
        total_energy: 目标相总能量 (eV)。不传则用 CHGNet 计算。
        gases: 要评估的气体列表, 默认 ["O2","CO2","H2O"]。
        temperature: 温度 (K), 默认 300。
        pressures: 各气体分压 (atm), 如 {"O2":0.21,"H2O":0.03}。缺省 0.21。
        gas_thermo_table: 覆盖 NIST-JANAF 热化学常数表 (高级用法)。
        make_plot: 是否输出对比图, 默认 False。
        save_format: 图片格式 (png/svg/pdf), 默认 png。
    """
    from pymatgen.core.periodic_table import Element
    from pymatgen.analysis.phase_diagram import (
        GrandPotentialPhaseDiagram, GrandPotPDEntry,
    )

    if gases is None:
        gases = ["O2", "CO2", "H2O"]
    gases = [g.upper() for g in gases]
    invalid = [g for g in gases if g not in ("O2", "CO2", "H2O")]
    if invalid:
        return {"error": f"不支持的气体 {invalid}, 仅支持 O2 / CO2 / H2O"}

    pressures = pressures or {}
    pmap = {g: float(pressures.get(g, 0.21)) for g in gases}

    # ═══════════════════════════════════════
    # 1. 目标相结构与能量
    # ═══════════════════════════════════════
    structure = universal_structure_parser(material_id, struct_id, None, symbols, positions, cell)
    if structure is None or not hasattr(structure, "composition"):
        return {"error": (
            "未获取到有效结构。请检查 material_id 是否存在/未弃用 (MP 可能返回空), "
            "或提供 struct_id / symbols+positions+cell。"
        )}

    target_comp = structure.composition
    if total_energy is None:
        try:
            total_energy = _engine.calculate_energy(structure)["energy"] * len(structure)
            energy_source = "CHGNet 0K single point"
        except Exception as e:
            return {"error": f"CHGNet 单点能计算失败: {type(e).__name__}: {e}"}
    else:
        energy_source = "user-provided"

    target_entry = ComputedEntry(
        composition=target_comp,
        energy=total_energy,
        entry_id="gas_stability_target",
        parameters={
            "potcar_symbols": [f"PBE {el.symbol}" for el in target_comp.elements],
            "run_type": "GGA",
        },
    )

    # chemsys = 样品元素 ∪ 所有气体涉及元素
    needed_elem_syms = set(str(e) for e in target_comp.elements)
    gas_elem_syms = set()
    for g in gases:
        if g == "O2":
            gas_elem_syms.add("O")
        else:
            data = gas_thermo_table.get(g) if gas_thermo_table else _GAS_JANAF.get(g)
            if data:
                gas_elem_syms.update(data["elements"].keys())
    needed_elem_syms |= gas_elem_syms

    # ═══════════════════════════════════════
    # 2. 拉取 MP 竞争相 + 统一校正
    # ═══════════════════════════════════════
    try:
        with MPRester(MP_API_KEY) as mpr:
            mp_entries = mpr.get_entries_in_chemsys(sorted(needed_elem_syms))
    except Exception as e:
        return {"error": f"MP 竞争相拉取失败 (chemsys={sorted(needed_elem_syms)}): {type(e).__name__}: {e}"}

    for entry in mp_entries:
        if entry.parameters.get("potcar_spec"):
            cleaned = []
            for spec in entry.parameters["potcar_spec"]:
                if spec is None:
                    continue
                if hasattr(spec, "model_dump"):
                    cleaned.append(spec.model_dump())
                elif isinstance(spec, dict):
                    cleaned.append(spec)
                else:
                    try:
                        cleaned.append(dict(spec))
                    except (TypeError, ValueError):
                        if hasattr(spec, "__dict__"):
                            cleaned.append(spec.__dict__)
            entry.parameters["potcar_spec"] = cleaned

    try:
        compat = MaterialsProject2020Compatibility(check_potcar_hash=False)
        processed_mp = compat.process_entries(mp_entries, clean=True)
    except Exception as e:
        return {"error": f"MP 能量校正失败: {type(e).__name__}: {e}"}

    # target_entry 保留原始能量 (CHGNet/user), 与 analyze_phase_stability 同约定;
    # CHGNet 已逼近 MP 能标, 固相校正对 target 近似可忽略。
    target_corrected = target_entry
    all_entries = processed_mp + [target_corrected]

    # 提取元素参考 μ_elem (eV/atom) — 从校正后的 MP 单质相
    elem_refs, missing_refs = _extract_element_refs(processed_mp, needed_elem_syms)
    if missing_refs:
        return {"error": f"MP 数据中缺少以下单质参考相: {missing_refs}。无法建立气相化学势。"}

    elem_refs_out = {k: round(float(v), 4) for k, v in elem_refs.items()}

    # ═══════════════════════════════════════
    # 3. 按气体分别过滤 chemsys + 建相图
    #    (避免 O2 段混入 C/H 相导致 Li2CO3 等伪产物; CO2/H2O 段同理)
    # ═══════════════════════════════════════
    sample_elem_syms = set(str(e) for e in target_comp.elements)