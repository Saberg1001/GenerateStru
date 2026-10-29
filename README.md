# GenerateStru

面向锂离子固态电解质的结构生成与弛豫流程。给定一个基体结构和目标化学式，自动完成
**元素取代、电荷补偿、间隙位点搜索、MatterSim 单点能筛选和批量弛豫**，并按能量输出候选结构。

本 README 只介绍核心的两个脚本：

| 文件 | 作用 |
| --- | --- |
| `main/generate_stru.py` | 单目标引擎。处理一次完整的 host → dopant 取代，生成并排序候选。 |
| `main/batch_generate.py` | 批量编排器。读取化学式清单，规划多阶段取代路径，逐个目标调用引擎，记录进度和汇总。 |

`order_stru.py`、`summarize_results.py`、PPT 脚本等属于辅助工具，不在本文范围内。

---

## 1. 工作流概览

```text
formula/*.csv（ID / 化学式 / base / 推荐化学式）
        │
        ▼
batch_generate.py
  ├─ 读取 profile（父体库、输出目录、默认参数）
  ├─ 为每个目标选择 parent
  ├─ 规划 1 个或多个 host → dopant 取代阶段
  ├─ 按整条路径计算一次超胞倍数
  └─ 逐阶段调用 generate_stru.py
        │
        ▼
generate_stru.py
  ├─ 解析基体结构（本地 CIF 或 Materials Project）
  ├─ 约化到 primitive / 保持给定胞
  ├─ 规划超胞，使目标组成可以整数化
  ├─ 计算取代数、掺杂数和 Li 补偿数
  ├─ Wyckoff 分析 + 几何间隙位点搜索
  ├─ 生成候选构型（可断点续跑）
  ├─ MatterSim 单点能筛选
  ├─ 前 N 名候选批量弛豫
  └─ 按弛豫能量排序，输出 CIF/CSV/JSON
        │
        ▼
output/<project>/
  ├─ <target formula>/      最终结构
  ├─ _stages/               多阶段中间结构
  ├─ progress.csv           任务状态
  ├─ plan.csv               阶段规划
  ├─ run.log                实时日志
  └─ summary.log            逐目标汇总
```

---

## 2. 环境安装

要求 Python 3.10–3.12（`pymatgen 2024.10.29` 不支持 3.13）。

### 使用 uv（推荐）

项目已提供 `pyproject.toml` 和 `uv.lock`：

```bash
uv sync --frozen
```

`uv sync` 会根据锁文件创建 `.venv/` 并安装完全一致的依赖。`.venv/` 已在
`.gitignore` 中，不要提交或跨机器拷贝；到服务器后重新执行 `uv sync` 即可。

### 使用 pip

```bash
python -m pip install -r requirements.txt
```

MatterSim 使用 PyTorch。锁文件中的 `torch==2.2.1+cu118` 是 CUDA 11.8 轮子，
没有 GPU 的机器也可以安装并运行，`--device auto` 会自动回退到 CPU。

### 模型文件

默认模型路径：

```text
models/mattersim-v1.0.0-1M.pth
```

也可以通过环境变量或命令行指定：

```bash
export MATTERSIM_MODEL=/path/to/mattersim-v1.0.0-1M.pth
# 或
python main/generate_stru.py ... --model /path/to/model.pth
```

`models/` 不在 Git 中，部署到服务器时需要单独拷贝模型文件。`.pth` 通过 pickle
加载，只应使用可信来源的权重。

---

## 3. 目录约定

```text
GenerateStru/
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── formula/                 # 输入化学式清单（TSV 格式）
├── profiles/                # 项目配置（JSON）
├── models/                  # MatterSim 权重（不进 Git）
├── main/
│   ├── generate_stru.py
│   ├── generation_params.py # 用户参数统一定义
│   ├── batch_generate.py
│   └── basestru/            # 本地基体 CIF
└── output/                  # 结果
```

---

## 4. 输入格式

### 4.1 化学式清单

文件扩展名可以是 `.csv`，但当前按 **Tab 分隔**读取，列名固定为：

| 列 | 必填 | 说明 |
| --- | --- | --- |
| `ID` | 是 | 任务 ID，例如 `exp_001` |
| `化学式` | 是 | 目标化学式，例如 `Li5.7PS4.7Cl1.3` |
| `base` | 是 | 父体族标签，需要能在 profile 的 `parent_library` 中找到 |
| `推荐化学式` | 否 | 配合 `--use-recommended` 使用，非空时优先于 `化学式` |

示例：

```text
ID	化学式	base	推荐化学式
exp_001	Li5.7PS4.7Cl1.3	LPSC
exp_002	Li5.5PS4.5Cl1.5	LPSC
```

化学式支持小数下标和括号，例如 `Li10Ge(PS6)2`、`Li9.54Si1.74P1.44S11.7Br0.3`。

### 4.2 通用 profile

`profiles/default.json` 是不含项目特有配置的通用 profile，直接使用内置父体库：

```json
{
  "csv": "formula/experiment.csv",
  "output_root": "output/experiment",
  "defaults": {
    "no_mp": true,
    "adaptive_budget": true,
    "n_candidates": 200,
    "n_relax": 10,
    "fmax": 0.05,
    "relax_steps": 1000,
    "relax_optimizer": "FIRE",
    "device": "auto",
    "max_supercell_atoms": 2000,
    "min_formula_units": 100
  }
}
```

项目需要特殊父体或本地基体约束时，再在 profile 中增加对应字段：

```json
{
  "label": ["LPSC"],
  "require_local_base": true,
  "parent_library": {
    "LPSC": ["Li6PS5Cl", "Li5.4PS4.4Cl1.6", "Li5.5PS4.5Cl1.5"]
  },
  "known_bases": ["Li6PS5Cl", "Li5.4PS4.4Cl1.6"]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `csv` | 输入清单，相对项目根目录 |
| `output_root` | 输出根目录，相对项目根目录 |
| `require_local_base` | 为 `true` 时，父体必须来自本地 CIF 或同批次已生成的任务 |
| `parent_library` | 每个 `base` 标签可选的父体列表，按顺序尝试；不写则使用内置库 |
| `known_bases` | 即使不在清单中也可以作为起点的稳定父体；不写则使用内置集合 |
| `label` | 只运行指定 base 标签 |
| `defaults` | 覆盖命令行默认值，包括 `adaptive_budget` |

命令行参数优先级高于 profile。

### 4.3 基体来源

`generate_stru.py` 按以下顺序查找基体结构：

1. `--base-structure` 直接指定的文件；
2. `--basestru-dir`（默认 `main/basestru/`）中与 `--base-formula` 匹配的 CIF；
3. Materials Project（需要 `MP_API_KEY`，且未被 `--no-mp` 禁止）。

本地基体匹配会忽略大小写、空格和部分格式差异。批量模式下还会查找：

```text
output/
output_local/
<profile output_root>/
<profile output_root>/_stages/
```

批量模式下，如果目标等于父体，任务会走 **base-only** 路径：直接复制已有 CIF 作为结果，
不生成候选、不加载 MatterSim。

---

## 5. 快速开始

以下命令均在项目根目录执行。

### 5.1 查看批量规划，不运行

```bash
uv run main/batch_generate.py --profile profiles/default.json --dry-run
```

输出每个任务的父体、阶段数和阶段详情，并写出 `plan.csv`。`--list` 行为类似。

### 5.2 跑全部任务

```bash
uv run main/batch_generate.py --profile profiles/default.json
```

### 5.3 只跑指定任务

```bash
uv run main/batch_generate.py \
  --profile profiles/default.json \
  --only exp_001 exp_011
```

`--only` 接受 ID 或化学式。

### 5.4 断点续跑

```bash
uv run main/batch_generate.py \
  --profile profiles/default.json \
  --only exp_001 \
  --resume
```

`--resume` 读取 `progress.csv`，跳过状态为 `done` / `base_done` 且输出目录仍存在的任务。
`--retry-skipped` 可重试之前被跳过的任务。

每个非 dry-run 运行都会更新 `output_root/progress.csv`；`--dry-run` / `--list`
只写 `plan.csv`，不写 `progress.csv`。`progress.csv` 现在按任务 ID 合并，
`--only` 部分运行不会删除其他任务的历史记录。

LPSC-exp 的续算使用项目 profile（输入为 Tab 分隔的 `formula/LPSC-exp.tsv`）：

```bash
uv run main/batch_generate.py --profile profiles/lpsc.json --resume
```

### 5.5 CPU 冒烟测试

没有 GPU 时可以用一个小体系验证环境和 MatterSim 是否能跑通：

```bash
uv run main/generate_stru.py \
  --base-formula Li6PS5Cl \
  --target-formula Li6PS4.5Cl1.5 \
  --host S --dopant Cl \
  --base-structure main/basestru/Li6PS5Cl.cif \
  --scaling auto --min-formula-units 0 \
  --n-candidates 2 --n-relax 1 \
  --relax-steps 50 --fmax 0.2 --no-relax-cell \
  --device cpu \
  --output /tmp/stru-cpu-smoke
```

这是 **冒烟测试参数**，不是生产参数：`--no-relax-cell`、`--fmax 0.2` 和
`--relax-steps 50` 只是为了在 CPU 上快速收敛。生产运行应使用 profile 中的默认值，
让晶胞和原子位置一起弛豫到 `fmax=0.05`。

运行成功时，最后会打印 `Summary`，并在 `/tmp/stru-cpu-smoke/` 下生成
`best_structure.cif`、`ranking.csv`、`summary.json` 等文件。

### 5.6 检查/同步基体

```bash
# 报告仍然缺少哪些基体，以及依赖它们的任务
uv run main/batch_generate.py --profile profiles/default.json --check-bases

# 把 output*/<formula>/best_structure.cif 复制到 main/basestru/
uv run main/batch_generate.py --profile profiles/default.json --sync-bases
```

`--sync-bases` 只复制 `basestru/` 中尚不存在的文件，不会覆盖已有基体。

---

## 6. 批量编排器 `batch_generate.py`

### 6.1 任务加载与过滤

1. 按 Tab 读取清单；
2. 用 `--label` 过滤 `base` 标签；
3. 用 `--only` / `--limit` 过滤任务；
4. 默认按化学式去重，只保留第一次出现的行（`--keep-duplicates` 可关闭）；
5. 按依赖关系排序：如果某个任务的父体也是本批次的目标，先跑父体任务。

### 6.2 父体选择

对每个目标，依次尝试 profile 中的父体（未指定时使用内置父体库）：

1. 找到能同时平衡 parent/target 价态的模型；
2. 要求 target 电荷守恒；
3. 找到一条 `host → dopant` 的取代路径；
4. 要求最终 Li 数与 target 一致。

在候选父体中选择：

- 阶段数最少；
- 本地已有 CIF 的优先；
- 各阶段电荷偏差之和最小。

如果目标组成无法通过「取代 + Li 补偿」得到，任务会被标记为 `skipped` 并记录原因，
不会静默修改目标化学式。

### 6.3 多阶段取代与超胞

目标可能需要多个中间阶段：

```text
parent → intermediate → ... → target
```

关键点是 **整条路径只扩胞一次**：

1. 根据所有阶段的分母计算体积倍数的最小公倍数；
2. 再乘以整数因子，使超胞至少包含 `--min-formula-units` 个 f.u.（默认 100）；
3. 阶段 1 用 `--supercell-multiplier` 扩胞；
4. 后续阶段使用 `--scaling 1,1,1 --keep-cell` 保持同一胞大小。

这样每个阶段的组成都能整数化，且不同阶段/目标之间的胞大小一致，能量更可比。

如果所需超胞超过 `--max-supercell-atoms`（默认 2000），任务在规划阶段就会被拒绝，
避免跑完阶段 1 才失败。

### 6.4 自适应阶段预算

默认开启 `--adaptive-budget`。批量模式会在调用引擎前，根据 **超胞原子数** 和
**当前阶段位置** 调整每个阶段的参数：

- 超胞越小、阶段越靠前：筛选范围更宽，弛豫候选更多；
- 超胞越大、阶段越靠后：减少 `n_relax`，增加 `relax_steps`。

具体做法：

1. 原子数在 200–800 之间线性映射为 `size`（0–1）；小于 200 视为 0，大于 800 视为 1；
2. 阶段位置映射为 `stage = stage_index / (总阶段数 - 1)`（0–1）；
3. `n_candidates` 和 `n_relax` 分别用阶段因子和原子数因子相乘：

   ```text
   n_candidates = 基准值 × stage_factor(1.5 → 0.75) × size_factor(1.0 → 0.8)
   n_relax      = 基准值 × stage_factor(2.0 → 0.5)  × size_factor(1.0 → 0.45)
   ```

4. `relax_steps` 用阶段和原子数的综合复杂度插值，且不会低于基准值：

   ```text
   complexity   = 0.75 × stage + 0.25 × size
   relax_steps  = 基准值 × (1.0 → 1.5)
   ```

原子数越大，`size_factor` 越小，单阶段任务的 `n_relax` 不会像小胞那样放大；后段
任务则会进一步减少弛豫数，同时增加步数。

以默认值 `n_candidates=200, n_relax=10, relax_steps=1000` 为例：

| 场景 | n_candidates | n_relax | relax_steps |
| --- | --- | --- | --- |
| 单阶段、约 150 原子 | 300 | 20 | 1000 |
| 3 阶段、约 150 原子，阶段 1 | 300 | 20 | 1000 |
| 3 阶段、约 150 原子，阶段 2 | 220 | 12 | 1200 |
| 3 阶段、约 150 原子，阶段 3 | 150 | 5 | 1400 |
| 单阶段、约 1000 原子 | 240 | 9 | 1100 |
| 3 阶段、约 1000 原子，阶段 3 | 120 | 3 | 1500 |

自适应后的 `n_relax` 通常最低为 3（且不超过 `n_candidates`）。显式传入的 `--n-candidates`、`--n-relax`、
`--relax-steps` 会覆盖对应参数的自动调整；`--no-adaptive-budget` 可关闭全部自适应。实际采用的预算会写进 `run.log` 的
`budget:` 行。

### 6.5 调用引擎

除 base-only 阶段外，每个阶段都会启动一个 `generate_stru.py` 子进程，并传入：

```text
--n-candidates / --n-relax / --fmax / --relax-steps（由 6.4 的预算决定）
--relax-optimizer / --device
--base-structure（如果已找到本地基体）
--model（如果 models/ 下存在默认模型）
--substitution-sampler random
```

另外，批量模式会自动：

- 对 `S → O/Se/Cl/Br/I/F` 使用 `--host-site-filter free-anion-prefer`，
  优先占据自由阴离子位点，再占据骨架位点；
- 对极稀掺杂（`amount < 0.05`）把原子预算提高到至少 1500。

### 6.6 日志与进度

| 文件 | 内容 |
| --- | --- |
| `run.log` | 带时间戳的实时日志 |
| `run_raw/` | 每个任务每个阶段的完整子进程输出；目录跟随 `--log` 所在目录，不是固定放在 output 根目录 |
| `progress.csv` | 机器可读的任务状态和统计 |
| `plan.csv` | 规划结果：父体、阶段、超胞原子数、阶段详情、跳过原因 |
| `summary.log` | 运行结束后的逐目标汇总 |

`progress.csv` 的主要列：

```text
task_index, id, formula, base, parent, stages, supercell_atoms, status,
candidate_count, screened_count, selected_for_relaxation,
relaxed_count, converged_count, best_energy_per_atom_ev,
best_converged, output_dir, error, elapsed_s
```

任务状态包括：

```text
pending / planned / running / done / base_done / failed / skipped
done_resumed / skipped_resumed
```

每次运行结束会自动调用 `summarize_results.py`，同时更新
`<output_root>/summary.log` 和 `main/print-out-summary.log`。

---

## 7. 单目标引擎 `generate_stru.py`

### 7.1 处理步骤

| 步骤 | 内容 |
| --- | --- |
| 1 | 确定基体结构：`--base-structure` → `basestru/` → Materials Project |
| 2 | 约化为 primitive；`--keep-cell` 时保留给定胞 |
| 3 | 规划超胞，使目标组成可以整数化 |
| 4 | 计算 host 移除数、dopant 添加数、Li 补偿数，并报告组成偏差 |
| 5 | 分析基体 primitive 的 Wyckoff 信息 |
| 6 | 需要插入原子时，做几何间隙位点搜索 |
| 7 | 生成候选构型，写入 `candidates/`，支持断点续跑 |
| 8 | 用 MatterSim 做单点能筛选 |
| 9 | 取筛选能量最低的前 N 个候选做批量弛豫 |
| 10 | 按弛豫能量排序，输出 `best_structure.cif`、`ranking.csv`、`summary.json` |

### 7.2 化学计量与超胞

引擎需要把目标化学式中的分数下标变成整数原子数。`--scaling` 有两种模式：

- `auto`：按所有下标分母的最小公倍数自动选择体积倍数；
- 显式矩阵：例如 `--scaling 1,2,5`，体积倍数 = 三个分量之积。

`--max-supercell-atoms` 是 `auto` 模式的原子数预算。精确整数化超过预算时：

- 默认允许改用有理逼近，并显式报告组成/取代比例偏差；
- `--no-allow-stoichiometry-approximation` 则直接报错；
- 偏差超过 `--stoichiometry-tolerance`（默认 3%）仍然报错。

### 7.3 取代位点控制

| 参数 | 作用 |
| --- | --- |
| `--host-wyckoff 1c` | 只允许取代基体 primitive 中指定 Wyckoff 轨道的 host 位点 |
| `--host-site-filter all` | 所有 host 位点都可取代（默认） |
| `--host-site-filter free-anion` | 只取代「自由阴离子」位点，数量不够时直接报错 |
| `--host-site-filter free-anion-prefer` | 先占满自由阴离子位点，不够时再取代骨架位点 |
| `--host-framework-element P,Si` | 判断自由阴离子时使用的骨架元素 |
| `--host-framework-cutoff 3.0` | host 到骨架元素的最小距离阈值（Å） |
| `--substitution-sampler greedy` | 贪心分散掺杂原子，适合极稀掺杂（默认） |
| `--substitution-sampler random` | 均匀随机取位点，适合需要遍历构型的场景 |

低对称（P1）有序胞中 Wyckoff 标签可能不可用，此时应使用距离判据
`free-anion` / `free-anion-prefer`。

### 7.4 候选生成

- 纯取代且组合数不超过 2000 时，直接枚举全部组合；
- 否则按结构判重抽样，穷尽后自动提前结束；
- 因此实际候选数可能少于 `--n-candidates`；
- 需要插入补偿原子时，先做随机几何采样，再按 clearance 和位点间距筛选；
- 每个候选都记录取代位点、插入位点、移除位点和 CIF 文件。

### 7.5 MatterSim 筛选与弛豫

- 筛选：批量单点能，按 `--screen-batch-size` 分块，默认 8；
- 弛豫：取筛选能量最低的 `--n-relax` 个候选，默认 `min(n_candidates, 20)`；
- 优化器：`FIRE`（默认）或 `BFGS`；
- 是否优化晶胞：`--relax-cell` / `--no-relax-cell`，默认优化；
- 收敛标准：`--fmax`，默认 0.05 eV/Å；
- 最大步数：`--relax-steps`；
- 设备：`--device auto/cpu/cuda/mps`。

如果所有候选都没有收敛，任务仍会完成，但会打印警告；`summary.json` 中
`converged_count` 为 0。

### 7.6 断点续跑

引擎用两个指纹判断缓存是否有效：

- **候选指纹**：base/target、host/dopant、scaling、seed、候选数量等；
- **ML 指纹**：候选指纹、模型身份、fmax、步数、优化器、弛豫批次等。

指纹一致的候选、单点能和弛豫结果会被复用；修改了会影响的参数后，旧结果自动失效，
不需要手工删除 `screening.csv` 或 `relaxed/`。

---

## 8. 输出文件

### 8.1 批量输出

```text
output/<project>/
├── plan.csv
├── progress.csv
├── run.log
├── run_raw/
│   └── <task_id>_stage01.log
├── summary.log
├── <target formula>/
│   └── ...（最终阶段的完整引擎输出）
└── _stages/
    └── <task_id>/
        └── 01_<intermediate formula>/
            └── ...（中间阶段输出）
```

### 8.2 单目标输出

| 文件 | 说明 |
| --- | --- |
| `base_input.cif` | 实际使用的起始结构 |
| `base_primitive.cif` | primitive 约化后的结构 |
| `base_supercell.cif` | 扩胞后的基体 |
| `wyckoff_report.json` | Wyckoff 分析结果 |
| `candidates/candidate_XXXX.cif` | 生成的候选结构 |
| `candidates/candidates_meta.json` | 候选的取代/插入/移除位点元数据 |
| `screening.csv` | 单点能筛选结果（带指纹） |
| `relaxed/candidate_XXXX.cif` | 单个候选的弛豫结构 |
| `relaxed/candidate_XXXX.json` | 单个候选的弛豫元数据（带指纹） |
| `relaxed/rank_XXX_candidate_XXXX.cif` | 按弛豫能量排序后的结构 |
| `best_structure.cif` | 最优结构 |
| `ranking.csv` | 筛选 + 弛豫汇总表 |
| `summary.json` | 计数、收敛情况和最优能量 |
| `results.json` | 完整运行记录：组成计算、候选、筛选、弛豫、配置 |

`ranking.csv` 的主要列：

```text
candidate_id, candidate_file,
initial_energy_ev, initial_energy_per_atom_ev, initial_rank,
selected_for_relaxation, converged, relaxation_steps, relaxation_seconds,
relaxed_energy_ev, relaxed_energy_per_atom_ev, relaxed_rank,
relaxed_file, error
```

---

## 9. 常用命令速查

### 批量

```bash
# 规划
uv run main/batch_generate.py --profile profiles/default.json --dry-run

# 全量
uv run main/batch_generate.py --profile profiles/default.json

# 指定任务 + 续跑
uv run main/batch_generate.py --profile profiles/default.json --only exp_001 --resume

# 指定 base 标签
uv run main/batch_generate.py --profile profiles/default.json --label LPSC

# 限制前 5 个
uv run main/batch_generate.py --profile profiles/default.json --limit 5

# 关闭自适应预算，所有阶段使用固定参数
uv run main/batch_generate.py --profile profiles/default.json --no-adaptive-budget

# 允许回退 Materials Project
MP_API_KEY=... uv run main/batch_generate.py --profile profiles/default.json --allow-mp

# 检查基体
uv run main/batch_generate.py --profile profiles/default.json --check-bases
```

### 单目标

```bash
uv run main/generate_stru.py \
  --base-formula Li6PS5Cl \
  --target-formula Li5.7PS4.7Cl1.3 \
  --host S --dopant Cl \
  --base-structure main/basestru/Li6PS5Cl.cif \
  --scaling auto \
  --n-candidates 50 --n-relax 10 \
  --device cuda \
  --output output/manual/Li5.7PS4.7Cl1.3
```

只生成结构、不做 MatterSim 计算：

```bash
uv run main/generate_stru.py ... --no-relax
```

---

## 10. 常见问题

### 提示 `需要 mp-api` 或 `未设置 MP_API_KEY`

Materials Project 是云端 API，不需要安装本地数据库；`mp-api` 只是客户端，
已在依赖中。本地 `basestru/` 没有对应基体时：

1. 把基体 CIF 放到 `main/basestru/`；
2. 设置 `MP_API_KEY`（https://materialsproject.org/api）：项目根目录的 `.env` 会自动加载，
   也可以直接 `export MP_API_KEY=...`；`.env` 已被 Git 忽略，服务器上需要单独创建或
   复制，不要提交到仓库；
3. 批量模式默认 `no_mp=true`，需要加 `--allow-mp` 才允许回退；
4. 单目标引擎默认会自动回退 MP；想强制走 MP 可加 `--no-use-basestru`。

### 任务被标记为 SKIPPED

常见原因：

- `base` 标签没有在 profile 中配置父体；
- target 电荷不守恒；
- 找不到合法的 `host → dopant` 取代路径；
- 非掺杂元素与父体不一致；
- 重复化学式；
- 目标等于父体，但父体既没有本地 CIF，也不在 `known_bases` 中。

具体原因见 `plan.csv` 的 `error` 列和 `run.log`。

### 超胞超过原子预算

报错会给出所需体积倍数和原子数。可以：

- 提高 `--max-supercell-atoms`；
- 降低 `--min-formula-units`；
- 换一个更小的父体；
- 接受有理逼近（默认已允许），并检查偏差是否在容差内。

### 弛豫没有收敛

可以：

- 增加 `--relax-steps`；
- 放宽 `--fmax`；
- 检查超胞是否过大；
- 使用 `--no-relax-cell` 先做固定晶胞测试；
- 在 GPU 上运行，CPU 只适合冒烟测试。

### 日志中只有 `[Screen]` 没有 `[Relax]`

说明筛选阶段还没结束，或者 `--no-relax` 关闭了 ML 流程。批量模式下可以查看
`run_raw/<task>_stage01.log` 的完整输出。

### 结果目录冲突

默认会按化学式去重。确实需要跑重复组成时加 `--keep-duplicates`，但输出目录会冲突，
需要自行调整 `--output-root`。

---

## 11. 已知限制

- 当前清单按 Tab 分隔读取，不解析逗号分隔的 CSV。
- 每个阶段只支持一种 `host → dopant` 取代；复杂组成会被拆成多个阶段。
- 电荷平衡依赖内置价态表，特殊氧化态体系可能无法规划。
- 几何间隙搜索是随机采样，不保证找到所有空隙位点。
- MatterSim 能量用于相对排序，不等同于 DFT 形成能。
- 自适应预算基于原子数和阶段位置的经验规则，不是实际性能模型；可用
  `--no-adaptive-budget` 或显式参数固定预算。
- 批量任务是串行的，不会自动并行；GPU 利用率主要来自单个任务的批量化筛选/弛豫。


---

## 12. 统一参数入口

用户参数集中定义在 `main/generation_params.py` 的 `GenerationParams`。它同时用于：

- 生成 `generate_stru.py` 的 argparse 参数；
- 生成传给引擎的 CLI 参数；
- 后续批量脚本、Web 后端或前端直接构造请求对象。

```python
from main.generation_params import GenerationParams

params = GenerationParams(
    base_formula="Li6PS5Cl",
    target_formula="Li5.5PS4.5Cl1.5",
    host_element="S",
    dopant_element="Cl",
    n_candidates=200,
    n_relax=10,
)
```

新增参数时只改 `GenerationParams` 一处，CLI 和后续调用方同步生效。
