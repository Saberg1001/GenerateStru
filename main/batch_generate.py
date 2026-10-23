import subprocess
import os
import sys
from pathlib import Path

# 基础路径配置
BASE_CIF = "/home/ziyiguo/project/GenerateStru/main/basestru/Li5.4PS4.4Cl1Br0.6.cif"
BASE_FORMULA = "Li5.4PS4.4Cl1Br0.6"
SCRIPT = "generate_stru.py"

# 目标组成：输出目录直接用化学式命名（与 output/Li5.4PS4.4Cl1.6 保持一致）。
# 这里不再传 --output，generate_stru.py 会默认落在 output/<目标化学式> 下。
TARGET1 = "Li5.4PS4ClBr0.6O0.4"
TARGET2 = "Li5.41P0.99W0.01S4.4ClBr0.6"
TARGET3A = "Li5.45P0.95S4.4ClBr0.6Si0.05"
TARGET3B = "Li5.45P0.95S4.3ClBr0.6Si0.05Se0.1"

OUTPUT_ROOT = Path("output")

# 全局参数：单点200，弛豫50，使用 FIRE 以防 BFGS 发散
COMMON_ARGS = [
    "--n-candidates", "200",
    "--n-relax", "50",
    "--relax-optimizer", "FIRE",
    "--fmax", "0.05",
    "--relax-steps", "300",
    "--device", "cpu",
    "DUMMY" # 解决脚本中 MP_API 位置参数必填的问题
]

def run_cmd(task_name, cmd):
    print(f"\n{'='*60}")
    print(f"🚀 开始执行任务: {task_name}")
    print(f"{'='*60}")
    print("执行命令:", " ".join(cmd))
    
    # 实时打印输出
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    process.wait()
    if process.returncode != 0:
        print(f"❌ 任务 {task_name} 失败！请检查上方报错信息。")
    else:
        print(f"✅ 任务 {task_name} 完成！")

def main():
    # 确保在执行目录下
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # =========================================================
    # 任务 1: O 掺杂 (同价取代 S)
    # =========================================================
    task1_cmd = [
        sys.executable, SCRIPT,
        "--base-structure", BASE_CIF,
        "--base-formula", BASE_FORMULA,
        "--target-formula", TARGET1,
        "--host", "S", "--dopant", "O", "--compensating-element", "Li",
    ] + COMMON_ARGS
    run_cmd(f"目标 1: O 取代 S ({TARGET1})", task1_cmd)

    # =========================================================
    # 任务 2: W 掺杂 (极低浓度 0.01)
    # 警告：这需要 100 倍超胞(约1300原子)，CPU 可能会非常慢或 OOM
    # =========================================================
    task2_cmd = [
        sys.executable, SCRIPT,
        "--base-structure", BASE_CIF,
        "--base-formula", BASE_FORMULA,
        "--target-formula", TARGET2,
        "--host", "P", "--dopant", "W", "--compensating-element", "Li",
        "--max-supercell-atoms", "1500",  # 强行放宽原子数限制
    ] + COMMON_ARGS
    # 注意：如果你觉得太慢，可以注释掉下面这行
    run_cmd(f"目标 2: W 掺杂 P ({TARGET2}, 极高算力需求)", task2_cmd)

    # =========================================================
    # 任务 3: Si、Se 双掺杂 (必须分两阶段)
    # 阶段 3A: P -> Si (需要 20 倍超胞)
    # =========================================================
    task3a_cmd = [
        sys.executable, SCRIPT,
        "--base-structure", BASE_CIF,
        "--base-formula", BASE_FORMULA,
        "--target-formula", TARGET3A,
        "--host", "P", "--dopant", "Si", "--compensating-element", "Li",
    ] + COMMON_ARGS
    run_cmd(f"目标 3A: Si 取代 P ({TARGET3A})", task3a_cmd)

    # 阶段 3B: 基于 3A 的最优结构做 S -> Se
    best_3a = OUTPUT_ROOT / TARGET3A / "best_structure.cif"
    if best_3a.exists():
        task3b_cmd = [
            sys.executable, SCRIPT,
            "--base-structure", str(best_3a),
            "--base-formula", TARGET3A,
            "--target-formula", TARGET3B,
            "--host", "S", "--dopant", "Se", "--compensating-element", "Li",
            "--scaling", "1,1,1", # 阶段B直接基于3A的最优超胞，不要再次扩胞
        ] + COMMON_ARGS
        run_cmd(f"目标 3B: Se 取代 S ({TARGET3B}, 基于 3A 最优结构)", task3b_cmd)
    else:
        print(f"⚠️ 跳过目标 3B: 找不到阶段 3A 的生成结果 {best_3a}")

    # =========================================================
    # 任务 4: 提醒无法直接生成
    # =========================================================
    print(f"\n{'='*60}")
    print("⚠️ 目标 4 未执行: Li9.54(Sn0.1Si0.9)1.74P1.44S11.1Br0.3O0.6")
    print("原因: 目标组成发生剧变，Cl 元素完全消失，不能基于目前的 Argyrodite 基体，")
    print("建议您去 Materials Project 下载 LGPS 相的基体结构后再单独掺杂。")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
