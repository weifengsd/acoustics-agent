# acoustics-agent 使用指南

本指南将介绍如何使用 `acoustics-agent` 框架（基于 `pyacoustics` 引擎）进行水声数值仿真。

## 1. 仿真结果可视化
`acoustics-agent` 提供了开箱即用的高质量可视化工具。下图展示了一个典型的深海射线追踪仿真结果及其声速剖面（SSP）。

![使用案例](usage_example.png)
*图 1：深海 Munk 声速剖面环境下的射线轨迹示例。左侧面板显示声速剖面，右侧面板显示声波的多径传播路径。*

## 2. 配置文件 (YAML)
`acoustics-agent` 使用 YAML 文件定义所有的仿真参数。一个典型的配置包含：

```yaml
project: "Munk 仿真示例"
frequency: 100.0
environment:
  ssp:
    type: "spline"
    data:
      - {depth: 0.0, c: 1548.52}
      - {depth: 1000.0, c: 1501.38}
      - {depth: 5000.0, c: 1551.91}
  bottom:
    type: "acousto-elastic"
    depth: 5000.0
    c_p: 1800.0
    density: 1.8
geometry:
  source:
    depths: [1000.0]
  receivers:
    ranges: [0.0, 50000.0]
    depths: [0.0, 5000.0]
solver:
  type: "bellhop"
  angles: [-20.0, 20.0]
  num_beams: 1000
```

## 2. 使用 Simulation API
`Simulation` 类负责加载配置、运行求解器以及绘图。

```python
from pyacoustics.simulation import Simulation

# 初始化
sim = Simulation("path/to/config.yaml")

# 运行射线追踪
rays = sim.run()

# 生成传播损失 (TL) 热力图
sim.plot_tl("tl_field.png")
```

## 3. 使用 Skills 技能库
`skills/` 目录包含了专为 AI Agent 调用的工具，开发者也可以直接使用：

- `make_env.py`: 从简单参数生成合法的 YAML 配置。
- `run_sim.py`: 运行仿真并返回射线路径数据。
- `plot_sim.py`: 生成可视化图表（射线轨迹或 TL 图）。

示例：
```python
from skills.make_env import make_env
from skills.run_sim import run_sim

# 创建环境
make_env("test.yaml", depth=4000, source_depth=100)

# 运行
rays = run_sim("test.yaml")
```

## 4. 自然语言编排 (Natural Language Orchestration)
`pyacoustics` 的核心优势之一是它对 AI Agent 极其友好。你不需要编写代码或手动编辑 YAML，只需使用自然语言描述场景，Agent 即可调用 Skills 库完成全流程仿真。

### 自然语言用例示例：
- **浅海多径仿真**："帮我做一个浅海仿真。水深 100 米，等声速 1500m/s，声源放在 10 米深，频率 500Hz。计算到 10 公里远并画出声线图。"
- **深海声道仿真**："使用 Munk 典型声速剖面跑一个深海仿真。声源放在 1000 米声道轴附近，计算 50 公里范围内的传播损失热力图。"
- **基准校验**："帮我计算一个 Pekeris 波导环境。50 米水深，刚性海底，声源在海面附近。"
- **自适应测试**："做一个 500 公里超远距离的深海仿真，频率 100Hz，帮我看看声线路径。"

## 5. 求解器说明
### PyBellhop (射线追踪)
目前的默认求解器。支持特性：
- **自适应步长**：根据仿真范围自动计算最优积分步长。
- **插值算法**：支持 `c-linear` (分段线性) 或 `spline` (平滑三次样条) 声速剖面。
- **边界条件**：支持 `vacuum` (真空), `rigid` (刚性), 以及 `acousto-elastic` (声弹性) 反射模型。

### 计算准确性
`acoustics-agent` 通过与权威的 Acoustics Toolbox 进行对标，确保了极高的计算精度。下图展示了原生 Python 引擎与传统 Fortran 版 AT 计算的传播损失 (TL) 的对比。

![准确性对比](calibK_TL_comparison.png)
*图 2：基于简正波 (Normal Mode) 分析计算的传播损失 (TL) 准确性对比。原生 Python 实现 (PyKraken) 与传统 Acoustics Toolbox (Kraken) 的计算结果高度吻合。*

### 传统求解器 (可选)
为了与原始的 Fortran 版 Acoustics Toolbox (AT) 进行对标校验，您可以切换到 legacy 模式：

1. 安装 AT 并设置 `AT_BIN_PATH` 环境变量。
2. 调用 `.run(mode="legacy")`。

```python
sim = Simulation("config.yaml")
# 调用传统的 Bellhop 可执行程序运行
results = sim.run(mode="legacy")
```
