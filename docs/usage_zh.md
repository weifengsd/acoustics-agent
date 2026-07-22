# 使用指南

本文介绍稳定的公开入口。各求解器参数和 AT 兼容细节见 [acoustics-agent_user_manual.md](acoustics-agent_user_manual.md)。

## 配置场景

场景采用 YAML，并直接映射到 `SimulationConfig`：

```yaml
project: Example waveguide
frequency: 100.0
environment:
  ssp:
    type: c-linear
    data:
      - {depth: 0.0, c: 1500.0}
      - {depth: 200.0, c: 1500.0}
  surface:
    type: vacuum
    depth: 0.0
  bottom:
    type: rigid
    depth: 200.0
geometry:
  source:
    depths: [100.0]
  receivers:
    ranges: [500.0, 1000.0]
    depths: [50.0, 100.0, 150.0]
solver:
  type: bellhop
  angles: [-10.0, 10.0]
  num_beams: 101
  step_size: 5.0
```

最小可运行配置位于 `configs/example.yaml`。

## 命令行

```bash
acoustics-agent run configs/example.yaml --ray-output output/example.ray --no-plot
```

常用参数：

- `--mode native|legacy`：选择 Python 原生求解器或本机 AT。
- `--at-bin-path PATH`：指定 legacy 模式使用的 AT 二进制目录。
- `--ray-output PATH`：写出 AT-compatible ray 文件。
- `--tl-output PATH`：计算并绘制 coherent TL。
- `--no-plot`：不弹出交互图。

## Python API

```python
import numpy as np

from pyacoustics.simulation import Simulation

sim = Simulation("configs/example.yaml")
rays = sim.run(mode="native")

ranges = np.linspace(100.0, 1000.0, 40)
depths = np.linspace(0.0, 200.0, 30)
coherent = sim.compute_coherent_tl(r_grid=ranges, z_grid=depths)
incoherent = sim.compute_incoherent_tl(r_grid=ranges, z_grid=depths)
semicoherent = sim.compute_semicoherent_tl(r_grid=ranges, z_grid=depths)
```

Arrivals 和 eigenrays：

```python
arrivals = sim.compute_arrival_grid(method="beam")
eigenrays = sim.run_eigenrays(range_m=1000.0, depth_m=100.0)
```

## AT 一致性验证

Native 仿真不依赖 AT。需要真实对比时运行：

```bash
export AT_BIN_PATH=/path/to/AcousticsToolbox/bin
acoustics-at-benchmark --suite all --at-bin-path "$AT_BIN_PATH" --strict-at
```

Benchmark 默认写入被 Git 忽略的 `reports/at_benchmark/`，也可通过 `--output-dir` 指定其他位置。

## Agent Skills

`skills/` 提供创建环境、执行仿真和绘图的小型辅助函数。它们与 Python 调用共享同一份 schema 和 `Simulation` API，不维护第二套求解逻辑。
