# acoustics-agent

`acoustics-agent` 是一个 Python 原生水声仿真框架，提供 Bellhop 风格二维射线、Bellhop3D 三维传播、Kraken 简正波、声学后处理、Acoustics Toolbox（AT）文件读写，以及可复现的 AT/native 一致性验证工具。

[English README](README.md) | [中文使用指南](docs/usage_zh.md) | [架构说明](docs/architecture.md) | [验证指南](docs/validation.md)

## 主要能力

- 基于 NumPy、SciPy 和 Numba 的 Python 原生求解器。
- Bellhop 2D ray、eigenray、arrival、coherent、incoherent 和 semi-coherent TL。
- Bellhop3D 射线、到达结构、三维地形和接收网格 TL。
- Kraken/Field 简正波及声场合成。
- 支持 `.env`、`.flp`、`.mod`、`.shd`、`.ati`、`.bty`、`.ssp`、`.irc`、`.trc`、`.brc`、`.sbp`、`.ray`、`.arr` 和 `.prt` 等 AT 格式。
- 可选调用本机 AT 可执行程序进行真实一致性验证。
- Delay-and-sum、Doppler 和 matched-filter 后处理。
- YAML、Python API、命令行和 Agent Skills 多种入口。

## 安装

需要 Python 3.10 或更高版本：

```bash
python -m pip install -e .
```

开发环境：

```bash
python -m pip install -e ".[dev]"
```

AT 不是运行 native 求解器的必需依赖。需要 parity 验证时配置：

```bash
export AT_BIN_PATH=/path/to/AcousticsToolbox/bin
```

## 快速开始

命令行运行示例：

```bash
acoustics-agent run configs/example.yaml --ray-output output/example.ray --no-plot
```

Python API：

```python
from pyacoustics.simulation import Simulation

simulation = Simulation("configs/example.yaml")
rays = simulation.run(mode="native")
tl = simulation.compute_coherent_tl(num_r=200, num_z=100)
```

## 验证

不依赖 AT 的快速检查：

```bash
pytest tests/unit -q
```

具备 AT 二进制时运行主 parity matrix：

```bash
acoustics-at-benchmark --suite all --at-bin-path "$AT_BIN_PATH" --strict-at
```

最近维护的基线覆盖 Bellhop 101 行、Kraken/Field 22 行、Bellhop3D 21 行、Field3D runtime 3 行、Postprocessing 11 行和 AT 文件格式 1192 行。该数字是历史基准证据，发布前仍应重新运行门禁。详细范围见 [验证指南](docs/validation.md)。

## 仓库结构

- `pyacoustics/`：求解器、AT I/O、benchmark 和后处理。
- `configs/`：可复用 YAML 场景。
- `skills/`：面向 Agent 的配置、运行和绘图辅助工具。
- `tests/`：单元测试、集成测试和必要数值夹具。
- `docs/`：使用、架构、验证与发布说明。
- `tools/`：仓库维护工具。

生成的 `reports/`、`scratch/`、构建产物、求解器输出和测试截图不进入版本控制。

## 引用要求

**科研引用要求：**论文、技术报告、基准测试、数据集及其他科研成果，如果实质性使用了 `acoustics-agent` 或由本项目生成的结果，应引用本软件仓库。可使用 GitHub 的 **Cite this repository** 功能，或按照 [CITATION.cff](CITATION.cff) 中的元数据引用。

重新分发源代码或二进制文件时，还必须按照 [LICENSE](LICENSE) 保留版权和 MIT 许可声明。科研引用要求用于说明项目的署名期望，不改变 MIT License 的许可条款。

## 参与贡献与许可

提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按照 [SECURITY.md](SECURITY.md) 私下报告。

Python 实现采用 [MIT License](LICENSE)。AT 是可选外部依赖，不随本项目分发；少量兼容性测试夹具采用独立许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
