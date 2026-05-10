# acoustics-agent: AI 原生水声仿真框架

`acoustics-agent` 是一个现代化的水声仿真框架，旨在结合高性能物理计算与无缝的 AI 编排。其核心引擎为 **pyacoustics** —— 这是一个对传统 Acoustics Toolbox 的高性能纯 Python 重构。

**官方网站**: [https://uacomm.com](https://uacomm.com)

## 🚀 核心特性
- **高性能**：基于 Numba 加速的 **pyacoustics** 引擎，射线追踪速度可达 1000+ 条射线/秒。
- **AI 原生**：内置模块化的 "Skills" 库，专为大语言模型 (LLM) 的自动化编排而设计。
- **零依赖**：在标准的 Anaconda/Miniconda 环境下即可开箱即用，无需配置复杂的 Fortran 编译器。
- **现代化工作流**：基于 YAML 的配置系统，彻底告别 legacy 系统繁琐的二进制或文本格式。
- **兼容性**：原生支持调用传统的 **Acoustics Toolbox (AT)** 二进制文件，便于结果校验与对比。
- **完备的测试集**：包含 27+ 个集成基准测试，涵盖 Munk, Pekeris, RAP 等经典声学场景。

## 📁 项目结构
- `pyacoustics/`: 核心物理引擎与求解器 (MIT 许可证)。
- `skills/`: AI 编排层 (make_env, run_sim, plot_sim)。
- `tests/`: 详尽的测试套件与环境基准。
- `docs/`: 详细的设计与使用说明文档。

## 🛠 安装
项目设计为在标准科学计算环境上“零额外依赖”。
```bash
# 推荐：标准的 Anaconda 环境
conda install numpy scipy numba pyyaml matplotlib
```
或者通过 pip 安装：
```bash
pip install -e .
```

### 可选的外部依赖
若需与传统的 Bellhop/Kraken 保持完全的功能对齐，您可以可选地配置 **Acoustics Toolbox (AT)**：
1. 从 [官方源](http://oalib.hlsresearch.com/AcousticsToolbox/) 下载或编译 AT 二进制文件。
2. 设置 `AT_BIN_PATH` 环境变量指向您的 `at/bin` 目录。
   ```bash
   export AT_BIN_PATH=/path/to/at/bin
   ```

## 📖 快速上手
您可以使用内置的 Skills 库或高级 Simulation API 运行仿真：

```python
from pyacoustics.simulation import Simulation

# 从 YAML 配置文件运行
sim = Simulation("tests/Munk/munk.yaml")
sim.run()

# 绘制结果
sim.plot_tl("output.png")
```

## 🧪 测试
运行完整的测试套件（27+ 个测试）：
```bash
pytest tests/
```
