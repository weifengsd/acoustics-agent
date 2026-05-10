# CLAUDE.md - pyacoustics 开发指南

本文件为开发人员和 AI 助手提供 `pyacoustics` 代码库的操作指南。

## 🛠 编译与测试命令
- **运行所有测试**: `pytest tests/`
- **运行特定测试**: `pytest tests/Munk/test_munk.py`
- **性能评估**: `python3 scripts/benchmark.py`
- **开发模式安装**: `pip install -e .`

## 🎨 代码风格与规范
- **命名规范**: 函数和变量使用 `snake_case`，类名使用 `PascalCase`。
- **类型注解**: 使用 Python 3.10+ 的类型提示。
- **物理核心**: `pyacoustics/solvers/` 中的核心算法通常使用 **Numba** (`@njit`)。
  - 避免在 JIT 函数内部创建对象（类或字典）。
  - 所有数值计算应使用 NumPy 数组。
  - 确保所有类型对 Numba 可推导（避免混合类型的列表）。
- **配置系统**: 统一使用 `pyacoustics/schema.py` 中的 `dataclasses`。核心逻辑应避免依赖 Pydantic。
- **格式化**: 遵循 PEP 8 标准。

## 🧱 架构概览
- **引擎入口**: `pyacoustics/simulation.py` 是主要入口类。
- **配置加载**: `pyacoustics/config.py` 处理 YAML 到 dataclass 的映射。
- **技能库**: AI 工具库位于 `skills/`，应保持高抽象层级，方便 LLM 调用。
- **测试驱动**: 每一个新环境或新功能都必须在 `tests/` 下有对应的测试目录。

## 🤝 协作流程
1. 使用 `skills/make_env.py` 生成新的测试场景。
2. 使用 `scripts/benchmark.py` 验证性能。
3. 在推送到 `main` 分支前，确保 100% 的测试通过率。
