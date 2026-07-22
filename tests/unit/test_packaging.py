from pathlib import Path
import tomllib


def test_pyproject_lists_all_pyacoustics_packages():
    root = Path(__file__).resolve().parents[2]
    configured = set(
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"]
    )
    discovered = {
        init_path.parent.relative_to(root).as_posix().replace("/", ".")
        for init_path in (root / "pyacoustics").rglob("__init__.py")
    }

    assert discovered <= configured
