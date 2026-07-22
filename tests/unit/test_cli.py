import pytest

from pyacoustics.cli import main


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "acoustics-agent" in capsys.readouterr().out


def test_cli_run_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])

    assert exc.value.code == 0
    assert "--ray-output" in capsys.readouterr().out
