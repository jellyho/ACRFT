"""Every serve_policy invocation in the README has to parse.

The README carried three commands in a CLI that stopped existing when serve_patch_critic.py was
folded into serve_policy.py -- `--config/--checkpoint/--mode` instead of the `policy:checkpoint`
subcommand and `--critic-mode`. Copy-pasting one got "Argument --critic: expected 1 argument",
because `--critic` now takes the critic directory and was swallowing the next flag.

Documented commands are the ones people actually run, so they are worth parsing in CI. This only
parses -- no checkpoint is loaded and no GPU is touched.
"""

import pathlib
import re
import shlex
import sys

import pytest
import tyro

sys.argv = ["serve_policy.py"]
from scripts.serve_policy import Args  # noqa: E402

_README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
_CMD = re.compile(r"((?:uv run [^\n]*?scripts/serve_policy\.py)(?:[^\n]*\\\n)*[^\n]*)")


def _commands() -> list[list[str]]:
    out = []
    for raw in _CMD.findall(_README.read_text()):
        cmd = raw.replace("\\\n", " ")
        if cmd.count("'") % 2:  # the srun example lives inside bash -lc '...'
            cmd = cmd.rstrip().rstrip("'")
        argv = shlex.split(cmd)
        out.append(argv[argv.index("scripts/serve_policy.py") + 1 :])
    return out


def test_the_readme_still_shows_serve_commands():
    """A regex that quietly matches nothing would make every other test in here vacuous."""
    assert len(_commands()) >= 8


@pytest.mark.parametrize("argv", _commands(), ids=lambda a: " ".join(a[:4]))
def test_readme_command_parses(argv):
    tyro.cli(Args, args=argv)
