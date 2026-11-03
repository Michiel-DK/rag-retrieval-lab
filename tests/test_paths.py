"""Where results/ resolves. Falsifier for the paths change: nothing may land inside site-packages."""
import os
import subprocess
import sys
from pathlib import Path

from ragexp.paths import ENV_VAR, RESULTS_DIR, resolve_results_dir

REPO = Path(__file__).resolve().parent.parent


def test_env_var_wins(tmp_path):
    got = resolve_results_dir({ENV_VAR: str(tmp_path / "r")}, REPO / "ragexp" / "paths.py", tmp_path)
    assert got == (tmp_path / "r").resolve()


def test_blank_env_var_is_ignored(tmp_path):
    got = resolve_results_dir({ENV_VAR: "  "}, REPO / "ragexp" / "paths.py", tmp_path)
    assert got == REPO / "results"


def test_checkout_resolves_to_repo_results():
    assert RESULTS_DIR == REPO / "results"  # the committed evidence the notebooks read


def test_installed_copy_never_resolves_into_site_packages(tmp_path):
    fake = tmp_path / "lib" / "python3.12" / "site-packages" / "ragexp" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.touch()
    cwd = tmp_path / "work"
    cwd.mkdir()
    got = resolve_results_dir({}, fake, cwd)
    assert got == cwd.resolve() / "results"
    assert "site-packages" not in got.parts


def test_env_var_redirects_save_scores(tmp_path):
    # Subprocess, not reload: the constants bind at import.
    out = subprocess.run(
        [sys.executable, "-c", "from ragexp.runs import save_scores; print(save_scores('zz_paths_probe', {}))"],
        env={**os.environ, ENV_VAR: str(tmp_path)},
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert Path(out) == tmp_path.resolve() / "scores" / "zz_paths_probe.json"
    assert not (REPO / "results" / "scores" / "zz_paths_probe.json").exists()
