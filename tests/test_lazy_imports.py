"""Tests for lightweight package imports."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_python(code):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_import_remag_is_lightweight():
    output = run_python(
        """
import json
import sys
import remag

modules = [
    "remag.cli",
    "remag.core",
    "torch",
    "sklearn",
    "igraph",
    "leidenalg",
    "pandas",
    "numpy",
    "scipy",
]
print(json.dumps({name: name in sys.modules for name in modules}))
"""
    )
    loaded = json.loads(output)

    assert loaded == {name: False for name in loaded}


def test_package_main_cli_export_does_not_import_core():
    output = run_python(
        """
import json
import sys
from remag import main_cli

print(json.dumps({
    "main_cli_callable": callable(main_cli),
    "remag.cli": "remag.cli" in sys.modules,
    "remag.core": "remag.core" in sys.modules,
    "torch": "torch" in sys.modules,
}))
"""
    )
    loaded = json.loads(output)

    assert loaded == {
        "main_cli_callable": True,
        "remag.cli": True,
        "remag.core": False,
        "torch": False,
    }
