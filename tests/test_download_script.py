"""Mocked tests for the competition download shell helper."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download.sh"


def _write_executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _mock_environment(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    _write_executable(
        mock_bin / "aws",
        """#!/bin/bash
printf 'aws %s\n' "$*" >> "$MOCK_LOG"
printf '%s' "$MOCK_CREDENTIAL"
""",
    )
    _write_executable(
        mock_bin / "curl",
        """#!/bin/bash
printf 'curl %s\n' "$*" >> "$MOCK_LOG"
if [[ " $* " == *" %{redirect_url} "* ]]; then
  printf '%s' "${MOCK_REDIRECT:-}"
else
  printf '%s' 'mock authentication failure'
fi
""",
    )
    _write_executable(
        mock_bin / "aria2c",
        """#!/bin/bash
printf 'aria2c %s\n' "$*" >> "$MOCK_LOG"
out_dir=
out_file=
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) out_dir=$2; shift 2 ;;
    -o) out_file=$2; shift 2 ;;
    *) shift ;;
  esac
done
: > "$out_dir/$out_file"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_LOG": str(tmp_path / "calls.log"),
            "MOCK_CREDENTIAL": "fake-token",
            "MOCK_REDIRECT": "https://signed.example/competition.zip",
            "AWS_REGION": "test-region",
            "KAGGLE_SSM_PARAMETER": "/test/kaggle-credential",
            "KAGGLE_COMPETITION": "test-competition",
            "OUT_DIR": str(tmp_path / "output"),
            "CONNECTIONS": "4",
        }
    )
    return env


def _run(env):
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_download_script_has_valid_bash_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], check=False)
    assert result.returncode == 0


def test_bearer_download_uses_configured_region_parameter_and_output(tmp_path):
    env = _mock_environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    calls = Path(env["MOCK_LOG"]).read_text()
    assert "--region test-region" in calls
    assert "--name /test/kaggle-credential" in calls
    assert "Authorization: Bearer fake-token" in calls
    assert "test-competition" in calls
    assert "-x4" in calls
    assert f"-d {env['OUT_DIR']} -o comp.zip https://signed.example/competition.zip" in calls
    assert (Path(env["OUT_DIR"]) / "comp.zip").exists()


def test_json_credential_uses_legacy_basic_authentication(tmp_path):
    env = _mock_environment(tmp_path)
    env["MOCK_CREDENTIAL"] = '{"username":"test-user","key":"test-key"}'

    result = _run(env)

    assert result.returncode == 0, result.stderr
    calls = Path(env["MOCK_LOG"]).read_text()
    assert "-u test-user:test-key" in calls
    assert "Authorization: Bearer" not in calls


def test_missing_https_redirect_fails_before_download(tmp_path):
    env = _mock_environment(tmp_path)
    env["MOCK_REDIRECT"] = ""

    result = _run(env)

    assert result.returncode == 1
    assert "No HTTPS redirect URL" in result.stderr
    assert "mock authentication failure" in result.stderr
    calls = Path(env["MOCK_LOG"]).read_text()
    assert "aria2c" not in calls
