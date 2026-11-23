"""Mocked tests for the AWS SSM remote-command shell helper."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ssm.sh"


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
if [[ "$1 $2" == "sts get-caller-identity" ]]; then
  exit "${MOCK_STS_STATUS:-0}"
fi
if [[ "$1 $2" == "ssm send-command" ]]; then
  printf '%s' 'command-test'
  exit 0
fi
if [[ "$1 $2" == "ssm get-command-invocation" ]]; then
  if [[ " $* " == *" --query Status "* ]]; then
    printf '%s' "${MOCK_STATUS:-Success}"
  else
    printf 'remote stdout\nremote stderr\n'
  fi
  exit 0
fi
printf 'unexpected aws invocation: %s\n' "$*" >&2
exit 1
""",
    )
    _write_executable(
        mock_bin / "ada",
        """#!/bin/bash
printf 'ada %s\n' "$*" >> "$MOCK_LOG"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_LOG": str(tmp_path / "calls.log"),
            "SSM_INSTANCE_ID": "i-test",
            "AWS_REGION": "test-region",
            "SSM_POLL_INTERVAL": "0",
        }
    )
    return env


def _run(env, *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_ssm_script_has_valid_bash_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], check=False)
    assert result.returncode == 0


def test_remote_command_uses_configured_instance_region_and_timeout(tmp_path):
    env = _mock_environment(tmp_path)
    env["SSM_REMOTE_USER"] = "gpu-user"

    result = _run(env, "printf 'hello world'", "12")

    assert result.returncode == 0, result.stderr
    assert "remote stdout" in result.stdout
    assert "remote stderr" in result.stdout
    assert "[status: Success]" in result.stdout
    calls = Path(env["MOCK_LOG"]).read_text()
    assert "--region test-region" in calls
    assert "--instance-ids i-test" in calls
    assert "--instance-id i-test" in calls
    assert '"executionTimeout": ["12"]' in calls
    assert "sudo -u gpu-user -H bash -lc" in calls
    assert "ada credentials" not in calls


def test_missing_instance_id_fails_before_calling_aws(tmp_path):
    env = _mock_environment(tmp_path)
    env.pop("SSM_INSTANCE_ID")
    env.pop("IID", None)

    result = _run(env, "echo hello")

    assert result.returncode == 2
    assert "Set SSM_INSTANCE_ID" in result.stderr
    assert not Path(env["MOCK_LOG"]).exists()


def test_invalid_timeout_is_rejected(tmp_path):
    env = _mock_environment(tmp_path)

    result = _run(env, "echo hello", "not-a-number")

    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not Path(env["MOCK_LOG"]).exists()


def test_credential_refresh_uses_environment_configuration(tmp_path):
    env = _mock_environment(tmp_path)
    env.update(
        {
            "MOCK_STS_STATUS": "1",
            "ADA_ACCOUNT": "test-account",
            "ADA_PROVIDER": "test-provider",
            "ADA_ROLE": "test-role",
        }
    )

    result = _run(env, "echo hello")

    assert result.returncode == 0, result.stderr
    calls = Path(env["MOCK_LOG"]).read_text()
    assert (
        "ada credentials update --account test-account --provider test-provider "
        "--role test-role --once" in calls
    )
    assert "aws ssm send-command" in calls


def test_credential_refresh_requires_role_configuration(tmp_path):
    env = _mock_environment(tmp_path)
    env.update({"MOCK_STS_STATUS": "1", "ADA_ACCOUNT": "test-account"})
    env.pop("ADA_ROLE", None)

    result = _run(env, "echo hello")

    assert result.returncode != 0
    assert "ADA_ROLE" in result.stderr
    calls = Path(env["MOCK_LOG"]).read_text()
    assert "aws ssm send-command" not in calls


def test_remote_failure_is_reported_and_returned(tmp_path):
    env = _mock_environment(tmp_path)
    env["MOCK_STATUS"] = "Failed"

    result = _run(env, "echo hello")

    assert result.returncode == 1
    assert "[status: Failed]" in result.stdout
