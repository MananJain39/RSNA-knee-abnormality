#!/bin/bash
# Run a command on a GPU host through AWS SSM. Usage: ssm.sh "<cmd>" [timeout_s]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s "<cmd>" [timeout_s]\n' "$0" >&2
  exit 2
fi

INSTANCE_ID=${SSM_INSTANCE_ID:-${IID:-}}
REGION=${AWS_REGION:-${R:-us-east-2}}
TIMEOUT=${2:-600}
REMOTE_USER=${SSM_REMOTE_USER:-ubuntu}
POLL_INTERVAL=${SSM_POLL_INTERVAL:-2}

if [[ -z "$INSTANCE_ID" ]]; then
  printf '%s\n' 'Set SSM_INSTANCE_ID (or IID) to the target EC2 instance.' >&2
  exit 2
fi
if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'The timeout must be a positive integer number of seconds.' >&2
  exit 2
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  : "${ADA_ACCOUNT:?Set ADA_ACCOUNT before refreshing credentials.}"
  : "${ADA_ROLE:?Set ADA_ROLE before refreshing credentials.}"
  ADA_PROVIDER=${ADA_PROVIDER:-conduit}
  ada credentials update --account "$ADA_ACCOUNT" --provider "$ADA_PROVIDER" \
    --role "$ADA_ROLE" --once >/dev/null
fi

PARAMS=$(python3 -c '
import json
import shlex
import sys

command = f"sudo -u {shlex.quote(sys.argv[3])} -H bash -lc {shlex.quote(sys.argv[1])}"
print(json.dumps({"commands": [command], "executionTimeout": [sys.argv[2]]}))
' "$1" "$TIMEOUT" "$REMOTE_USER")
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript --parameters "$PARAMS" --timeout-seconds 60 \
  --query Command.CommandId --output text)

STATUS=Unknown
for ((attempt = 0; attempt < TIMEOUT / 2 + 30; attempt++)); do
  STATUS=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CID" \
    --instance-id "$INSTANCE_ID" --query Status --output text 2>/dev/null || true)
  case "$STATUS" in
    Success | Failed | Cancelled | TimedOut) break ;;
  esac
  sleep "$POLL_INTERVAL"
done

aws ssm get-command-invocation --region "$REGION" --command-id "$CID" \
  --instance-id "$INSTANCE_ID" --query '[StandardOutputContent,StandardErrorContent]' \
  --output text | tail -c 6000
printf '[status: %s]\n' "$STATUS"
[[ "$STATUS" == Success ]]
