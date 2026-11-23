#!/bin/bash
# Download the competition zip with parallel connections.
# Credentials come from an SSM SecureString containing either a bearer token or kaggle.json.
set -euo pipefail

REGION=${AWS_REGION:-us-east-2}
PARAMETER=${KAGGLE_SSM_PARAMETER:-/rsna-knee/kaggle-cred}
COMPETITION=${KAGGLE_COMPETITION:-rsna-knee-abnormality-detection}
OUT_DIR=${OUT_DIR:-/data/raw}
OUT_FILE=${OUT_FILE:-comp.zip}
CONNECTIONS=${CONNECTIONS:-16}

CREDENTIAL=$(aws ssm get-parameter \
  --region "$REGION" --name "$PARAMETER" --with-decryption \
  --query Parameter.Value --output text)
API="https://www.kaggle.com/api/v1/competitions/data/download-all/${COMPETITION}"
if [[ "$CREDENTIAL" == \{* ]]; then
  USERNAME=$(printf '%s' "$CREDENTIAL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["username"])')
  KEY=$(printf '%s' "$CREDENTIAL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')
  AUTH=(-u "$USERNAME:$KEY")
else
  AUTH=(-H "Authorization: Bearer $CREDENTIAL")
fi

URL=$(curl -s -o /dev/null -w '%{redirect_url}' "${AUTH[@]}" "$API")
if [[ "$URL" != https://* ]]; then
  printf '%s\n' "No HTTPS redirect URL; authentication failed:" >&2
  RESPONSE=$(curl -s "${AUTH[@]}" "$API" || true)
  printf '%s\n' "${RESPONSE:0:300}" >&2
  exit 1
fi

printf '%s\n' "Got signed URL"
mkdir -p "$OUT_DIR"
aria2c -c -x"$CONNECTIONS" -s32 -k64M --file-allocation=none \
  --summary-interval=30 --console-log-level=warn \
  -d "$OUT_DIR" -o "$OUT_FILE" "$URL"
ls -la "$OUT_DIR/$OUT_FILE"
