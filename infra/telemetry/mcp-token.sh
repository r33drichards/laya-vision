#!/usr/bin/env bash
# Claude Code headersHelper for the grafana MCP gateway (.mcp.json): mints a short-lived Keycloak access token with
# the client-credentials grant and prints {"Authorization": "Bearer <jwt>"}. Claude Code runs it on each connection
# and again on a 401.
#
# Claude Code strips variables named like *SECRET*/*TOKEN*/*KEY*/*AUTH*/*PASSWORD* from a helper's environment, so the
# client secret is read from a file: $GRAFANA_MCP_CLIENT_FILE, else ~/.config/laya/grafana-mcp-client. In a Claude
# Code cloud environment, a setup script can write that file from an environment variable.
set -euo pipefail
KC_URL=${GRAFANA_MCP_KC_URL:-https://keycloak-production-a492.up.railway.app}
FILE=${GRAFANA_MCP_CLIENT_FILE:-$HOME/.config/laya/grafana-mcp-client}
[ -r "$FILE" ] || { echo "mcp-token.sh: no client secret at $FILE" >&2; exit 1; }
curl -sf -X POST "$KC_URL/realms/mcp/protocol/openid-connect/token" \
  -d grant_type=client_credentials -d client_id=grafana-mcp --data-urlencode "client_secret@$FILE" |
  python3 -c 'import sys,json;print(json.dumps({"Authorization":"Bearer "+json.load(sys.stdin)["access_token"]}))'
