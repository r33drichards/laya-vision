#!/usr/bin/env bash
# Create (or update) the "mcp" realm and the confidential "grafana-mcp" client on the irc project's Keycloak.
# Idempotent. Needs: KC_URL, KC_ADMIN_PASSWORD (the service's KC_BOOTSTRAP_ADMIN_PASSWORD), MCP_CLIENT_SECRET.
#   KC_URL=https://keycloak-production-a492.up.railway.app KC_ADMIN_PASSWORD=... MCP_CLIENT_SECRET=... \
#     bash infra/telemetry/keycloak-setup.sh
# Access tokens for the client last TOKEN_TTL seconds (default 3600); Claude Code mints a fresh one per connection
# through mcp-token.sh, so a short lifetime costs nothing.
set -euo pipefail
: "${KC_URL:?}" "${KC_ADMIN_PASSWORD:?}" "${MCP_CLIENT_SECRET:?}"
REALM=${REALM:-mcp}
CLIENT=${CLIENT:-grafana-mcp}
TOKEN_TTL=${TOKEN_TTL:-3600}

admin=$(curl -sf -X POST "$KC_URL/realms/master/protocol/openid-connect/token" \
  -d grant_type=password -d client_id=admin-cli -d username="${KC_ADMIN_USER:-admin}" \
  --data-urlencode password="$KC_ADMIN_PASSWORD" | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
api() { curl -sf -H "Authorization: Bearer $admin" -H 'Content-Type: application/json' "$@"; }

if api "$KC_URL/admin/realms/$REALM" >/dev/null 2>&1; then
  echo "realm $REALM exists"
else
  api -X POST "$KC_URL/admin/realms" -d "{\"realm\":\"$REALM\",\"enabled\":true}"
  echo "created realm $REALM"
fi

body=$(CLIENT="$CLIENT" SECRET="$MCP_CLIENT_SECRET" TTL="$TOKEN_TTL" python3 -c '
import json, os
print(json.dumps({
    "clientId": os.environ["CLIENT"], "enabled": True, "protocol": "openid-connect",
    "publicClient": False, "clientAuthenticatorType": "client-secret", "secret": os.environ["SECRET"],
    "serviceAccountsEnabled": True, "standardFlowEnabled": False, "directAccessGrantsEnabled": False,
    "implicitFlowEnabled": False, "attributes": {"access.token.lifespan": os.environ["TTL"]},
}))')
id=$(api "$KC_URL/admin/realms/$REALM/clients?clientId=$CLIENT" | python3 -c 'import sys,json;c=json.load(sys.stdin);print(c[0]["id"] if c else "")')
if [ -n "$id" ]; then
  api -X PUT "$KC_URL/admin/realms/$REALM/clients/$id" -d "$body"
  echo "updated client $CLIENT"
else
  api -X POST "$KC_URL/admin/realms/$REALM/clients" -d "$body"
  echo "created client $CLIENT"
fi
echo "JWKS: $KC_URL/realms/$REALM/protocol/openid-connect/certs"
