# Telemetry infrastructure (Railway, `irc` project)

`laya.telemetry` sends traces, metrics and job logs to a `grafana/otel-lgtm` stack on Railway.

| Service | What it is | Reachable at |
|---|---|---|
| `otel-lgtm` | `grafana/otel-lgtm:0.30.2`: OTel collector, Prometheus, Loki, Tempo, Grafana; data on the `otel-lgtm-data` volume at `/data` | Grafana: `https://otel-lgtm-production-6693.up.railway.app` (login: `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` on the service). OTLP: `otel-lgtm.railway.internal:4317/4318` on the private network |
| `otlp-auth` | Railway Function, source in [`otlp-auth-proxy.ts`](otlp-auth-proxy.ts). Checks `Authorization: Bearer $OTLP_TOKEN` and forwards `POST /v1/{traces,metrics,logs}` to `otel-lgtm` over the private network | `https://otlp-auth-production.up.railway.app` |
| `mcp-grafana` | Grafana's MCP server (`mcp/grafana`), streamable HTTP | `mcp-grafana.railway.internal:8000/mcp`, private only |
| `keycloak` | Keycloak 26.4 in production mode on its own Postgres (`Postgres-iFvK`). Realm `mcp`, confidential client `grafana-mcp` (client credentials, 1-hour tokens), made by [`keycloak-setup.sh`](keycloak-setup.sh) | `https://keycloak-production-a492.up.railway.app` (admin: `admin` / `KC_BOOTSTRAP_ADMIN_PASSWORD` on the service) |
| `grafana-mcp-js` | [mcp-js](https://r33drichards.github.io/mcp-js/) `0.21.0-rc.2`: every request needs a Keycloak JWT (`JWKS_URL`); `mcp-grafana` is its upstream `grafana` server (`MCP_V8_MCP_CONFIG`) | `https://grafana-mcp-js-production.up.railway.app/mcp` |

## Sending data

The token is the `OTLP_TOKEN` variable on `otlp-auth`. Clients send it as a bearer header:

- **laya jobs on Modal** read `LAYA_OTLP_TOKEN` from the `laya-otel` Modal secret.
- **laya locally:** `export LAYA_OTLP_TOKEN=...`
- **Any other OTel SDK:** `OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-auth-production.up.railway.app` and
  `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20<token>`, using the OTLP/HTTP protocol (gRPC isn't proxied).

To rotate the token, set a new `OTLP_TOKEN` on `otlp-auth`, then update the `laya-otel` Modal secret
(`modal secret create --force laya-otel LAYA_OTLP_TOKEN=...`) and any other clients.

The old unauthenticated domain `otel-lgtm-production-ee87.up.railway.app` (port 4318 on `otel-lgtm`) still
accepts data. Remove it from `otel-lgtm` → Settings → Networking once its remaining clients use `otlp-auth`.

## Querying Grafana from Claude Code

The repo's [`.mcp.json`](../../.mcp.json) registers the `grafana-mcp-js` gateway as the `grafana` MCP server. Its
`headersHelper`, [`mcp-token.sh`](mcp-token.sh), gets a fresh Keycloak token on each connection. The gateway
exposes `run_js`, and Grafana's tools appear as `runjs__grafana__<tool>` stubs. Calling a stub returns
instructions: the tools actually run from JavaScript inside `run_js`, for example:

```js
const ds = await mcp.callTool("grafana", "list_datasources", {});
const r = await mcp.callTool("grafana", "query_prometheus",
  {datasourceUid: "prometheus", expr: "last_over_time(laya_eval_acc[1d])", queryType: "instant",
   startTime: "now", endTime: "now"});
console.log(r.content[0].text);
```

The helper reads the `grafana-mcp` client secret from a file, because Claude Code strips variables named
`*SECRET*`, `*TOKEN*`, and similar from a helper's environment. The file is `$GRAFANA_MCP_CLIENT_FILE`, or by default
`~/.config/laya/grafana-mcp-client`.

- **Locally:** put the secret in `~/.config/laya/grafana-mcp-client` (`chmod 600`), then approve the server when
  `claude` asks.
- **Claude Code on the web:** add `GRAFANA_MCP_CLIENT_SECRET=<secret>` to the cloud environment's variables and this
  to its setup script:
  `mkdir -p ~/.config/laya && printf %s "$GRAFANA_MCP_CLIENT_SECRET" > ~/.config/laya/grafana-mcp-client && chmod 600 ~/.config/laya/grafana-mcp-client`

To rotate the client secret, re-run `keycloak-setup.sh` with a new `MCP_CLIENT_SECRET`. It updates the existing
client. Then update the secret file or environment variable.

This works for Claude Code, not for claude.ai custom connectors. Those need OAuth discovery (RFC 9728
protected-resource metadata), which mcp-js doesn't serve.
