# Telemetry infrastructure (Railway, `irc` project)

`laya.telemetry` sends traces, metrics and job logs to a `grafana/otel-lgtm` stack on Railway.

| Service | What it is | Reachable at |
|---|---|---|
| `otel-lgtm` | `grafana/otel-lgtm:0.30.2`: OTel collector, Prometheus, Loki, Tempo, Grafana; data on the `otel-lgtm-data` volume at `/data` | Grafana: `https://otel-lgtm-production-6693.up.railway.app` (login: `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` on the service). OTLP: `otel-lgtm.railway.internal:4317/4318` on the private network |
| `otlp-auth` | Railway Function, source in [`otlp-auth-proxy.ts`](otlp-auth-proxy.ts). Checks `Authorization: Bearer $OTLP_TOKEN` and forwards `POST /v1/{traces,metrics,logs}` to `otel-lgtm` over the private network | `https://otlp-auth-production.up.railway.app` |
| `mcp-grafana` | Grafana's MCP server (`mcp/grafana`), streamable HTTP | `mcp-grafana.railway.internal:8000/mcp`, private only |

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
