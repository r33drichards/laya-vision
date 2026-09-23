// OTLP/HTTP auth proxy (Railway Function "otlp-auth" in the irc project).
// Accepts POST /v1/{traces,metrics,logs} with "Authorization: Bearer $OTLP_TOKEN" and forwards the body to the
// otel-lgtm collector over the private network. Everything else gets 401/404. GET /healthz is open.
import { timingSafeEqual } from "node:crypto";

const TOKEN = Bun.env.OTLP_TOKEN ?? "";
const UPSTREAM = (Bun.env.OTLP_UPSTREAM ?? "http://otel-lgtm.railway.internal:4318").replace(/\/+$/, "");
const PATHS = new Set(["/v1/traces", "/v1/metrics", "/v1/logs"]);
const FORWARD = ["content-type", "content-encoding"];
const enc = new TextEncoder();

function authorized(header: string | null): boolean {
  if (!TOKEN || !header || !header.startsWith("Bearer ")) return false;
  const got = enc.encode(header.slice(7).trim());
  const want = enc.encode(TOKEN);
  return got.length === want.length && timingSafeEqual(got, want);
}

Bun.serve({
  port: Number(Bun.env.PORT ?? 8080),
  hostname: "::",
  maxRequestBodySize: 64 * 1024 * 1024,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/healthz") return new Response("ok");
    if (!PATHS.has(url.pathname)) return new Response("not found", { status: 404 });
    if (req.method !== "POST") return new Response("method not allowed", { status: 405 });
    if (!authorized(req.headers.get("authorization"))) {
      return new Response("unauthorized", { status: 401, headers: { "www-authenticate": "Bearer" } });
    }
    const headers = new Headers();
    for (const h of FORWARD) {
      const v = req.headers.get(h);
      if (v) headers.set(h, v);
    }
    try {
      const up = await fetch(UPSTREAM + url.pathname, { method: "POST", headers, body: await req.arrayBuffer() });
      return new Response(up.body, {
        status: up.status,
        headers: { "content-type": up.headers.get("content-type") ?? "application/x-protobuf" },
      });
    } catch (e) {
      console.error("upstream error:", e);
      return new Response("bad gateway", { status: 502 });
    }
  },
});
console.log(`otlp-auth: forwarding to ${UPSTREAM}, token ${TOKEN ? "set" : "MISSING (all requests rejected)"}`);
