// Headless end-to-end check of the page: serve web-demo/ over HTTP, open it in Chromium, load a model folder,
// upload an image, run the default questions and print the answers as JSON on the last line of stdout.
//
//   node smoke_test.mjs --models ./models/laya-vision/ --image test.png [--variant q8] [--backend wasm]
//
// The pinned jsDelivr URLs are answered from node_modules (npm install here first), so the test needs no network
// and fails if the installed versions drift from the pinned ones. Chromium comes from PLAYWRIGHT_BROWSERS_PATH
// (or --chrome); nothing is downloaded. WebGPU is usually unavailable headless, so the WASM path is what runs.
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, statSync, createReadStream } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { chromium } from "playwright-core";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const ROOT = resolve(new URL(".", import.meta.url).pathname);
const PINS = {
  "onnxruntime-web@1.30.0": "onnxruntime-web",
  "@huggingface/tokenizers@0.2.0": "@huggingface/tokenizers",
};
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
                ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };

function findChrome() {
  if (args.chrome) return args.chrome;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || "/opt/pw-browsers";
  const dir = readdirSync(base).filter((d) => d.startsWith("chromium-")).sort().pop();
  return join(base, dir, "chrome-linux", "chrome");
}

const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "");
  const file = join(ROOT, path || "index.html");
  if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream", "content-length": statSync(file).size });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ executablePath: findChrome(), args: ["--enable-unsafe-webgpu"] });
const context = await browser.newContext();
await context.route("https://cdn.jsdelivr.net/npm/**", (route) => {
  const url = new URL(route.request().url());
  const m = url.pathname.match(/^\/npm\/((?:@[^/]+\/)?[^@/]+@[^/]+)\/(.*)$/);
  const pkg = m && PINS[m[1]];
  if (!pkg) return route.fulfill({ status: 404, body: `not pinned: ${url.pathname}` });
  const file = join(ROOT, "node_modules", pkg, m[2]);
  if (!existsSync(file)) return route.fulfill({ status: 404, body: `missing ${file}` });
  return route.fulfill({ status: 200, body: readFileSync(file), contentType: TYPES[extname(file)] || "application/octet-stream",
                         headers: { "access-control-allow-origin": "*" } });
});
const page = await context.newPage();
page.on("console", (m) => console.error("[page]", m.type(), m.text()));
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
page.on("worker", (w) => w.on("console", (m) => console.error("[worker]", m.text())));

const q = new URLSearchParams({ models: args.models || "./models/laya-vision/", variant: args.variant || "q8",
                                backend: args.backend || "wasm" });
const t0 = Date.now();
try {
  await page.goto(`${origin}/index.html?${q}`);
  if (args.image) await page.setInputFiles("#image", args.image);
  if (args.state !== undefined) await page.fill("#state", args.state);
  if (args.questions) await page.fill("#questions", readFileSync(args.questions, "utf8"));
  if (args.perms) await page.selectOption("#perms", args.perms);
  await page.click("#load");
  await page.waitForFunction(() => !document.getElementById("run").disabled || window.__lastError, null, { timeout: 900_000 });
  const loadStatus = await page.textContent("#model-status");
  console.error("[load]", loadStatus, `${((Date.now() - t0) / 1000).toFixed(1)} s`);
  if (await page.evaluate(() => window.__lastError)) throw new Error(await page.evaluate(() => window.__lastError));
  await page.waitForFunction(() => document.getElementById("preview").width > 0 || !document.getElementById("image").value);
  const t1 = Date.now();
  await page.click("#run");
  await page.waitForFunction(() => window.__lastResult || window.__lastError, null, { timeout: 900_000 });
  const err = await page.evaluate(() => window.__lastError);
  if (err) throw new Error(err);
  console.error("[run]", await page.textContent("#run-status"), `${((Date.now() - t1) / 1000).toFixed(1)} s wall`);
  if (args.screenshot) await page.screenshot({ path: args.screenshot, fullPage: true });
  console.log(JSON.stringify({ load: loadStatus, result: await page.evaluate(() => window.__lastResult) }));
} finally {
  await browser.close();
  server.close();
}
