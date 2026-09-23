// Node checks for laya.js, driven by tests/test_web_demo.py (needs `npm install` here for @huggingface/tokenizers).
//   node test_parity.mjs ids <dir with tokenizer.json + tokenizer_config.json>
//   node test_parity.mjs pixels <raw HxWx3 uint8 file> <H> <W> <out float32 file>
import { readFileSync, writeFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildInputs, permutations, pixelValues, prefixIds, stateText, toInternal, renderOptions } from "./laya.js";

const [mode, ...args] = process.argv.slice(2);
const fixture = JSON.parse(readFileSync(new URL("./fixtures/parity.json", import.meta.url)));
const cfg = fixture.cfg;

if (mode === "ids") {
  const tj = JSON.parse(readFileSync(`${args[0]}/tokenizer.json`));
  const tc = JSON.parse(readFileSync(`${args[0]}/tokenizer_config.json`));
  const tok = new Tokenizer(tj, tc);
  const encode = (s) => tok.encode(s, { add_special_tokens: false }).ids;
  let n = 0, bad = 0;
  for (const c of fixture.cases) {
    const prefix = prefixIds(encode, cfg, c.n_images);
    if (JSON.stringify(prefix) !== JSON.stringify(c.prefix_ids)) { console.log(`${c.name}: prefix differs`); bad++; }
    const text = stateText(c.state);
    for (const r of c.rows) {
      const q = toInternal(c.questions[r.qid]);
      const order = permutations(renderOptions(q).length, 2)[r.order[0] === 0 ? 0 : 1];
      const got = buildInputs(encode, cfg, prefix, text, q, order);
      n++;
      for (const key of ["ids", "markers", "option_span"]) {
        if (JSON.stringify(got[key]) !== JSON.stringify(r[key])) {
          bad++;
          const i = got[key].findIndex((v, j) => v !== r[key][j]);
          console.log(`${c.name}/${r.qid}/${r.order}: ${key} differ at ${i} (js ${got[key].length}, py ${r[key].length})`);
          console.log("  js", JSON.stringify(got[key].slice(Math.max(0, i - 3), i + 5)), "py", JSON.stringify(r[key].slice(Math.max(0, i - 3), i + 5)));
        }
      }
    }
  }
  if (bad) { console.log(`${bad} mismatches over ${n} rows`); process.exit(1); }
  console.log(`${n} rows match`);
} else if (mode === "pixels") {
  const [path, h, w, out] = [args[0], +args[1], +args[2], args[3]];
  const rgb = readFileSync(path);
  const rgba = new Uint8Array(h * w * 4);
  for (let i = 0; i < h * w; i++) { rgba.set(rgb.subarray(3 * i, 3 * i + 3), 4 * i); rgba[4 * i + 3] = 255; }
  const { data } = pixelValues(rgba, h, w, cfg);
  writeFileSync(out, Buffer.from(data.buffer));
} else {
  console.log("usage: node test_parity.mjs ids <tokenizer dir> | pixels <rgb> <H> <W> <out>");
  process.exit(2);
}
