// ゴールデンテスト：app/model.js の結果が、Python の基準の実装（app_reference.py）の期待値と一致するか。
// 実行: node tests/test_model.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const here = new URL("..", import.meta.url).pathname;
const { evaluate } = require(here + "app/model.js");
const ctx = { window: {} };
vm.runInNewContext(readFileSync(here + "app/data/app_data.js", "utf8"), ctx);
const data = ctx.window.APP_DATA;
const golden = JSON.parse(readFileSync(here + "tests/golden.json", "utf8"));
const TOL = 1e-6;

function compare(a, b, path, errors) {
  if (typeof b === "number" && typeof a === "number") {
    if (Math.abs(a - b) > TOL * Math.max(1, Math.abs(b))) errors.push(`${path}: ${a} ≠ ${b}`);
  } else if (b === null || typeof b !== "object") {
    if (a !== b) errors.push(`${path}: ${JSON.stringify(a)} ≠ ${JSON.stringify(b)}`);
  } else if (Array.isArray(b)) {
    if (!Array.isArray(a) || a.length !== b.length) errors.push(`${path}: 長さが違う`);
    else b.forEach((v, i) => compare(a[i], v, `${path}[${i}]`, errors));
  } else {
    if (a === null || typeof a !== "object") { errors.push(`${path}: オブジェクトでない`); return; }
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) compare(a[k], b[k], `${path}.${k}`, errors);
  }
}

let failed = 0;
for (const c of golden) {
  const errors = [];
  compare(evaluate(c.input, data), c.expected, c.name, errors);
  if (errors.length) { failed++; console.log(`NG ${c.name}\n  ` + errors.slice(0, 5).join("\n  ")); }
  else console.log(`OK ${c.name}`);
}
// サンプル3名がデータに入っているか
if (!data.samples || data.samples.length !== 3) { failed++; console.log("NG サンプルが3名そろっていない"); }
console.log(`\n${failed ? "NG" : "OK"}：ゴールデンテスト ${golden.length} 件（許容 ${TOL}）、失敗 ${failed} 件`);
process.exit(failed ? 1 : 0);
