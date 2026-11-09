import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const origin = new URL(process.argv[2] ?? "http://127.0.0.1:18763");
assert(["127.0.0.1", "localhost"].includes(origin.hostname), "Local validation only");
const source = readFileSync(new URL("../../venn/playground.html", import.meta.url), "utf8");
const elements = new Map();
for (const id of ["result", "store", "store-status", "form", "state", "questions", "media"]) {
  elements.set(`#${id}`, {textContent: "", checked: false, value: "",
    addEventListener(event, callback) { this[event] = callback; }});
}
const buttons = ["nearest", "membership", "shift", "sweep"].map(space => ({
  dataset: {space}, addEventListener(event, callback) { this[event] = callback; }
}));
const context = vm.createContext({
  document: {
    querySelector(selector) { return elements.get(selector); },
    querySelectorAll() { return buttons; }
  },
  TextDecoder,
  async fetch(path, options) {
    const url = new URL(path, origin);
    assert.equal(url.origin, origin.origin);
    return fetch(url, options);
  },
});
vm.runInContext(source.match(/<script type="module">([\s\S]*?)<\/script>/)[1], context);
for (const button of buttons) {
  await button.click();
  const result = elements.get("#result").textContent;
  assert(!result.startsWith("Error:"), `${button.dataset.space}: ${result}`);
  if (button.dataset.space === "sweep") {
    const batches = result.trim().split("\n").map(line => JSON.parse(line));
    assert.equal(batches.flatMap(batch => batch.rows).length, 32);
  } else {
    assert(JSON.parse(result).answers);
  }
}
const basis = axis => Array.from({length: 768}, (_, i) => Number(i === axis));
const store = {vectors: [basis(0), basis(1)], ids: ["a", "b"], outcomes: ["account", "billing"]};
await elements.get("#store").change({target: {files: [{
  size: 10000, async text() { return JSON.stringify(store); }
}]}});
assert.match(elements.get("#store-status").textContent, /Registered 2 rows/);
await buttons[0].click();
assert.equal(JSON.parse(elements.get("#result").textContent).answers.nearest.answer, "account");
console.log("Playground JavaScript: four live API workflows and uploaded-store routing passed");
