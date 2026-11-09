import assert from "node:assert/strict";
import { OpenRouter } from "@openrouter/sdk";
import { HTTPClient } from "@openrouter/sdk/lib/http.js";
import * as Schema from "effect/Schema";
import * as OpenRouterSchema from "@effect/ai-openrouter/OpenRouterSchema";

const endpoint = process.argv[2];
assert(endpoint?.startsWith("http://127.0.0.1:"), "Use a local test endpoint");
const request = {
  model: "venn-1",
  state: { ticket: "The sign-in page is broken", tier: "paid" },
  questions: {
    team: {
      type: "choice",
      instructions: "Which team owns the ticket?",
      criteria: { account: "Login and account access", billing: "Payments and refunds" },
    },
    defect: { type: "noul", instructions: "Is there a software defect?" },
    urgency: {
      type: "score",
      instructions: "How urgent is the request?",
      criteria: ["Can wait", "Should be fixed soon", "Blocking the customer"],
    },
  },
};
Schema.decodeUnknownSync(OpenRouterSchema.DecisionsRequest)(request);
const raw = await fetch(`${endpoint}/api/alpha/decisions`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(request),
});
assert.equal(raw.status, 200);
const response = await raw.json();
Schema.decodeUnknownSync(OpenRouterSchema.DecisionsResponse)(response);
const httpClient = new HTTPClient({ fetcher: (input) => {
  assert.equal(new URL(input.url).origin, endpoint, "Contract traffic must stay local");
  return fetch(input);
} });
const sdk = new OpenRouter({ apiKey: "local-contract-only", httpClient });
const result = await sdk.alpha.decisions.create(
  { decisionsRequest: request }, { serverURL: endpoint },
);
assert.equal(result.model, "venn-1");
const standardAnswers = Object.fromEntries(Object.entries(response.answers)
  .map(([name, { venn, ...answer }]) => [name, answer]));
assert.deepEqual(result.answers, standardAnswers);
assert.equal(result.answers.defect.type, "noul");
assert(result.answers.defect.noul >= 0 && result.answers.defect.noul <= 1);
const score = result.answers.urgency;
assert.equal(score.type, "score");
assert(Math.abs(score.score - Object.entries(score.probabilities)
  .reduce((sum, [level, p]) => sum + Number(level) * p, 0)) < 1e-6);
console.log(JSON.stringify({ typescript_sdk: "1.5.33", effect_schemas: "4.1.2",
  endpoint, passed: ["choice", "noul", "score", "request_schema", "response_schema"] }));
